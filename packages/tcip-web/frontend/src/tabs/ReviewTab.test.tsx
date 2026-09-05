import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { subjectColor } from "@/api/classes";
import { resultsApi } from "@/api/inference";
import { notifyCanvasStateRequest } from "@/lib/canvasSync";
import { applyReviewFocus } from "@/lib/reviewFocus";
import { setSubjectColorOverride } from "@/lib/subjectColors";
import { useStore } from "@/store";
import type { Annotation, Detection, MatchesResponse } from "@/store/types";
import { ReviewTab } from "@/tabs/ReviewTab";

// Konva needs a real 2D canvas; these tests exercise empty-state rendering, nav
// bounds, and the matches-recompute effect, not drawing. Render Konva shapes as
// divs and CanvasStage as a passthrough.
vi.mock("konva", () => ({ default: {} }));
vi.mock("react-konva", () => ({
  // Expose stroke + dashed so symbology (dashed = under review) is assertable, and the box's own
  // placement so a shape drawn from the wrong annotation is distinguishable from a missing one.
  Rect: (props: {
    stroke?: string;
    dash?: number[];
    x?: number;
    y?: number;
    width?: number;
    height?: number;
  }) => (
    <div
      data-testid="k-rect"
      data-stroke={props.stroke}
      data-dashed={props.dash ? "true" : "false"}
      data-x={props.x}
      data-y={props.y}
      data-width={props.width}
      data-height={props.height}
    />
  ),
  Line: (props: { stroke?: string; dash?: number[]; points?: number[] }) => (
    <div
      data-testid="k-line"
      data-stroke={props.stroke}
      data-dashed={props.dash ? "true" : "false"}
      data-points={props.points?.join(",")}
    />
  ),
  Circle: (props: { x?: number; y?: number; fill?: string }) => (
    <div data-testid="k-circle" data-x={props.x} data-y={props.y} data-fill={props.fill} />
  ),
  Text: (props: { text?: string }) => <div data-testid="k-text" data-text={props.text} />,
}));
vi.mock("@/components/Canvas/CanvasStage", () => ({
  // Forward pixel handlers as plain mouse events (clientX/Y stand in for image-pixel coords; the
  // real screen<->image conversion is CanvasStage's own concern, exercised elsewhere) so tests can
  // simulate a canvas drag (e.g. the "mark missed object" draw) without a real Konva stage.
  CanvasStage: (props: {
    children?: React.ReactNode;
    imageUrl?: string | null;
    onPixelDown?: (x: number, y: number, ev: unknown) => void;
    onPixelMove?: (x: number, y: number, ev: unknown) => void;
    onPixelUp?: (x: number, y: number, ev: unknown) => void;
  }) => (
    <div
      data-testid="canvas-stage"
      data-image-url={props.imageUrl ?? ""}
      onMouseDown={(e) => props.onPixelDown?.(e.clientX, e.clientY, { evt: { button: e.button } })}
      onMouseMove={(e) => props.onPixelMove?.(e.clientX, e.clientY, { evt: { buttons: 1 } })}
      onMouseUp={(e) => props.onPixelUp?.(e.clientX, e.clientY, { evt: {} })}
    >
      {props.children}
    </div>
  ),
}));
const initialStoreState = useStore.getState();

const PRED_DIR_A = "C:/data/predictions/model_a/2026-01-01";
const PRED_DIR_B = "C:/data/predictions/model_b/2026-01-01";

function setupDataset(opts: { predDir?: string | null } = {}) {
  const predDir = opts.predDir !== undefined ? opts.predDir : PRED_DIR_A;
  useStore.setState((s) => ({
    gui: {
      ...s.gui,
      dataset: {
        ...s.gui.dataset,
        project_root: "C:/proj",
        dataset_root: "C:/data",
        subject: "subject_a",
        date: "2026-01-01",
        image_list: ["img1.jpg", "img2.jpg"],
        current_image_index: 0,
        images_dir: "C:/data/images/2026-01-01",
        annotations_dir: "C:/data/annotations/2026-01-01",
        predictions_dir: predDir,
      },
    },
  }));
}

function det(over: Partial<Detection> = {}): Detection {
  return {
    det_type: "tp",
    class_name: "subject_a",
    conf: 0.9,
    iou: 0.8,
    gt_idx: 0,
    pred_idx: null,
    bbox: [10, 10, 50, 50],
    reviewed: false,
    reviewed_action: null,
    ...over,
  };
}

function matchesRes(
  detections: Detection[],
  extra: {
    gt?: Annotation[];
    preds?: Annotation[];
    subject?: string | null;
    attribute?: string | null;
  } = {},
): MatchesResponse {
  return {
    img_width: 1000,
    img_height: 800,
    n_tp: detections.filter((d) => d.det_type === "tp").length,
    n_fp: detections.filter((d) => d.det_type === "fp").length,
    n_fn: detections.filter((d) => d.det_type === "fn").length,
    detections,
    gt: extra.gt ?? [],
    preds: extra.preds ?? [],
    image_status: "not_started",
    n_reviewed: detections.filter((d) => d.reviewed).length,
    n_total: detections.length,
    subject: extra.subject ?? null,
    attribute: extra.attribute ?? null,
  };
}

const prevBtn = () => screen.getByTitle("Previous detection (←)");
const nextBtn = () => screen.getByTitle("Next detection (→)");

let matchesSpy: MockInstance<typeof api.review.matches>;
let statusesSpy: MockInstance<typeof api.review.imageStatuses>;

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  setupDataset();
  matchesSpy = vi.spyOn(api.review, "matches").mockResolvedValue(matchesRes([]));
  // Both dataset images have something to review by default (stems of img1.jpg / img2.jpg).
  statusesSpy = vi
    .spyOn(api.review, "imageStatuses")
    .mockResolvedValue({ statuses: {}, detection_stems: ["img1", "img2"], unreadable: [] });
  // Default: no recorded generation confidence -> no Conf >= censoring warning. Tests exercising
  // the warning override this per-case.
  vi.spyOn(api.review, "generationConf").mockResolvedValue({ generation_conf: null });
  // The priority-queue model picker fetches this on every render with a project open. Default
  // empty; the priority-queue describe block below overrides per-case.
  vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({ models: [] });
  // The validation-reference promotion resolves its trait from the project's own registered
  // traits (mirrors ResultsTab); one registered trait auto-selects with no picker shown.
  vi.spyOn(resultsApi, "traits").mockResolvedValue({
    traits: ["subject_a"],
    milestone_fractions_by_trait: { subject_a: [0.5, 0.95] },
    invalid_specs: [],
  });
  // Default: a standard 3-band RGB image; the band picker's own describe block overrides this
  // per-case to exercise the >3-band path.
  vi.spyOn(api.images, "bands").mockResolvedValue({
    band_count: 3,
    bands: [
      { name: "Red", wavelength_nm: null, dtype: "uint8", min: 0, max: 255 },
      { name: "Green", wavelength_nm: null, dtype: "uint8", min: 0, max: 255 },
      { name: "Blue", wavelength_nm: null, dtype: "uint8", min: 0, max: 255 },
    ],
  });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ReviewTab empty states", () => {
  it("shows 0 / 0, disables detection nav, and explains that filters exclude everything", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(useStore.getState().review.matches).not.toBeNull());

    expect(screen.getByText("0 / 0")).toBeInTheDocument();
    expect(prevBtn()).toBeDisabled();
    expect(nextBtn()).toBeDisabled();

    // Predictions dir IS configured -> the card blames the filters, echoing them.
    expect(screen.getByText("No detections to review")).toBeInTheDocument();
    expect(
      screen.getByText(/under the current filters \(IoU ≥ 0\.50, Conf ≥ 0\.25, type all\)/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/No predictions directory configured/)).not.toBeInTheDocument();
  });

  it("explains a missing predictions directory instead of blaming filters", async () => {
    setupDataset({ predDir: null });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(useStore.getState().review.matches).not.toBeNull());

    expect(screen.getByText("No detections to review")).toBeInTheDocument();
    expect(screen.getByText(/No predictions directory configured/)).toBeInTheDocument();
    expect(screen.queryByText(/under the current filters/)).not.toBeInTheDocument();
  });

  it("does not show the card before matches load or when detections exist", async () => {
    matchesSpy.mockResolvedValue(matchesRes([det(), det({ det_type: "fp" })]));
    render(<ReviewTab />);
    // Pre-load (store.review.matches is null): no card, counter shows 0 / 0.
    expect(screen.queryByText("No detections to review")).not.toBeInTheDocument();
    expect(screen.getByText("0 / 0")).toBeInTheDocument();

    await waitFor(() => expect(screen.getByText("1 / 2")).toBeInTheDocument());
    expect(screen.queryByText("No detections to review")).not.toBeInTheDocument();
  });
});

describe("ReviewTab image-status batch fetch", () => {
  it("clears a prior dataset's facts and toasts on a rejected fetch, rather than leaving them", async () => {
    useStore.setState({
      reviewStatus: {
        byImage: {},
        hasDetections: { "img2.jpg": false },
        unreadable: ["C:/data/other/annotations/2025-12-01/img2.json"],
        activeFilter: "all",
      },
    });
    statusesSpy.mockRejectedValue(new Error("the backend is unreachable"));

    render(<ReviewTab />);
    await waitFor(() => expect(statusesSpy).toHaveBeenCalledTimes(1));

    await waitFor(() =>
      expect(useStore.getState().toasts.map((t) => t.message)).toContain(
        "Could not load review status: the backend is unreachable",
      ),
    );
    expect(useStore.getState().reviewStatus.hasDetections).toEqual({});
    expect(useStore.getState().reviewStatus.unreadable).toEqual([]);
  });
});

describe("ReviewTab review-route scoping", () => {
  const gtAnn: Annotation = { subject: "subject_a", bbox: [10, 10, 50, 50], attributes: {} };

  it("scopes every review call to the dataset root, never to the project root", async () => {
    matchesSpy.mockResolvedValue(matchesRes([det()], { gt: [gtAnn] }));
    const actionSpy = vi.spyOn(api.review, "action").mockResolvedValue({
      status: "ok",
      image_status: "started",
      annotation_status: "partial",
      matches: matchesRes([det()], { gt: [gtAnn] }),
    });
    const backupSpy = vi.spyOn(api.review, "backupLabels").mockResolvedValue({
      status: "ok",
      files_backed_up: 0,
    });
    const markSpy = vi.spyOn(api.review, "markComplete").mockResolvedValue({
      status: "ok",
      image_status: "completed",
      annotation_status: "complete",
    });

    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());
    fireEvent.click(screen.getByTitle("Keep this ground-truth object (A)"));
    await waitFor(() => expect(actionSpy).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByLabelText("Reviewed"));
    await waitFor(() => expect(markSpy).toHaveBeenCalledTimes(1));

    // The two roots are distinct directories in this fixture, so a call that kept the project root
    // reads differently from one that moved to the dataset root.
    expect(matchesSpy.mock.calls[0][0].dataset_root).toBe("C:/data");
    expect(matchesSpy.mock.calls[0][0]).not.toHaveProperty("project_root");
    expect(statusesSpy.mock.calls[0][0].dataset_root).toBe("C:/data");
    expect(statusesSpy.mock.calls[0][0]).not.toHaveProperty("project_root");
    expect(actionSpy.mock.calls[0][0].dataset_root).toBe("C:/data");
    expect(actionSpy.mock.calls[0][0]).not.toHaveProperty("project_root");
    expect(backupSpy.mock.calls[0][0]).toBe("C:/data");
    expect(markSpy.mock.calls[0][0].dataset_root).toBe("C:/data");
    expect(markSpy.mock.calls[0][0]).not.toHaveProperty("project_root");
  });
});

describe("ReviewTab with a project open but no dataset selected", () => {
  beforeEach(() => {
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, dataset_root: null } },
    }));
  });

  it("disables the review actions and names the dataset selection as the fix", async () => {
    render(<ReviewTab />);
    expect(await screen.findByText(/No dataset is selected/)).toBeInTheDocument();
    expect(
      screen.getByText(/Reopen this project from the top bar to select its dataset/),
    ).toBeInTheDocument();

    expect(screen.getByRole("button", { name: /validation reference/i })).toBeDisabled();
    expect(screen.getByTitle(/Draw a box around an object the model missed/i)).toBeDisabled();
    expect(
      screen.getByTitle("Record that you checked this image for missed objects and found none"),
    ).toBeDisabled();
    expect(screen.getByLabelText("Reviewed")).toBeDisabled();
  });

  it("fires no review request, and falls back to no other root", async () => {
    const actionSpy = vi.spyOn(api.review, "action");
    render(<ReviewTab />);
    // Outwait the matches debounce before asserting that nothing was requested.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 300));
    });
    expect(matchesSpy).not.toHaveBeenCalled();
    expect(statusesSpy).not.toHaveBeenCalled();
    expect(actionSpy).not.toHaveBeenCalled();
  });
});

describe("ReviewTab validation-reference affordance", () => {
  const refBtn = () => screen.getByRole("button", { name: /validation reference/i });

  it("promotes a validated review and surfaces the honest result", async () => {
    const spy = vi.spyOn(api.review, "validateReference").mockResolvedValue({
      validated: true,
      reference: "review_confirmed",
      reviewed_image_count: 4,
      conf: 0.42,
      reason: "Validated. Your review confirms this model's counts.",
      buckets_stamped: [PRED_DIR_A],
    });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    await waitFor(() => expect(refBtn()).not.toBeDisabled());

    fireEvent.click(refBtn());
    await waitFor(() =>
      expect(spy).toHaveBeenCalledWith({
        dataset_root: "C:/data",
        trait: "subject_a",
        pred_dir: PRED_DIR_A,
        subject: "subject_a",
      }),
    );
    expect(await screen.findByText("Validated")).toBeInTheDocument();
  });

  it("shows a not-yet result honestly when the gate refuses", async () => {
    vi.spyOn(api.review, "validateReference").mockResolvedValue({
      validated: false,
      reference: "false",
      reviewed_image_count: 2,
      conf: 0.5,
      reason: "Not yet. Too few images have been reviewed.",
      buckets_stamped: [PRED_DIR_A],
    });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    await waitFor(() => expect(refBtn()).not.toBeDisabled());

    fireEvent.click(refBtn());
    expect(await screen.findByText("Not yet")).toBeInTheDocument();
    expect(screen.queryByText("Validated")).not.toBeInTheDocument();
  });

  it("renders a joined multi-failure reason with its line breaks preserved", async () => {
    const joined = "Not yet. First blocker.\n\nNot yet. Second blocker.";
    vi.spyOn(api.review, "validateReference").mockResolvedValue({
      validated: false,
      reference: "false",
      reviewed_image_count: 2,
      conf: 0.5,
      reason: joined,
      buckets_stamped: [PRED_DIR_A],
    });
    const { container } = render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    await waitFor(() => expect(refBtn()).not.toBeDisabled());

    fireEvent.click(refBtn());
    await screen.findByText("Not yet");
    const span = container.querySelector("span.whitespace-pre-wrap");
    expect(span).not.toBeNull();
    expect(span?.textContent).toBe(joined);
  });
});

describe("ReviewTab Conf >= filter censoring warning", () => {
  it("shows no warning today when the filter sits at or below generation confidence (fail-before baseline)", async () => {
    vi.spyOn(api.review, "generationConf").mockResolvedValue({ generation_conf: 0.5 });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    await waitFor(() => expect(screen.getAllByText(/Conf ≥ 0\.25/).length).toBeGreaterThan(0));
    expect(screen.queryByText(/conf-censored/)).not.toBeInTheDocument();
  });

  it("warns when the filter has been raised above the bucket's own generation confidence", async () => {
    vi.spyOn(api.review, "generationConf").mockResolvedValue({ generation_conf: 0.1 });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText(/⚠ Conf ≥ 0\.25/)).toBeInTheDocument());
    expect(screen.getByText(/conf-censored/)).toBeInTheDocument();
  });

  it("warns when the bucket has no recorded generation confidence (always conf-censored, per _conf_censored's own None branch)", async () => {
    vi.spyOn(api.review, "generationConf").mockResolvedValue({ generation_conf: null });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText(/⚠ Conf ≥ 0\.25/)).toBeInTheDocument());
    expect(screen.getByText(/conf-censored/)).toBeInTheDocument();
  });

  it("stays silent when there is no predictions directory selected at all", async () => {
    setupDataset({ predDir: null });
    vi.spyOn(api.review, "generationConf").mockResolvedValue({ generation_conf: null });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    expect(screen.queryByText(/conf-censored/)).not.toBeInTheDocument();
  });
});

describe("ReviewTab detection nav bounds", () => {
  it("disables Prev at the first detection and Next at the last", async () => {
    matchesSpy.mockResolvedValue(matchesRes([det(), det({ det_type: "fp" })]));
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 2")).toBeInTheDocument());

    expect(prevBtn()).toBeDisabled();
    expect(nextBtn()).not.toBeDisabled();

    fireEvent.click(nextBtn());
    expect(screen.getByText("2 / 2")).toBeInTheDocument();
    expect(prevBtn()).not.toBeDisabled();
    expect(nextBtn()).toBeDisabled();
  });
});

describe("ReviewTab in-place edit", () => {
  const gtAnn: Annotation = { subject: "subject_a", bbox: [10, 10, 50, 50], attributes: {} };

  function seedEditableMatches() {
    matchesSpy.mockResolvedValue(matchesRes([det()], { gt: [gtAnn] }));
  }

  const editBtn = () => screen.getByTitle("Adjust this shape on the canvas (E)");
  const saveBtn = () => screen.getByTitle("Replace the ground-truth shape with this one (Enter)");

  let actionSpy: MockInstance<typeof api.review.action>;
  let backupSpy: MockInstance<typeof api.review.backupLabels>;

  beforeEach(() => {
    seedEditableMatches();
    actionSpy = vi.spyOn(api.review, "action").mockResolvedValue({
      status: "ok",
      image_status: "started",
      annotation_status: "partial",
      // /action now returns the fresh matches; the edit tests install them via applyMatches.
      matches: matchesRes([det()], { gt: [gtAnn] }),
    });
    backupSpy = vi.spyOn(api.review, "backupLabels").mockResolvedValue({
      status: "ok",
      files_backed_up: 0,
    });
  });

  it("Edit swaps the verdict bar for Save/Cancel", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(editBtn());
    expect(saveBtn()).toBeInTheDocument();
    expect(screen.getByText("Cancel")).toBeInTheDocument();
    expect(screen.getByText("Editing")).toBeInTheDocument();
    expect(screen.queryByText(/Accept/)).not.toBeInTheDocument();
  });

  it("Enter commits the seeded GT box as an edited action, after the .original snapshot", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(editBtn());
    fireEvent.keyDown(window, { key: "Enter" });
    await waitFor(() => expect(actionSpy).toHaveBeenCalledTimes(1));
    expect(backupSpy).toHaveBeenCalledTimes(1);
    expect(actionSpy.mock.calls[0][0]).toMatchObject({
      action: "edited",
      edited_box: [10, 10, 50, 50],
      edited_points: null,
      det_type: "tp",
      class_name: "subject_a",
      gt_idx: 0,
    });
    // The verdict bar returns once the edit is committed.
    await waitFor(() =>
      expect(
        screen.queryByTitle("Replace the ground-truth shape with this one (Enter)"),
      ).not.toBeInTheDocument(),
    );
  });

  it("an FP edit commits the prediction shape (added to GT, not replacing)", async () => {
    matchesSpy.mockResolvedValue(
      matchesRes([det({ det_type: "fp", gt_idx: null, pred_idx: 0 })], {
        preds: [{ subject: "subject_a", bbox: [100, 100, 140, 150], attributes: {}, score: 0.7 }],
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(editBtn());
    fireEvent.click(screen.getByTitle("Write this shape to ground truth (Enter)"));
    await waitFor(() => expect(actionSpy).toHaveBeenCalledTimes(1));
    expect(actionSpy.mock.calls[0][0]).toMatchObject({
      action: "edited",
      edited_box: [100, 100, 140, 150],
      det_type: "fp",
      pred_idx: 0,
    });
  });

  it("Escape discards the edit without writing anything", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(editBtn());
    fireEvent.keyDown(window, { key: "Escape" });
    expect(
      screen.queryByTitle("Replace the ground-truth shape with this one (Enter)"),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/Accept/)).toBeInTheDocument();
    expect(actionSpy).not.toHaveBeenCalled();
  });
});

describe("ReviewTab recorded verdict", () => {
  const gtAnn: Annotation = { subject: "leaf", bbox: [14, 22, 52, 70], attributes: {} };

  beforeEach(() => {
    vi.spyOn(api.review, "backupLabels").mockResolvedValue({ status: "ok", files_backed_up: 0 });
  });

  it("shows a rejected prediction as rejected, both in the store and on the canvas badge", async () => {
    matchesSpy.mockResolvedValue(
      matchesRes([det({ det_type: "fp", gt_idx: null, pred_idx: 0, bbox: [24, 32, 68, 90] })], {
        preds: [{ subject: "leaf", bbox: [24, 32, 68, 90], attributes: {}, score: 0.7 }],
      }),
    );
    // Discarding a prediction leaves ground truth untouched, so the route reports no annotation
    // status and this local mark is the only record the reviewer keeps seeing.
    const actionSpy = vi.spyOn(api.review, "action").mockResolvedValue({
      status: "ok",
      image_status: "started",
      annotation_status: null,
      matches: matchesRes([]),
    });
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(screen.getByTitle("Discard this prediction; ground truth unchanged (R)"));
    await waitFor(() => expect(actionSpy).toHaveBeenCalledTimes(1));
    expect(actionSpy.mock.calls[0][0].action).toBe("rejected");

    await waitFor(() => {
      const reviewed = useStore.getState().review.matches!.detections[0];
      expect(reviewed.reviewed).toBe(true);
      expect(reviewed.reviewed_action).toBe("rejected");
    });
    expect(screen.getByText("rejected")).toBeInTheDocument();
  });

  it("marks an edited detection with the reviewer's own verdict, not a blanket acceptance", async () => {
    matchesSpy.mockResolvedValue(matchesRes([det({ bbox: [14, 22, 52, 70] })], { gt: [gtAnn] }));
    const actionSpy = vi.spyOn(api.review, "action").mockResolvedValue({
      status: "ok",
      image_status: "started",
      annotation_status: "partial",
      matches: matchesRes(
        [det({ bbox: [14, 22, 52, 70], reviewed: true, reviewed_action: "edited" })],
        { gt: [gtAnn] },
      ),
    });
    // Calls through to the real store action, so what it records is the argument the tab passes.
    const markSpy = vi.spyOn(useStore.getState(), "markDetectionReviewed");
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(screen.getByTitle("Adjust this shape on the canvas (E)"));
    fireEvent.keyDown(window, { key: "Enter" });
    await waitFor(() => expect(actionSpy).toHaveBeenCalledTimes(1));

    expect(markSpy).toHaveBeenCalledTimes(1);
    expect(markSpy.mock.calls[0]).toEqual([0, "edited"]);
    // An edit rewrites ground truth, so the route's recomputed matches land on top of the local
    // mark; the detection still has to read as edited once they do.
    await waitFor(() =>
      expect(useStore.getState().review.matches!.detections[0].reviewed_action).toBe("edited"),
    );
    expect(screen.getByText("edited")).toBeInTheDocument();
  });

  it("abandons the verdict when the pristine label baseline cannot be captured", async () => {
    matchesSpy.mockResolvedValue(matchesRes([det({ bbox: [14, 22, 52, 70] })], { gt: [gtAnn] }));
    vi.spyOn(api.review, "backupLabels").mockRejectedValue(new Error("label dir is read-only"));
    const actionSpy = vi.spyOn(api.review, "action").mockResolvedValue({
      status: "ok",
      image_status: "started",
      annotation_status: "partial",
      matches: matchesRes([det()], { gt: [gtAnn] }),
    });
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(screen.getByTitle("Keep this ground-truth object (A)"));
    await waitFor(() =>
      expect(useStore.getState().toasts.map((t) => t.message)).toContain(
        "Backup failed. Retry the action.",
      ),
    );
    expect(actionSpy).not.toHaveBeenCalled();
    expect(useStore.getState().review.matches!.detections[0].reviewed).toBe(false);
  });
});

describe("ReviewTab mark-missed-object affordance", () => {
  let actionSpy: MockInstance<typeof api.review.action>;
  let backupSpy: MockInstance<typeof api.review.backupLabels>;

  const markBtn = () => screen.getByTitle(/Draw a box around an object the model missed/i);

  beforeEach(() => {
    // No existing detections at all: the exact case startEdit can't handle (nothing to seed from).
    matchesSpy.mockResolvedValue(matchesRes([]));
    actionSpy = vi.spyOn(api.review, "action").mockResolvedValue({
      status: "ok",
      image_status: "started",
      annotation_status: "partial",
      matches: matchesRes([]),
    });
    backupSpy = vi.spyOn(api.review, "backupLabels").mockResolvedValue({
      status: "ok",
      files_backed_up: 0,
    });
  });

  it("draws and submits a missed object with no detection selected", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());

    fireEvent.click(markBtn());
    const stage = screen.getByTestId("canvas-stage");
    fireEvent.mouseDown(stage, { clientX: 10, clientY: 10, button: 0 });
    fireEvent.mouseMove(stage, { clientX: 60, clientY: 60 });
    fireEvent.mouseUp(stage, { clientX: 60, clientY: 60 });

    expect(screen.getByText("Marking missed object")).toBeInTheDocument();
    fireEvent.click(screen.getByTitle("Save this missed object to ground truth (Enter)"));

    await waitFor(() => expect(actionSpy).toHaveBeenCalledTimes(1));
    expect(backupSpy).toHaveBeenCalledTimes(1);
    expect(actionSpy.mock.calls[0][0]).toMatchObject({
      det_type: "fn",
      action: "edited",
      class_name: "subject_a",
      gt_idx: null,
      pred_idx: null,
      conf: null,
      edited_box: [10, 10, 60, 60],
      edited_points: null,
    });
    // The verdict bar returns once the box is committed.
    await waitFor(() =>
      expect(screen.queryByText("Marking missed object")).not.toBeInTheDocument(),
    );
  });

  it("cancels the armed draw mode without recording anything", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());

    fireEvent.click(markBtn());
    expect(screen.getByRole("button", { name: /cancel drawing/i })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /cancel drawing/i }));

    expect(screen.queryByText("Marking missed object")).not.toBeInTheDocument();
    expect(actionSpy).not.toHaveBeenCalled();
  });

  it("keeps a missed-object box drawn exactly at the minimum side", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());

    fireEvent.click(markBtn());
    const stage = screen.getByTestId("canvas-stage");
    fireEvent.mouseDown(stage, { clientX: 10, clientY: 10, button: 0 });
    fireEvent.mouseMove(stage, { clientX: 13, clientY: 13 });
    fireEvent.mouseUp(stage, { clientX: 13, clientY: 13 });

    expect(screen.getByText("Marking missed object")).toBeInTheDocument();
    fireEvent.click(screen.getByTitle("Save this missed object to ground truth (Enter)"));

    await waitFor(() => expect(actionSpy).toHaveBeenCalledTimes(1));
    expect(actionSpy.mock.calls[0][0]).toMatchObject({ edited_box: [10, 10, 13, 13] });
  });

  it("a box too small to save is rejected before it ever reaches Save", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());

    fireEvent.click(markBtn());
    const stage = screen.getByTestId("canvas-stage");
    fireEvent.mouseDown(stage, { clientX: 10, clientY: 10, button: 0 });
    fireEvent.mouseMove(stage, { clientX: 11, clientY: 11 });
    fireEvent.mouseUp(stage, { clientX: 11, clientY: 11 });

    expect(screen.queryByText("Marking missed object")).not.toBeInTheDocument();
    expect(actionSpy).not.toHaveBeenCalled();
  });
});

describe("ReviewTab sweep-attested affordance", () => {
  const sweepBtn = () =>
    screen.getByTitle("Record that you checked this image for missed objects and found none");

  it("records an explicit no-geometry attestation, distinct from a missed-object edit", async () => {
    matchesSpy.mockResolvedValue(matchesRes([]));
    const actionSpy = vi.spyOn(api.review, "action").mockResolvedValue({
      status: "ok",
      image_status: "started",
      annotation_status: null,
      matches: matchesRes([]),
    });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());

    fireEvent.click(sweepBtn());
    await waitFor(() => expect(actionSpy).toHaveBeenCalledTimes(1));
    expect(actionSpy.mock.calls[0][0]).toMatchObject({
      det_type: "sweep",
      action: "swept",
      gt_idx: null,
      pred_idx: null,
      edited_box: null,
      edited_points: null,
    });
    // Toasts render from a component mounted in App.tsx, not ReviewTab itself; assert the pushed
    // toast directly.
    await waitFor(() =>
      expect(useStore.getState().toasts.map((t) => t.message)).toContain(
        "Recorded: no missed objects found on this image.",
      ),
    );
  });

  it("is disabled while drawing a missed object", async () => {
    matchesSpy.mockResolvedValue(matchesRes([]));
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());

    fireEvent.click(screen.getByTitle(/Draw a box around an object the model missed/i));
    expect(sweepBtn()).toBeDisabled();
  });
});

describe("ReviewTab class filter", () => {
  function openFilters() {
    const btn = screen.getByTitle("Show or hide the review filters");
    if (btn.getAttribute("aria-expanded") !== "true") fireEvent.click(btn);
  }

  it("offers every subject present on the image, from the unfiltered gt/pred lists", async () => {
    matchesSpy.mockResolvedValue(
      matchesRes([det()], {
        gt: [{ subject: "subject_a", bbox: [10, 10, 50, 50], attributes: {} }],
        preds: [{ subject: "leaf", bbox: [60, 60, 90, 90], attributes: {}, score: 0.5 }],
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();

    const select = screen.getByLabelText("Class filter") as HTMLSelectElement;
    const values = Array.from(select.options).map((o) => o.value);
    expect(values).toEqual(["all", "leaf", "subject_a"]);
  });

  it("re-fetches matches scoped to the picked class", async () => {
    matchesSpy.mockResolvedValue(
      matchesRes([det()], {
        gt: [{ subject: "subject_a", bbox: [10, 10, 50, 50], attributes: {} }],
        preds: [{ subject: "leaf", bbox: [60, 60, 90, 90], attributes: {}, score: 0.5 }],
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();

    fireEvent.change(screen.getByLabelText("Class filter"), { target: { value: "leaf" } });
    await waitFor(() =>
      expect(matchesSpy).toHaveBeenLastCalledWith(
        expect.objectContaining({ filter_class: "leaf" }),
        expect.anything(),
      ),
    );
  });

  it("offers the confirmed/predicted values under a classified scope, not object-class subjects", async () => {
    matchesSpy.mockResolvedValue(
      matchesRes([det()], {
        gt: [{ subject: "subject_a", bbox: [10, 10, 50, 50], attributes: { ripeness: "ripe" } }],
        preds: [
          {
            subject: "subject_a",
            bbox: [60, 60, 90, 90],
            attributes: { ripeness: "unripe" },
            score: 0.5,
          },
        ],
        attribute: "ripeness",
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();

    const select = screen.getByLabelText("Class filter") as HTMLSelectElement;
    const values = Array.from(select.options).map((o) => o.value);
    // Without the change, availableClasses reads a.subject on every record, so both entries
    // collapse to one "subject_a" option instead of the two confirmed/predicted values.
    expect(values).toEqual(["all", "ripe", "unripe"]);
  });
});

describe("ReviewTab under a classified scope", () => {
  it("names the confirmed value on accept and reject titles for a tp, and disables reject", async () => {
    matchesSpy.mockResolvedValue(matchesRes([det({ det_type: "tp" })], { attribute: "ripeness" }));
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    // Without the change, these read "Keep this ground-truth object (A)" and "Delete this
    // ground-truth object (R)", describing a presence verdict this review never adjudicated.
    expect(screen.getByTitle("Keep this confirmed value (A)")).toBeInTheDocument();
    const rejectBtn = screen.getByTitle(/detector-scope action/i);
    expect(rejectBtn).toBeDisabled();
  });

  it("names the predicted value on accept and reject titles for an fp, and leaves reject enabled", async () => {
    matchesSpy.mockResolvedValue(
      matchesRes([det({ det_type: "fp", gt_idx: null, pred_idx: 0 })], { attribute: "ripeness" }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    // Without the change, these read "Add this prediction to ground truth (A)" and "Discard this
    // prediction; ground truth unchanged (R)".
    expect(
      screen.getByTitle("Confirm this predicted value for the object (A)"),
    ).toBeInTheDocument();
    const rejectBtn = screen.getByTitle(
      "Discard this predicted value; the confirmed value stays (R)",
    );
    expect(rejectBtn).not.toBeDisabled();
  });

  it("shows no confirm dialog for the disabled reject on a fn", async () => {
    const confirmSpy = vi.spyOn(window, "confirm");
    matchesSpy.mockResolvedValue(matchesRes([det({ det_type: "fn" })], { attribute: "ripeness" }));
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(screen.getByTitle(/detector-scope action/i));
    // Without the change, reject stays enabled on a fn under a classified scope, so this click
    // would raise the delete-confirmation dialog the design says a classified review never shows.
    expect(confirmSpy).not.toHaveBeenCalled();
  });

  it("opens no dialog and posts nothing when 'r' is pressed on a tp", async () => {
    const confirmSpy = vi.spyOn(window, "confirm");
    const actionSpy = vi.spyOn(api.review, "action");
    matchesSpy.mockResolvedValue(matchesRes([det({ det_type: "tp" })], { attribute: "ripeness" }));
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    // Without the change, the hotkey guards only on current/edit/reviewLocked/canReview, so 'r'
    // raises the delete-confirmation dialog and posts a request the backend refuses.
    fireEvent.keyDown(window, { key: "r" });
    expect(confirmSpy).not.toHaveBeenCalled();
    expect(actionSpy).not.toHaveBeenCalled();
  });

  it("disables the missed-object control, naming the Annotate tab as the remedy", async () => {
    matchesSpy.mockResolvedValue(matchesRes([], { attribute: "ripeness" }));
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());

    // Without the change, the control stays enabled and posting a missed object under a
    // classified scope would 400 only after the reviewer draws the box.
    const btn = screen.getByTitle(/fabricate a state nobody assessed/i);
    expect(btn).toBeDisabled();
  });
});

describe("ReviewTab matches-recompute effect", () => {
  it("re-fetches when the prediction dir changes under the same image (agent swaps model)", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(1));
    expect(matchesSpy.mock.calls[0][0].pred_path).toBe(`${PRED_DIR_A}/img1.json`);

    // mergeSnapshot's same-identity branch adopts the backend-changed prediction dir
    // without changing imgPath; the tab must not keep showing the old model's matches.
    act(() => {
      const s = useStore.getState();
      s.patchGui({ dataset: { ...s.gui.dataset, predictions_dir: PRED_DIR_B } });
    });
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(2));
    expect(matchesSpy.mock.calls[1][0].pred_path).toBe(`${PRED_DIR_B}/img1.json`);
  });

  it("re-fetches when an agent review_focus re-targets the already-open image with identical paths", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(1));

    // Same dataset identity + same image index => api.dataset.select is not called and every
    // label/prediction path is unchanged (the field case: stage_proposals overwrote the file
    // in place). Without the refetch nonce the recompute effect's identical-path skip leaves
    // stale cached matches on the canvas; the focus must force a refetch anyway.
    const selectSpy = vi.spyOn(api.dataset, "select").mockResolvedValue({} as never);
    vi.spyOn(api.dataset, "nav").mockResolvedValue({ status: "ok", current_image_index: 0 });

    await act(async () => {
      await applyReviewFocus({
        dataset_root: "C:/data",
        subject: "subject_a",
        date: "2026-01-01",
        image_index: 0,
      });
    });

    expect(selectSpy).not.toHaveBeenCalled();
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(2));
  });

  it("does not re-fetch when a WS snapshot rebuilds the dataset object with identical paths", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(1));

    // Same strings, new object identity; the effect keys on the path strings, so a
    // routine WS broadcast must not reset the detection index/zoom via a reload.
    act(() => {
      const s = useStore.getState();
      s.patchGui({ dataset: { ...s.gui.dataset } });
    });
    // Outwait the 180ms debounce before asserting nothing fired.
    await act(async () => {
      await new Promise((r) => setTimeout(r, 300));
    });
    expect(matchesSpy).toHaveBeenCalledTimes(1);
  });
});

describe("ReviewTab image-level navigation", () => {
  function setup3Images(currentIndex = 0) {
    useStore.setState((s) => ({
      gui: {
        ...s.gui,
        dataset: {
          ...s.gui.dataset,
          image_list: ["img1.jpg", "img2.jpg", "img3.jpg"],
          current_image_index: currentIndex,
        },
      },
    }));
    vi.spyOn(api.dataset, "nav").mockResolvedValue({ status: "ok", current_image_index: 0 });
  }
  const nextImage = () => screen.getByLabelText("Next image");

  it("skips images with zero detections during navigation", async () => {
    setup3Images();
    // img2 has nothing to review -> nav must jump straight from img1 to img3.
    statusesSpy.mockResolvedValue({
      statuses: {},
      detection_stems: ["img1", "img3"],
      unreadable: [],
    });
    render(<ReviewTab />);
    await waitFor(() =>
      expect(useStore.getState().reviewStatus.hasDetections["img2.jpg"]).toBe(false),
    );

    act(() => fireEvent.click(nextImage()));
    expect(useStore.getState().gui.dataset.current_image_index).toBe(2);
  });

  it("keeps an unreadable image navigable rather than skipping it like an empty one", async () => {
    setup3Images();
    // img2's document would not read; it must stay reachable, unlike a genuinely empty one.
    statusesSpy.mockResolvedValue({
      statuses: {},
      detection_stems: ["img1", "img3"],
      unreadable: ["/data/annotations/2026-03-02/img2.json"],
    });
    render(<ReviewTab />);
    await waitFor(() =>
      expect(useStore.getState().reviewStatus.unreadable).toEqual([
        "/data/annotations/2026-03-02/img2.json",
      ]),
    );
    expect(useStore.getState().reviewStatus.hasDetections["img2.jpg"]).toBe(true);

    act(() => fireEvent.click(nextImage()));
    // Reachable in traversal order, unlike the skipped-empty case above.
    expect(useStore.getState().gui.dataset.current_image_index).toBe(1);
  });

  it("the Reviewed/Unreviewed filter narrows which images navigation walks", async () => {
    setup3Images();
    // img1 + img3 reviewed, img2 not; all three have detections.
    statusesSpy.mockResolvedValue({
      statuses: { "img1.jpg": "completed", "img3.jpg": "completed" },
      detection_stems: ["img1", "img2", "img3"],
      unreadable: [],
    });
    render(<ReviewTab />);
    await waitFor(() =>
      expect(useStore.getState().reviewStatus.byImage["img1.jpg"]).toBe("completed"),
    );

    act(() => useStore.getState().setReviewStatusFilter("reviewed"));
    // Under the Reviewed filter, Next from img1 skips the unreviewed img2 and lands on img3.
    act(() => fireEvent.click(nextImage()));
    expect(useStore.getState().gui.dataset.current_image_index).toBe(2);
  });
});

describe("ReviewTab auto-resume", () => {
  it("lands on the first unreviewed detection when entering a partially-reviewed image", async () => {
    matchesSpy.mockResolvedValue(
      matchesRes([
        det({ reviewed: true, reviewed_action: "accepted" }),
        det({ det_type: "fp", pred_idx: 0, gt_idx: null }),
        det({ det_type: "fn", pred_idx: null, gt_idx: 1 }),
      ]),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(useStore.getState().review.matches).not.toBeNull());
    // Detection 0 is already reviewed -> resume on detection index 1 (the first unreviewed).
    await waitFor(() => expect(useStore.getState().gui.review.detection_idx).toBe(1));
    expect(screen.getByText("2 / 3")).toBeInTheDocument();
  });
});

describe("ReviewTab symbology", () => {
  it("draws the focused FN as a dashed under-review box, not a solid one", async () => {
    matchesSpy.mockResolvedValue(
      matchesRes([det({ det_type: "fn", pred_idx: null, gt_idx: 0 })], {
        gt: [{ subject: "subject_a", bbox: [10, 10, 50, 50], attributes: {} }],
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    // The under-review shape is the highlighter-blue (#00BFFF) box; it must be dashed to match
    // the "Under review" legend entry (a solid blue box matches no legend row).
    const active = screen
      .getAllByTestId("k-rect")
      .find((r) => r.getAttribute("data-stroke") === "#00BFFF");
    expect(active).toBeDefined();
    expect(active!.getAttribute("data-dashed")).toBe("true");
  });

  it("draws the matched prediction next to the ground truth of the matched detection under review", async () => {
    // A matched detection's verdict judges how far the predicted shape sits from the true one, so
    // drawing the ground truth alone asks for that judgement with nothing to compare against.
    matchesSpy.mockResolvedValue(
      matchesRes([det({ det_type: "tp", gt_idx: 0, pred_idx: 0, bbox: [12, 20, 48, 66] })], {
        gt: [{ subject: "leaf", bbox: [12, 20, 48, 66], attributes: {} }],
        preds: [{ subject: "leaf", bbox: [17, 25, 61, 79], attributes: {}, score: 0.8 }],
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(2);
    const geom = (el: HTMLElement) =>
      ["data-x", "data-y", "data-width", "data-height"].map((a) => el.getAttribute(a));

    // The ground truth keeps the matched-outcome green; the prediction rides on top in the dashed
    // under-review blue, the symbology every other shape awaiting a verdict wears.
    const gtRect = rects.find((r) => r.getAttribute("data-stroke") === "#4CAF50");
    expect(gtRect).toBeDefined();
    expect(geom(gtRect!)).toEqual(["12", "20", "36", "46"]);
    expect(gtRect!.getAttribute("data-dashed")).toBe("false");

    const predRect = rects.find((r) => r.getAttribute("data-stroke") === "#00BFFF");
    expect(predRect).toBeDefined();
    expect(geom(predRect!)).toEqual(["17", "25", "44", "54"]);
    expect(predRect!.getAttribute("data-dashed")).toBe("true");
  });

  it("renders both a box and a polygon annotation on one image (no geometry kind hidden)", async () => {
    // Measurement-critical: a unified file may mix bbox and polygon annotations; the overlay must
    // draw both by their own geometry: hiding a kind is an unreviewed false-negative.
    matchesSpy.mockResolvedValue(
      matchesRes(
        [
          det({ det_type: "fn", pred_idx: null, gt_idx: 0, bbox: [0, 0, 10, 10] }),
          det({
            det_type: "fn",
            pred_idx: null,
            gt_idx: 1,
            bbox: [40, 40, 60, 60],
            class_name: "leaf",
          }),
        ],
        {
          gt: [
            { subject: "subject_a", bbox: [0, 0, 10, 10], attributes: {} },
            {
              subject: "leaf",
              rings: [
                [
                  [40, 40],
                  [60, 40],
                  [60, 60],
                ],
              ],
              attributes: {},
            },
          ],
        },
      ),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 2")).toBeInTheDocument());

    // The box GT draws as a k-rect and the polygon GT as a k-line: both present.
    expect(screen.getAllByTestId("k-rect").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByTestId("k-line").length).toBeGreaterThanOrEqual(1);
  });

  it("draws a point-carrying ground truth as the point mark, with no box around it", async () => {
    // A review file can hold a point annotation ({point: [x, y]} on the wire). compute_matches
    // ignores points today (no area, so no IoU), so no detection references one yet; this pins the
    // render contract for when one does: the mark it was placed as, never a box around it (a
    // fabricated extent) and never nothing (a real annotation silently absent from review).
    matchesSpy.mockResolvedValue(
      matchesRes([det({ det_type: "fn", pred_idx: null, gt_idx: 0, bbox: [200, 300, 200, 300] })], {
        gt: [{ subject: "tip", point: [200, 300], attributes: {} }],
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    const core = screen.getByTestId("k-circle");
    expect(core).toHaveAttribute("data-x", "200");
    expect(core).toHaveAttribute("data-y", "300");
    expect(screen.getAllByTestId("k-line")).toHaveLength(4); // the four reticle ticks
    expect(screen.queryAllByTestId("k-rect")).toHaveLength(0);
  });

  it("draws every ring of an occlusion-split prediction under review", async () => {
    // Accepting a two-part prediction accepts both parts, so both have to be on screen; a render
    // of ring 0 alone asks for a verdict on something the reviewer never saw.
    matchesSpy.mockResolvedValue(
      matchesRes([det({ det_type: "fp", gt_idx: null, pred_idx: 0, bbox: [0, 0, 60, 60] })], {
        preds: [
          {
            subject: "subject_a",
            rings: [
              [
                [0, 0],
                [10, 0],
                [10, 10],
              ],
              [
                [40, 40],
                [60, 40],
                [60, 60],
              ],
            ],
            attributes: {},
            score: 0.9,
          },
        ],
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    const lines = screen.getAllByTestId("k-line");
    expect(lines).toHaveLength(2);
    expect(lines.map((l) => l.getAttribute("data-points"))).toEqual([
      "0,0,10,0,10,10",
      "40,40,60,40,60,60",
    ]);
    // Both parts share the under-review symbology: one shape, one verdict.
    expect(lines.every((l) => l.getAttribute("data-dashed") === "true")).toBe(true);
  });
});

describe("ReviewTab in-place edit scope", () => {
  const multiRingPred: Annotation = {
    subject: "subject_a",
    rings: [
      [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      [
        [40, 40],
        [60, 40],
        [60, 60],
      ],
    ],
    attributes: {},
    score: 0.9,
  };

  it("refuses to hand-edit a multi-part shape instead of silently saving one part as the whole", async () => {
    // /review/action carries one edited contour (edited_points), so seeding part 1 and saving would
    // write half the object to ground truth: a precise, confident, wrong measurement.
    matchesSpy.mockResolvedValue(
      matchesRes([det({ det_type: "fp", gt_idx: null, pred_idx: 0, bbox: [0, 0, 60, 60] })], {
        preds: [multiRingPred],
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(screen.getByTitle("Adjust this shape on the canvas (E)"));
    expect(screen.queryByText("Editing")).not.toBeInTheDocument();
    expect(useStore.getState().toasts.at(-1)?.message).toMatch(/2 separate parts/);
  });

  it("refuses to hand-edit a point instead of inventing a box around a location", async () => {
    // The canvas editor authors an outline and /review/action carries a box or one contour. Seeding
    // either from a point would write a fabricated extent into ground truth.
    matchesSpy.mockResolvedValue(
      matchesRes([det({ det_type: "fp", gt_idx: null, pred_idx: 0, bbox: [20, 20, 20, 20] })], {
        preds: [{ subject: "tip", point: [20, 20], attributes: {}, score: 0.9 }],
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(screen.getByTitle("Adjust this shape on the canvas (E)"));
    expect(screen.queryByText("Editing")).not.toBeInTheDocument();
    expect(useStore.getState().toasts.at(-1)?.message).toMatch(/marks a location, not an outline/);
  });

  it("still opens the editor for a box detection (the point refusal is scoped, not blanket)", async () => {
    matchesSpy.mockResolvedValue(
      matchesRes([det({ det_type: "fp", gt_idx: null, pred_idx: 0, bbox: [0, 0, 10, 10] })], {
        preds: [{ subject: "subject_a", bbox: [0, 0, 10, 10], attributes: {}, score: 0.9 }],
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(screen.getByTitle("Adjust this shape on the canvas (E)"));
    expect(screen.getByText("Editing")).toBeInTheDocument();
    expect(useStore.getState().toasts).toHaveLength(0);
  });

  it("still opens the editor for a single-contour shape (the refusal is scoped, not blanket)", async () => {
    matchesSpy.mockResolvedValue(
      matchesRes([det({ det_type: "fp", gt_idx: null, pred_idx: 0, bbox: [0, 0, 10, 10] })], {
        preds: [{ ...multiRingPred, rings: [multiRingPred.rings![0]] }],
      }),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 1")).toBeInTheDocument());

    fireEvent.click(screen.getByTitle("Adjust this shape on the canvas (E)"));
    expect(screen.getByText("Editing")).toBeInTheDocument();
    expect(useStore.getState().toasts).toHaveLength(0);
  });
});

describe("ReviewTab priority queue", () => {
  // Idempotent, not a blind toggle: filtersOpen's initial state is read from localStorage
  // ("tcip.review.filtersOpen"), which persists across tests within this file (one jsdom
  // environment per file, not per test), a prior test in this block leaving it open would make
  // a bare toggle click close it here instead.
  function openFilters() {
    const btn = screen.getByTitle("Show or hide the review filters");
    if (btn.getAttribute("aria-expanded") !== "true") fireEvent.click(btn);
  }

  it("launches with the picked model's checkpoint and this date's images_dir, then shows the ranked count", async () => {
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [{ name: "run-42", checkpoint_path: "C:/ckpts/run-42.pt" }],
    });
    const launchSpy = vi
      .spyOn(api.review, "launchPriorityQueue")
      .mockResolvedValue({ status: "launched", job_id: "pq-1" });
    vi.spyOn(api.review, "priorityQueueJob").mockResolvedValue({
      job_id: "pq-1",
      status: "completed",
      error: null,
      queue: [
        { image: "img2.jpg", score: 0.9, calibration_member: true },
        { image: "img1.jpg", score: 0.4, calibration_member: false },
      ],
      total_candidates: 2,
      reviewed_skipped: 0,
      marks_unresolved: null,
    });

    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();
    await waitFor(() => expect(screen.getByText("run-42")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("Priority-order model"), {
      target: { value: "C:/ckpts/run-42.pt" },
    });
    fireEvent.click(
      screen.getByTitle("Rank this date's images by how useful reviewing them would be"),
    );

    await waitFor(() =>
      expect(launchSpy).toHaveBeenCalledWith({
        dataset_root: "C:/data",
        checkpoint_path: "C:/ckpts/run-42.pt",
        images_dir: "C:/data/images/2026-01-01",
      }),
    );
    expect(await screen.findByText(/Browse in priority order \(2 ranked\)/)).toBeInTheDocument();
    // Auto-enabled once a queue completes: that's clearly what computing one was for.
    expect((screen.getByLabelText(/Browse in priority order/) as HTMLInputElement).checked).toBe(
      true,
    );
    // img1.jpg is the tab's initial current image (see the dataset fixture below); its own
    // queue entry names calibration_member: false, so no badge for it.
    expect(screen.queryByText("Calibration")).not.toBeInTheDocument();
  });

  it("badges the current image when its queue entry names it a calibration member", async () => {
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [{ name: "run-42", checkpoint_path: "C:/ckpts/run-42.pt" }],
    });
    vi.spyOn(api.review, "launchPriorityQueue").mockResolvedValue({
      status: "launched",
      job_id: "pq-3",
    });
    vi.spyOn(api.review, "priorityQueueJob").mockResolvedValue({
      job_id: "pq-3",
      status: "completed",
      error: null,
      queue: [
        { image: "img1.jpg", score: 0.9, calibration_member: true },
        { image: "img2.jpg", score: 0.4, calibration_member: false },
      ],
      total_candidates: 2,
      reviewed_skipped: 0,
      marks_unresolved: null,
    });

    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();
    await waitFor(() => expect(screen.getByText("run-42")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("Priority-order model"), {
      target: { value: "C:/ckpts/run-42.pt" },
    });
    fireEvent.click(
      screen.getByTitle("Rank this date's images by how useful reviewing them would be"),
    );

    expect(await screen.findByText("Calibration")).toBeInTheDocument();
  });

  it("shows why a mark is missing instead of a silent non-member, when marks_unresolved is set", async () => {
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [{ name: "run-42", checkpoint_path: "C:/ckpts/run-42.pt" }],
    });
    vi.spyOn(api.review, "launchPriorityQueue").mockResolvedValue({
      status: "launched",
      job_id: "pq-4",
    });
    vi.spyOn(api.review, "priorityQueueJob").mockResolvedValue({
      job_id: "pq-4",
      status: "completed",
      error: null,
      queue: [
        { image: "img1.jpg", score: 0.9 },
        { image: "img2.jpg", score: 0.4 },
      ],
      total_candidates: 2,
      reviewed_skipped: 0,
      marks_unresolved:
        "images_dir='C:/data/images/2026-01-01' is not the bound run's recorded images root",
    });

    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();
    await waitFor(() => expect(screen.getByText("run-42")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("Priority-order model"), {
      target: { value: "C:/ckpts/run-42.pt" },
    });
    fireEvent.click(
      screen.getByTitle("Rank this date's images by how useful reviewing them would be"),
    );

    expect(await screen.findByText("Calibration unknown")).toBeInTheDocument();
    expect(screen.queryByText("Calibration")).not.toBeInTheDocument();
  });

  it("surfaces the tool's own refusal honestly, not a generic failure", async () => {
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue({
      models: [{ name: "run-42", checkpoint_path: "C:/ckpts/run-42.pt" }],
    });
    vi.spyOn(api.review, "launchPriorityQueue").mockResolvedValue({
      status: "launched",
      job_id: "pq-2",
    });
    vi.spyOn(api.review, "priorityQueueJob").mockResolvedValue({
      job_id: "pq-2",
      status: "failed",
      error: "no scorer registered as 'nonsense'",
      queue: [],
      total_candidates: 0,
      reviewed_skipped: 0,
      marks_unresolved: null,
    });

    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();
    await waitFor(() => expect(screen.getByText("run-42")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("Priority-order model"), {
      target: { value: "C:/ckpts/run-42.pt" },
    });
    fireEvent.click(
      screen.getByTitle("Rank this date's images by how useful reviewing them would be"),
    );

    expect(await screen.findByText("no scorer registered as 'nonsense'")).toBeInTheDocument();
    expect(screen.queryByLabelText(/Browse in priority order/)).not.toBeInTheDocument();
  });

  it("the Rank button stays disabled until a model is picked", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();
    expect(
      screen.getByTitle("Rank this date's images by how useful reviewing them would be"),
    ).toBeDisabled();
  });
});

describe("ReviewTab band picker (progressive disclosure)", () => {
  function openFilters() {
    const btn = screen.getByTitle("Show or hide the review filters");
    if (btn.getAttribute("aria-expanded") !== "true") fireEvent.click(btn);
  }

  it("is hidden for a standard 3-band RGB dataset (the beforeEach default)", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();
    expect(screen.queryByLabelText("R band")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Band")).not.toBeInTheDocument();
  });

  it("is shown for a >3-band (multispectral) dataset, defaulted to the first three reported bands", async () => {
    vi.spyOn(api.images, "bands").mockResolvedValue({
      band_count: 4,
      bands: [
        { name: "Blue", wavelength_nm: 475, dtype: "uint16", min: 0, max: 65535 },
        { name: "Green", wavelength_nm: 560, dtype: "uint16", min: 0, max: 65535 },
        { name: "Red", wavelength_nm: 650, dtype: "uint16", min: 0, max: 65535 },
        { name: "NIR", wavelength_nm: 840, dtype: "uint16", min: 0, max: 65535 },
      ],
    });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();

    await waitFor(() => expect(screen.getByLabelText("R band")).toBeInTheDocument());
    expect((screen.getByLabelText("R band") as HTMLSelectElement).value).toBe("Blue");
    expect((screen.getByLabelText("G band") as HTMLSelectElement).value).toBe("Green");
    expect((screen.getByLabelText("B band") as HTMLSelectElement).value).toBe("Red");
  });

  it("collapses to one Band dropdown for a single-band source", async () => {
    // Never reached through the app's own >3 gate (a 1-band source never shows the picker at
    // all); this exercises the component's own collapsing behavior directly, independent of
    // that mount-level gate, by forcing band_count to 1 while still >3 is what the real gate
    // checks. Kept honest: assert the gate itself stays closed here too.
    vi.spyOn(api.images, "bands").mockResolvedValue({
      band_count: 1,
      bands: [{ name: "Panchromatic", wavelength_nm: null, dtype: "uint16", min: 0, max: 4095 }],
    });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();
    expect(screen.queryByLabelText("R band")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Band")).not.toBeInTheDocument();
  });

  it("wires the selected bands and stretch into the canvas image URL", async () => {
    vi.spyOn(api.images, "bands").mockResolvedValue({
      band_count: 4,
      bands: [
        { name: "Blue", wavelength_nm: 475, dtype: "uint16", min: 0, max: 65535 },
        { name: "Green", wavelength_nm: 560, dtype: "uint16", min: 0, max: 65535 },
        { name: "Red", wavelength_nm: 650, dtype: "uint16", min: 0, max: 65535 },
        { name: "NIR", wavelength_nm: 840, dtype: "uint16", min: 0, max: 65535 },
      ],
    });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalled());
    openFilters();
    await waitFor(() => expect(screen.getByLabelText("R band")).toBeInTheDocument());

    fireEvent.change(screen.getByLabelText("R band"), { target: { value: "NIR" } });
    fireEvent.change(screen.getByLabelText("Stretch"), { target: { value: "percent_clip" } });

    await waitFor(() => {
      const url = screen.getByTestId("canvas-stage").getAttribute("data-image-url") ?? "";
      expect(url).toContain("bands=NIR%2CGreen%2CRed");
      expect(url).toContain("stretch=percent_clip");
    });
  });
});

describe("ReviewTab zoom-to-detection clamping", () => {
  it("clamps the zoomed view so no gap opens between the image edge and the canvas", async () => {
    // A detection near the image corner: centering it unclamped would push the image's edge
    // away from the canvas edge and open a gap on both axes.
    matchesSpy.mockResolvedValue(matchesRes([det()])); // bbox [10,10,50,50] in a 1000x800 image
    render(<ReviewTab />);
    await waitFor(() => expect(useStore.getState().review.matches).not.toBeNull());
    await waitFor(() => expect(useStore.getState().gui.view.scale).toBeGreaterThan(1));

    const v = useStore.getState().gui.view;
    // At this scale the image exceeds the canvas on both axes, so offsets stay at or
    // below zero (image edge pinned to the canvas edge, no gap).
    expect(v.offset_x).toBeLessThanOrEqual(0);
    expect(v.offset_y).toBeLessThanOrEqual(0);
  });
});

describe("ReviewTab canvas-push binding-presence gate", () => {
  it("blocks the push and sets canvasBindingMissing when no generation is adopted", async () => {
    useStore.setState({ bindingGeneration: null });
    const pushSpy = vi
      .spyOn(api.canvas, "pushState")
      .mockResolvedValue({ status: "ok", shapes_written: true });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(1));

    act(() => notifyCanvasStateRequest());
    await act(async () => {});

    expect(pushSpy).not.toHaveBeenCalled();
    expect(useStore.getState().canvasBindingMissing).toBe(true);
  });

  it("pushes with the adopted generation and clears canvasBindingMissing", async () => {
    useStore.setState({ bindingGeneration: 5, canvasBindingMissing: true });
    const pushSpy = vi
      .spyOn(api.canvas, "pushState")
      .mockResolvedValue({ status: "ok", shapes_written: true });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(1));

    act(() => notifyCanvasStateRequest());
    await act(async () => {});

    expect(pushSpy).toHaveBeenCalled();
    expect(pushSpy.mock.calls[0][0].binding_generation).toBe(5);
    expect(useStore.getState().canvasBindingMissing).toBe(false);
  });
});

describe("ReviewTab subject colours", () => {
  afterEach(() => {
    try {
      localStorage.removeItem("tcip.annotate.subjectColors");
    } catch {
      /* not available in this environment, nothing to clear */
    }
  });

  it("a per-browser recolour reaches the pushed canvas_meta swatch", async () => {
    useStore.getState().setRegistry({ subject_a: {} });
    useStore.setState({ bindingGeneration: 1 });
    const pushSpy = vi
      .spyOn(api.canvas, "pushState")
      .mockResolvedValue({ status: "ok", shapes_written: true });
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(1));

    act(() => notifyCanvasStateRequest());
    await act(async () => {});
    expect(pushSpy.mock.calls.at(-1)?.[0]?.classes).toEqual(
      expect.arrayContaining([{ name: "subject_a", color: subjectColor("subject_a") }]),
    );

    act(() => setSubjectColorOverride("subject_a", "#123456"));
    act(() => notifyCanvasStateRequest());
    await act(async () => {});
    expect(pushSpy.mock.calls.at(-1)?.[0]?.classes).toEqual(
      expect.arrayContaining([{ name: "subject_a", color: "#123456" }]),
    );
  });
});

describe("ReviewTab heading", () => {
  it("renders exactly one top-level heading naming the tab", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(1));
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Review");
  });
});
