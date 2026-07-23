import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { api } from "@/api/client";
import { applyReviewFocus } from "@/lib/reviewFocus";
import { useStore } from "@/store";
import type { Annotation, Detection, MatchesResponse } from "@/store/types";
import { ReviewTab } from "@/tabs/ReviewTab";

// Konva needs a real 2D canvas; these tests exercise empty-state rendering, nav
// bounds, and the matches-recompute effect, not drawing. Render Konva shapes as
// divs and CanvasStage as a passthrough.
vi.mock("konva", () => ({ default: {} }));
vi.mock("react-konva", () => ({
  // Expose stroke + dashed so symbology (dashed = under review) is assertable.
  Rect: (props: { stroke?: string; dash?: number[] }) => (
    <div
      data-testid="k-rect"
      data-stroke={props.stroke}
      data-dashed={props.dash ? "true" : "false"}
    />
  ),
  Line: (props: { stroke?: string; dash?: number[] }) => (
    <div
      data-testid="k-line"
      data-stroke={props.stroke}
      data-dashed={props.dash ? "true" : "false"}
    />
  ),
  Circle: () => <div data-testid="k-circle" />,
  Text: (props: { text?: string }) => <div data-testid="k-text" data-text={props.text} />,
}));
vi.mock("@/components/Canvas/CanvasStage", () => ({
  CanvasStage: (props: { children?: React.ReactNode }) => (
    <div data-testid="canvas-stage">{props.children}</div>
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
        subject: "catkin",
        date: "2026-01-01",
        image_list: ["img1.jpg", "img2.jpg"],
        current_image_index: 0,
        annotations_dir: "C:/data/annotations/2026-01-01",
        predictions_dir: predDir,
      },
    },
  }));
}

function det(over: Partial<Detection> = {}): Detection {
  return {
    det_type: "tp",
    class_name: "catkin",
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
  extra: { gt?: Annotation[]; preds?: Annotation[] } = {},
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
    .mockResolvedValue({ statuses: {}, detection_stems: ["img1", "img2"] });
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

    fireEvent.click(refBtn());
    await waitFor(() =>
      expect(spy).toHaveBeenCalledWith({
        project_root: "C:/proj",
        trait: "catkin",
        pred_dir: PRED_DIR_A,
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

    fireEvent.click(refBtn());
    expect(await screen.findByText("Not yet")).toBeInTheDocument();
    expect(screen.queryByText("Validated")).not.toBeInTheDocument();
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
  const gtAnn: Annotation = { subject: "catkin", bbox: [10, 10, 50, 50], attributes: {} };

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
      class_name: "catkin",
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
        preds: [{ subject: "catkin", bbox: [100, 100, 140, 150], attributes: {}, score: 0.7 }],
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

describe("ReviewTab matches-recompute effect", () => {
  it("re-fetches when the prediction dir changes under the same image (agent swaps model)", async () => {
    render(<ReviewTab />);
    await waitFor(() => expect(matchesSpy).toHaveBeenCalledTimes(1));
    expect(matchesSpy.mock.calls[0][0].pred_path).toBe(`${PRED_DIR_A}/img1.json`);

    // mergeSnapshot's same-identity branch adopts the backend-changed prediction dir
    // without changing imgPath — the tab must not keep showing the old model's matches.
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
    // stale cached matches on the canvas — the focus must force a refetch anyway.
    const selectSpy = vi.spyOn(api.dataset, "select").mockResolvedValue({} as never);
    vi.spyOn(api.dataset, "nav").mockResolvedValue({ status: "ok", current_image_index: 0 });

    await act(async () => {
      await applyReviewFocus({
        dataset_root: "C:/data",
        subject: "catkin",
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

    // Same strings, new object identity — the effect keys on the path strings, so a
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
    statusesSpy.mockResolvedValue({ statuses: {}, detection_stems: ["img1", "img3"] });
    render(<ReviewTab />);
    await waitFor(() =>
      expect(useStore.getState().reviewStatus.hasDetections["img2.jpg"]).toBe(false),
    );

    act(() => fireEvent.click(nextImage()));
    expect(useStore.getState().gui.dataset.current_image_index).toBe(2);
  });

  it("the Reviewed/Unreviewed filter narrows which images navigation walks", async () => {
    setup3Images();
    // img1 + img3 reviewed, img2 not; all three have detections.
    statusesSpy.mockResolvedValue({
      statuses: { "img1.jpg": "completed", "img3.jpg": "completed" },
      detection_stems: ["img1", "img2", "img3"],
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
        gt: [{ subject: "catkin", bbox: [10, 10, 50, 50], attributes: {} }],
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

  it("renders both a box and a polygon annotation on one image (no geometry kind hidden)", async () => {
    // Measurement-critical: a unified file may mix bbox and polygon annotations; the overlay must
    // draw BOTH by their own geometry — hiding a kind is an unreviewed false-negative.
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
            { subject: "catkin", bbox: [0, 0, 10, 10], attributes: {} },
            {
              subject: "leaf",
              points: [
                [40, 40],
                [60, 40],
                [60, 60],
              ],
              attributes: {},
            },
          ],
        },
      ),
    );
    render(<ReviewTab />);
    await waitFor(() => expect(screen.getByText("1 / 2")).toBeInTheDocument());

    // The box GT draws as a k-rect and the polygon GT as a k-line — both present.
    expect(screen.getAllByTestId("k-rect").length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByTestId("k-line").length).toBeGreaterThanOrEqual(1);
  });
});
