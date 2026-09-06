import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { StatusBar } from "@/components/StatusBar";
import { useStore } from "@/store";
import type { Detection, MatchesResponse } from "@/store/types";

const initialStoreState = useStore.getState();

function det(over: Partial<Detection> = {}): Detection {
  return {
    det_type: "tp",
    class_name: "leaf",
    conf: 0.9,
    iou: 0.8,
    gt_idx: 0,
    pred_idx: null,
    bbox: [12, 20, 48, 66],
    reviewed: false,
    reviewed_action: null,
    ...over,
  };
}

function matchesRes(over: Partial<MatchesResponse> = {}): MatchesResponse {
  return {
    img_width: 1000,
    img_height: 800,
    n_tp: 0,
    n_fp: 0,
    n_fn: 0,
    detections: [],
    gt: [],
    preds: [],
    image_status: "started",
    n_reviewed: 0,
    n_total: 0,
    subject: null,
    attribute: null,
    ...over,
  };
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.getState().setActiveTab("review");
});

afterEach(cleanup);

describe("StatusBar review readout", () => {
  it("reports the outcome counts the route computed, not a recount of the detections on screen", () => {
    // A type or class filter narrows the detection list without narrowing the image's own outcome
    // counts, so the two populations differ and the readout has to name the route's.
    useStore.getState().setMatches(
      matchesRes({
        n_tp: 5,
        n_fp: 3,
        n_fn: 4,
        detections: [det(), det(), det({ det_type: "fp", gt_idx: null, pred_idx: 0 })],
      }),
    );
    render(<StatusBar />);

    expect(screen.getByText("TP 5")).toBeInTheDocument();
    expect(screen.getByText("FP 3")).toBeInTheDocument();
    expect(screen.getByText("FN 4")).toBeInTheDocument();
  });

  it("reports the route's own reviewed/total, not a recount of the filtered detections on screen", () => {
    // n_reviewed/n_total come from review_progress over the whole image; a type filter (here
    // narrowing detections to the two fp's) must never be recounted as the wheel's own numbers.
    useStore.getState().setMatches(
      matchesRes({
        n_tp: 2,
        n_fp: 2,
        n_reviewed: 1,
        n_total: 4,
        detections: [
          det({ det_type: "fp", gt_idx: null, pred_idx: 0 }),
          det({ det_type: "fp", gt_idx: null, pred_idx: 1 }),
        ],
      }),
    );
    const { container } = render(<StatusBar />);

    expect(screen.getByText("1 / 4 reviewed")).toBeInTheDocument();
    // One of four reviewed is a quarter of the r=7 wheel's 44-unit circumference.
    const arc = container.querySelectorAll("circle")[1];
    expect(arc).toBeDefined();
    expect(arc.getAttribute("stroke-dasharray")).toBe("11 44");
  });

  it("stays visible when a filter matches nothing, since n_total still counts the whole image", () => {
    useStore
      .getState()
      .setMatches(matchesRes({ n_tp: 2, n_fp: 2, n_reviewed: 0, n_total: 4, detections: [] }));
    render(<StatusBar />);

    expect(screen.getByText("0 / 4 reviewed")).toBeInTheDocument();
  });

  it("hides the wheel only when the image itself carries no detections at all", () => {
    useStore.getState().setMatches(matchesRes({ n_reviewed: 0, n_total: 0, detections: [] }));
    render(<StatusBar />);

    expect(screen.queryByText(/reviewed$/)).not.toBeInTheDocument();
  });

  it("keeps the review readout off the annotate tab", () => {
    useStore.getState().setMatches(matchesRes({ n_tp: 5, n_total: 1, detections: [det()] }));
    useStore.getState().setActiveTab("annotate");
    render(<StatusBar />);

    expect(screen.queryByText("TP 5")).not.toBeInTheDocument();
    expect(screen.queryByText(/reviewed$/)).not.toBeInTheDocument();
  });
});

describe("StatusBar canvas facts", () => {
  beforeEach(() => {
    useStore.getState().setActiveTab("annotate");
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, images_dir: "/data/images/2026-01-01" } },
    }));
  });

  it("shows the image size and shape counts for a canvas loaded from the open dataset", () => {
    useStore.setState((s) => ({
      canvas: {
        ...s.canvas,
        loadedImagePath: "/data/images/2026-01-01/img1.jpg",
        imgWidth: 800,
        imgHeight: 600,
        boxes: [{ x1: 0, y1: 0, x2: 10, y2: 10, subject: "fruit", attributes: {} }],
      },
    }));
    render(<StatusBar />);

    expect(screen.getByText("Image: 800×600")).toBeInTheDocument();
    expect(screen.getByText(/boxes/)).toBeInTheDocument();
  });

  it("hides the image size and shape counts for a canvas loaded from another dataset", () => {
    useStore.setState((s) => ({
      canvas: {
        ...s.canvas,
        loadedImagePath: "/data/images/2025-11-02/img1.jpg",
        imgWidth: 800,
        imgHeight: 600,
        boxes: [{ x1: 0, y1: 0, x2: 10, y2: 10, subject: "fruit", attributes: {} }],
      },
    }));
    render(<StatusBar />);

    expect(screen.queryByText(/Image: /)).not.toBeInTheDocument();
    expect(screen.queryByText(/boxes/)).not.toBeInTheDocument();
  });
});

describe("StatusBar shape counts (stored records only, never a derived box)", () => {
  beforeEach(() => {
    useStore.getState().setActiveTab("annotate");
    useStore.setState((s) => ({
      gui: { ...s.gui, dataset: { ...s.gui.dataset, images_dir: "/data/images/2026-01-01" } },
      canvas: { ...s.canvas, loadedImagePath: "/data/images/2026-01-01/img1.jpg" },
    }));
  });

  it("counts a cut polygon's two pieces as polygons, never as boxes", () => {
    useStore.setState((s) => ({
      canvas: {
        ...s.canvas,
        polygons: [
          {
            rings: [
              [
                [0, 0],
                [10, 0],
                [10, 10],
                [0, 10],
              ],
            ],
            subject: "bud",
            attributes: {},
          },
          {
            rings: [
              [
                [20, 0],
                [30, 0],
                [30, 10],
                [20, 10],
              ],
            ],
            subject: "bud",
            attributes: {},
          },
        ],
      },
    }));
    render(<StatusBar />);

    expect(screen.getByText("2 polygons")).toBeInTheDocument();
    expect(screen.queryByText(/boxes/)).not.toBeInTheDocument();
  });

  it("shows the separator only between two shown counts, not with polygons and no boxes", () => {
    useStore.setState((s) => ({
      canvas: {
        ...s.canvas,
        polygons: [
          {
            rings: [
              [
                [0, 0],
                [10, 0],
                [10, 10],
                [0, 10],
              ],
            ],
            subject: "bud",
            attributes: {},
          },
        ],
      },
    }));
    const { container } = render(<StatusBar />);

    expect(screen.getByText("1 polygons")).toBeInTheDocument();
    expect(container.textContent).not.toContain("|");
  });

  it("shows both counts with the separator between them when both exist", () => {
    useStore.setState((s) => ({
      canvas: {
        ...s.canvas,
        polygons: [
          {
            rings: [
              [
                [0, 0],
                [10, 0],
                [10, 10],
                [0, 10],
              ],
            ],
            subject: "bud",
            attributes: {},
          },
        ],
        boxes: [{ x1: 0, y1: 0, x2: 10, y2: 10, subject: "fruit", attributes: {} }],
      },
    }));
    render(<StatusBar />);

    // Both counts and the separator sit in one span as sibling text/element nodes, so a
    // whole-string match misses; a substring regex still finds each independently.
    expect(screen.getByText(/1 polygons/)).toBeInTheDocument();
    expect(screen.getByText(/1 boxes/)).toBeInTheDocument();
    expect(screen.getByText("|")).toBeInTheDocument();
  });
});
