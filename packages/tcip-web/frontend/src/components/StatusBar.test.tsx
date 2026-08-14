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

  it("counts the detections already reviewed, and fills the wheel to the same share", () => {
    useStore.getState().setMatches(
      matchesRes({
        n_tp: 2,
        n_fp: 2,
        detections: [
          det({ reviewed: true, reviewed_action: "rejected" }),
          det(),
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

  it("keeps the review readout off the annotate tab", () => {
    useStore.getState().setMatches(matchesRes({ n_tp: 5, detections: [det()] }));
    useStore.getState().setActiveTab("annotate");
    render(<StatusBar />);

    expect(screen.queryByText("TP 5")).not.toBeInTheDocument();
    expect(screen.queryByText(/reviewed$/)).not.toBeInTheDocument();
  });
});
