import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { StatusBar } from "@/components/StatusBar";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.getState().setActiveTab("results");
});

afterEach(cleanup);

describe("StatusBar agent activity footer", () => {
  it("names the declared harness, not a generic Agent label", async () => {
    useStore
      .getState()
      .pushAgentActivity("annotate", "labels_written", { stem: "IMG_1" }, "claude-code 2.1.238");
    render(<StatusBar />);

    expect(await screen.findByText(/claude-code 2\.1\.238: labels_written/)).toBeInTheDocument();
  });

  it("reads Process for a write the sender declared no harness on", async () => {
    useStore.getState().pushAgentActivity("annotate", "labels_written", { stem: "IMG_1" }, null);
    render(<StatusBar />);

    expect(await screen.findByText(/Process: labels_written/)).toBeInTheDocument();
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
    expect(screen.getByText("1 box")).toBeInTheDocument();
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
    expect(screen.queryByText("1 box")).not.toBeInTheDocument();
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

    expect(screen.getByText("1 polygon")).toBeInTheDocument();
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
    expect(screen.getByText(/1 polygon/)).toBeInTheDocument();
    expect(screen.getByText(/1 box/)).toBeInTheDocument();
    expect(screen.getByText("|")).toBeInTheDocument();
  });

  it("pluralizes each count on its own value, not the other's", () => {
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
        boxes: [
          { x1: 0, y1: 0, x2: 10, y2: 10, subject: "fruit", attributes: {} },
          { x1: 20, y1: 0, x2: 30, y2: 10, subject: "fruit", attributes: {} },
        ],
      },
    }));
    render(<StatusBar />);

    // Both counts sit in one span as sibling text/element nodes beside the separator, so a
    // whole-string match misses; a substring regex still finds each independently.
    expect(screen.getByText(/1 polygon/)).toBeInTheDocument();
    expect(screen.getByText(/2 boxes/)).toBeInTheDocument();
  });

  it("labels a single point in the singular", () => {
    useStore.setState((s) => ({
      canvas: {
        ...s.canvas,
        points: [{ x: 5, y: 6, subject: "tip", attributes: {} }],
      },
    }));
    render(<StatusBar />);

    expect(screen.getByText("1 point")).toBeInTheDocument();
  });

  it("labels more than one point in the plural, on the point count's own value", () => {
    useStore.setState((s) => ({
      canvas: {
        ...s.canvas,
        points: [
          { x: 5, y: 6, subject: "tip", attributes: {} },
          { x: 8, y: 9, subject: "tip", attributes: {} },
        ],
      },
    }));
    render(<StatusBar />);

    expect(screen.getByText("2 points")).toBeInTheDocument();
  });
});
