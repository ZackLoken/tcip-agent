import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { StatusBar } from "@/components/StatusBar";
import { useStore } from "@/store";
import { LANDED, loadOf } from "@/test/annotate";
import { PANEL_EVENTS } from "@/test/panelEvents";
import { GROUPED_SELECTION, RAW_SELECTION } from "@/test/store";
import { AGENT_CLIENT } from "@/test/training";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.getState().setActiveTab("results");
});

afterEach(cleanup);

describe("StatusBar agent activity footer", () => {
  const { panel, event } = PANEL_EVENTS.labels_written;

  it("names the declared harness and the written image, not a generic Agent label", async () => {
    useStore.getState().pushAgentActivity(panel, event, AGENT_CLIENT);
    render(<StatusBar />);

    expect(
      await screen.findByText(`${AGENT_CLIENT}: labels written on ${event.name}`),
    ).toBeInTheDocument();
  });

  it("reads Process for a write the sender declared no harness on", async () => {
    useStore.getState().pushAgentActivity(panel, event, null);
    render(<StatusBar />);

    expect(await screen.findByText(`Process: labels written on ${event.name}`)).toBeInTheDocument();
  });
});

const SELECTED_CAPTURE = RAW_SELECTION;
const IN_VIEW = RAW_SELECTION.images[0].path;

/** The produced document of the first photograph holding one box, as the save answered it. */
const ONE_BOX = LANDED.box.document;

/** Load the produced one-box document as the document of `image_path`. */
function loadIn(image_path: string) {
  useStore.getState().loadLabelsIntoCanvas({ ...ONE_BOX, image_path });
}

describe("StatusBar canvas facts", () => {
  beforeEach(() => {
    useStore.getState().setActiveTab("annotate");
    useStore.setState((s) => ({ gui: { ...s.gui, dataset: SELECTED_CAPTURE } }));
  });

  it("shows the image size and shape counts for a canvas loaded from the open dataset", () => {
    loadIn(IN_VIEW);
    render(<StatusBar />);

    expect(
      screen.getByText(`Image: ${ONE_BOX.img_width}×${ONE_BOX.img_height}`),
    ).toBeInTheDocument();
    expect(screen.getByText("1 box")).toBeInTheDocument();
  });

  it("shows no canvas facts off the Annotate tab", () => {
    useStore.getState().setActiveTab("inference");
    loadIn(IN_VIEW);
    render(<StatusBar />);

    expect(screen.queryByText(/Zoom: /)).not.toBeInTheDocument();
    expect(screen.queryByText(/Image: /)).not.toBeInTheDocument();
    expect(screen.queryByText("1 box")).not.toBeInTheDocument();
  });

  it("hides the image size and shape counts for a canvas loaded from another dataset", () => {
    loadIn(GROUPED_SELECTION.images[0].path);
    render(<StatusBar />);

    expect(screen.queryByText(/Image: /)).not.toBeInTheDocument();
    expect(screen.queryByText("1 box")).not.toBeInTheDocument();
  });
});

describe("StatusBar shape counts (stored records only, never a derived box)", () => {
  beforeEach(() => {
    useStore.getState().setActiveTab("annotate");
    useStore.setState((s) => ({ gui: { ...s.gui, dataset: SELECTED_CAPTURE } }));
    useStore.getState().loadLabelsIntoCanvas(loadOf(IN_VIEW));
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
            iscrowd: false,
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
            iscrowd: false,
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
            iscrowd: false,
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
            iscrowd: false,
          },
        ],
        boxes: [{ x1: 0, y1: 0, x2: 10, y2: 10, subject: "fruit", attributes: {}, iscrowd: false }],
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
            iscrowd: false,
          },
        ],
        boxes: [
          { x1: 0, y1: 0, x2: 10, y2: 10, subject: "fruit", attributes: {}, iscrowd: false },
          { x1: 20, y1: 0, x2: 30, y2: 10, subject: "fruit", attributes: {}, iscrowd: false },
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
        points: [{ x: 5, y: 6, subject: "tip", attributes: {}, iscrowd: false }],
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
          { x: 5, y: 6, subject: "tip", attributes: {}, iscrowd: false },
          { x: 8, y: 9, subject: "tip", attributes: {}, iscrowd: false },
        ],
      },
    }));
    render(<StatusBar />);

    expect(screen.getByText("2 points")).toBeInTheDocument();
  });
});
