import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { StructuredRefusalError } from "@/api/http";
import type { CompareResult } from "@/api/training";
import { trainingApi } from "@/api/training";
import { RunComparison, type MarkedRun } from "@/components/RunComparison";

// The overlay chart owns its own per-run WebSocket streams; only the comparison's own rendering
// is under test here, so the transport is replaced while the rest of the module stays real.
vi.mock("@/api/training", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/training")>();
  return { ...actual, openTrainingStream: vi.fn(() => () => {}) };
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const MARKED: MarkedRun[] = [
  { runId: "run-a", experimentId: "exp-a" },
  { runId: "run-b", experimentId: "exp-b" },
];

function baseResult(overrides: Partial<CompareResult>): CompareResult {
  return {
    experiments: [{ experiment_id: "exp-a" }, { experiment_id: "exp-b" }],
    count: 2,
    same_dataset_fingerprint: null,
    ...overrides,
  };
}

describe("RunComparison data lines", () => {
  it("reads the images line as same source images when the tool says so", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({ same_dataset_fingerprint: true }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    expect(await screen.findByText("same source images")).toBeInTheDocument();
  });

  it("reads the images line as not the same when the tool says so", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({ same_dataset_fingerprint: false }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    expect(await screen.findByText("not the same source images")).toBeInTheDocument();
  });

  it("names which columns carry no record when the images line reads not comparable", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          { experiment_id: "exp-a", dataset_fingerprint: "v1:aaaa" },
          { experiment_id: "exp-b" },
        ],
        same_dataset_fingerprint: null,
      }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    expect(await screen.findByText(/not comparable/)).toBeInTheDocument();
    expect(screen.getByText(/exp-b carry no fingerprint/)).toBeInTheDocument();
  });

  it("reads the partition line from each column's own split state", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          { experiment_id: "exp-a", split: { case: "bound", manifest_dir: "splits/d1", seed: 7 } },
          { experiment_id: "exp-b", split: { case: "drawn", seed: 42 } },
        ],
      }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    expect(await screen.findByText(/bound to splits\/d1/)).toBeInTheDocument();
    expect(screen.getByText(/drawn again \(seed 42\)/)).toBeInTheDocument();
  });

  it("renders the library's own registry_error text with no added prefix", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          {
            experiment_id: "exp-a",
            registry_error: "registry unreadable: simulated decode failure",
          },
          { experiment_id: "exp-b" },
        ],
      }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    expect(
      await screen.findByText("registry unreadable: simulated decode failure"),
    ).toBeInTheDocument();
    expect(screen.queryByText(/registry unreadable: registry unreadable/)).not.toBeInTheDocument();
  });
});

describe("RunComparison on a change of the marked set", () => {
  it("reads as reading, never a stale absence claim, for a newly marked column", async () => {
    const compare = vi.spyOn(trainingApi, "compare").mockResolvedValue(baseResult({}));
    const { rerender } = render(<RunComparison marked={[MARKED[0]]} projectRoot={null} />);
    await screen.findByText("exp-a");

    compare.mockImplementation(() => new Promise(() => {})); // never resolves for the new set
    rerender(<RunComparison marked={MARKED} projectRoot={null} />);

    expect(await screen.findByText("Loading comparison...")).toBeInTheDocument();
  });
});

describe("RunComparison rank control", () => {
  function registeredResult(): CompareResult {
    return baseResult({
      experiments: [
        {
          experiment_id: "exp-a",
          registry: [
            {
              name: "exp-a",
              metrics: { val_map99: 0.7 },
              metrics_source: "trainer",
              registered_at: "2026-01-01T00:00:00Z",
            },
          ],
        },
        { experiment_id: "exp-b", registry: [] },
      ],
    });
  }

  it("shows the direction toggle only after the tool refuses an undeclared metric", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(registeredResult());
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      new StructuredRefusalError(
        {
          error: "'val_map99' has no declared ranking direction. Pass higher_is_better explicitly",
          needs_direction: true,
        },
        422,
        "",
      ),
    );

    render(<RunComparison marked={MARKED} projectRoot={null} />);
    fireEvent.change(await screen.findByRole("combobox"), {
      target: { value: "val_map99" },
    });
    expect(screen.queryByText("higher is better")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Rank" }));
    expect(await screen.findByText("higher is better")).toBeInTheDocument();
    expect(screen.getByText("lower is better")).toBeInTheDocument();
    expect(screen.getByText(/has no declared ranking direction/)).toBeInTheDocument();
  });

  it("shows the unverified switch only after the tool says every candidate was unverified", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(registeredResult());
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      new StructuredRefusalError(
        {
          error:
            "every registered model carrying 'val_map99' is unverified (metrics_source is not 'trainer')",
          all_unverified: true,
        },
        422,
        "",
      ),
    );

    render(<RunComparison marked={MARKED} projectRoot={null} />);
    const select = await screen.findByRole("combobox");
    fireEvent.change(select, { target: { value: "val_map99" } });
    expect(
      screen.queryByText("include checkpoints the trainer did not stamp"),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Rank" }));
    expect(
      await screen.findByText("include checkpoints the trainer did not stamp"),
    ).toBeInTheDocument();
  });

  it("renders the projected best-model answer once ranking succeeds", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(registeredResult());
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue({
      name: "exp-a",
      experiment_id: "exp-a",
      metrics: { val_map99: 0.7 },
      metrics_source: "trainer",
      higher_is_better: true,
      direction_source: "stated",
      excluded_unverified: [],
    });

    render(<RunComparison marked={MARKED} projectRoot={null} />);
    const select = await screen.findByRole("combobox");
    fireEvent.change(select, { target: { value: "val_map99" } });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    await waitFor(() =>
      expect(screen.getByText("exp-a", { selector: "span.font-mono" })).toBeInTheDocument(),
    );
    expect(screen.getByText(/trainer \/ stated direction/)).toBeInTheDocument();
  });
});
