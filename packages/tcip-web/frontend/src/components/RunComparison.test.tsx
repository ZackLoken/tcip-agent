import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { StructuredRefusalError } from "@/api/http";
import type { CompareResult } from "@/api/training";
import { openTrainingStream, trainingApi } from "@/api/training";
import { RunComparison, type MarkedRun } from "@/components/RunComparison";

// The overlay chart owns its own per-run WebSocket streams; only the comparison's own rendering
// is under test here, so the transport is replaced while the rest of the module stays real.
vi.mock("@/api/training", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/training")>();
  return { ...actual, openTrainingStream: vi.fn(() => () => {}) };
});

beforeEach(() => {
  // Every render fetches the declared-direction table once on mount; a test that cares about
  // its content overrides this default with its own mockResolvedValue.
  vi.spyOn(trainingApi, "metricDirections").mockResolvedValue({ higher_is_better: {} });
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

// exp-a carries one registered, verified entry stamping val_map99; exp-b registered nothing.
function oneRegisteredResult(): CompareResult {
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

  it("shows the direction toggle as soon as an undeclared metric is chosen, before any Rank press", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(registeredResult());
    const compareBest = vi.spyOn(trainingApi, "compareBest");

    render(<RunComparison marked={MARKED} projectRoot={null} />);
    expect(screen.queryByText("higher is better")).not.toBeInTheDocument();

    fireEvent.change(await screen.findByRole("combobox"), {
      target: { value: "val_map99" },
    });
    expect(await screen.findByText("higher is better")).toBeInTheDocument();
    expect(screen.getByText("lower is better")).toBeInTheDocument();
    expect(compareBest).not.toHaveBeenCalled();
  });

  it("clears a pending rank refusal when a direction is chosen, since it answers what the refusal asked for", async () => {
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
    fireEvent.change(await screen.findByRole("combobox"), {
      target: { value: "val_map99" },
    });
    fireEvent.click(await screen.findByRole("button", { name: "higher is better" }));
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));
    expect(await screen.findByRole("status")).toHaveTextContent(/every registered model/);

    fireEvent.click(screen.getByRole("button", { name: "lower is better" }));
    expect(screen.queryByText(/every registered model/)).not.toBeInTheDocument();
  });

  it("shows the unverified switch only after the tool says every candidate was unverified", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(registeredResult());
    // The direction is already declared for this metric so Rank is reachable without first
    // picking a direction, keeping this test scoped to the unverified switch alone.
    vi.spyOn(trainingApi, "metricDirections").mockResolvedValue({
      higher_is_better: { map99: true },
    });
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
    vi.spyOn(trainingApi, "metricDirections").mockResolvedValue({
      higher_is_better: { map99: true },
    });
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

    const status = await screen.findByRole("status");
    expect(within(status).getByText("exp-a", { selector: "span.font-mono" })).toBeInTheDocument();
    expect(within(status).getByText(/trainer \/ stated direction/)).toBeInTheDocument();
  });
});

describe("RunComparison rank exclusions", () => {
  it("names the tool's own no-checkpoint sentence beside a disabled Rank instead of an empty chooser", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          { experiment_id: "exp-a", registry: [] },
          { experiment_id: "exp-b", registry: [] },
        ],
      }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    expect(
      await screen.findByText("none of the marked experiments registered a checkpoint"),
    ).toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Rank" })).toBeDisabled();
  });

  it("lists a marked run left out for having no registered checkpoint, beside excluded_unverified", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    vi.spyOn(trainingApi, "metricDirections").mockResolvedValue({
      higher_is_better: { map99: true },
    });
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue({
      name: "exp-a",
      experiment_id: "exp-a",
      metrics: { val_map99: 0.7 },
      metrics_source: "trainer",
      higher_is_better: true,
      direction_source: "declared",
      excluded_unverified: [],
    });

    render(<RunComparison marked={MARKED} projectRoot={null} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: "val_map99" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    expect(
      await screen.findByText(/exp-b: not ranked: no registered checkpoint/),
    ).toBeInTheDocument();
  });
});

describe("RunComparison rank answer", () => {
  it("shows the winner's own value for the ranked metric, and its name once when it equals the experiment id", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    vi.spyOn(trainingApi, "metricDirections").mockResolvedValue({
      higher_is_better: { map99: true },
    });
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue({
      name: "exp-a",
      experiment_id: "exp-a",
      metrics: { val_map99: 0.71234 },
      metrics_source: "trainer",
      higher_is_better: true,
      direction_source: "declared",
      excluded_unverified: [],
    });

    render(<RunComparison marked={MARKED} projectRoot={null} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: "val_map99" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    expect(await screen.findByText(/val_map99: 0.712/)).toBeInTheDocument();
    expect(screen.queryByText("(exp-a)")).not.toBeInTheDocument();
  });

  it("resets rankResult, rankError, the direction choice, needsUnverifiedOption and rankMetric when the marked set changes", async () => {
    const compare = vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      new StructuredRefusalError({ error: "boom" }, 422, ""),
    );

    const { rerender } = render(<RunComparison marked={MARKED} projectRoot={null} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: "val_map99" },
    });
    // val_map99 has no declared direction (the default beforeEach table): the toggle appears
    // immediately, and a direction must be picked before Rank is reachable at all.
    expect(await screen.findByText("higher is better")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "higher is better" }));
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));
    expect(await screen.findByText("boom")).toBeInTheDocument();

    compare.mockResolvedValue(
      baseResult({
        experiments: [
          {
            experiment_id: "exp-c",
            registry: [
              {
                name: "exp-c",
                metrics: { val_map99: 0.5 },
                metrics_source: "trainer",
                registered_at: null,
              },
            ],
          },
          { experiment_id: "exp-d", registry: [] },
        ],
      }),
    );
    rerender(
      <RunComparison
        marked={[
          { runId: "run-c", experimentId: "exp-c" },
          { runId: "run-d", experimentId: "exp-d" },
        ]}
        projectRoot={null}
      />,
    );

    await waitFor(() => {
      expect(screen.queryByText("higher is better")).not.toBeInTheDocument();
      expect(screen.queryByText("boom")).not.toBeInTheDocument();
    });
    expect(await screen.findByRole("combobox", { name: "Rank by metric" })).toHaveValue("");
  });
});

describe("RunComparison per-column absence and loading", () => {
  it("reads 'not in the comparison answer' for a marked column the answer omits, never 'reading'", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({ experiments: [{ experiment_id: "exp-a" }] }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    expect((await screen.findAllByText("not in the comparison answer")).length).toBeGreaterThan(0);
    expect(screen.queryByText("reading")).not.toBeInTheDocument();
  });

  it("renders the Images statement in one table row", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({ same_dataset_fingerprint: true }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    const label = await screen.findByText("Images");
    const row = label.closest("tr") as HTMLElement;
    expect(within(row).getByText("same source images")).toBeInTheDocument();
  });

  it("uses one wording for an absent value and the runs list's three-decimal convention", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          {
            experiment_id: "exp-a",
            registry: [{ name: "exp-a", metrics: {}, metrics_source: null, registered_at: null }],
            last_logged_metrics: { epoch: 1, val_map50: 0.712345 },
          },
          { experiment_id: "exp-b", registry: [] },
        ],
      }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    expect(await screen.findByText("0.712")).toBeInTheDocument();
    expect(screen.queryByText("0.7123")).not.toBeInTheDocument();
    expect((await screen.findAllByText("unrecorded")).length).toBeGreaterThan(0);
    expect(screen.queryByText("no metrics")).not.toBeInTheDocument();
  });
});

describe("RunComparison overlay", () => {
  it("tells the breeder to open the project instead of waiting silently with no project root", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(baseResult({}));
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    expect(await screen.findByText("open the project to stream metrics")).toBeInTheDocument();
    expect(screen.queryByText("Waiting for metrics...")).not.toBeInTheDocument();
  });

  it("names both metric choosers accessibly", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    vi.mocked(openTrainingStream).mockImplementation((_root, _runId, onMessage) => {
      onMessage({ type: "metric", row: { epoch: 1, loss: 0.3 } } as never);
      return () => {};
    });
    render(<RunComparison marked={MARKED} projectRoot="/proj" />);
    expect(await screen.findByRole("combobox", { name: "Overlay metric" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Rank by metric" })).toBeInTheDocument();
  });
});

describe("RunComparison rank metric chooser", () => {
  it("lists declared-direction keys first, groups the rest under 'no declared direction'", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          {
            experiment_id: "exp-a",
            registry: [
              {
                name: "exp-a",
                metrics: { val_map50: 0.5, val_loss: 0.2 },
                metrics_source: "trainer",
                registered_at: null,
              },
            ],
          },
          { experiment_id: "exp-b", registry: [] },
        ],
      }),
    );
    // The table's own keys are bare (val_-stripped): map50 declares a direction, loss does not.
    vi.spyOn(trainingApi, "metricDirections").mockResolvedValue({
      higher_is_better: { map50: true },
    });

    render(<RunComparison marked={MARKED} projectRoot={null} />);
    const rankSelect = await screen.findByRole("combobox", { name: "Rank by metric" });
    await waitFor(() =>
      expect(
        within(rankSelect).getByRole("group", { name: "no declared direction" }),
      ).toBeInTheDocument(),
    );
    const optionTexts = within(rankSelect)
      .getAllByRole("option")
      .map((o) => o.textContent);
    expect(optionTexts).toEqual(["Choose a metric...", "val_map50", "val_loss"]);
  });

  it("never calls compareBest before Rank is pressed", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    vi.spyOn(trainingApi, "metricDirections").mockResolvedValue({
      higher_is_better: { map99: true },
    });
    const compareBest = vi.spyOn(trainingApi, "compareBest");

    render(<RunComparison marked={MARKED} projectRoot={null} />);
    const rankSelect = await screen.findByRole("combobox", { name: "Rank by metric" });
    await waitFor(() => expect(within(rankSelect).getAllByRole("option")).toHaveLength(2));

    expect(compareBest).not.toHaveBeenCalled();
  });
});

describe("RunComparison ranking direction control", () => {
  it("gives the direction choice the row group's own role, aria-pressed and focus-visible styling", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());

    render(<RunComparison marked={MARKED} projectRoot={null} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: "val_map99" },
    });

    const group = await screen.findByRole("group", { name: "Ranking direction" });
    const higher = within(group).getByRole("button", { name: "higher is better" });
    expect(higher).toHaveAttribute("aria-pressed", "false");
    expect(higher.className).toContain("focus-visible:ring-tcip-accent/70");
    fireEvent.click(higher);
    expect(higher).toHaveAttribute("aria-pressed", "true");
  });

  it("renders a rank refusal in a status region without moving focus off the Rank control", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    // The direction is already declared for this metric, so Rank is reachable right after
    // choosing it; the refusal under test here is unrelated to direction.
    vi.spyOn(trainingApi, "metricDirections").mockResolvedValue({
      higher_is_better: { map99: true },
    });
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      new StructuredRefusalError({ error: "boom" }, 422, ""),
    );

    render(<RunComparison marked={MARKED} projectRoot={null} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: "val_map99" },
    });
    const rankButton = screen.getByRole("button", { name: "Rank" });
    rankButton.focus();
    fireEvent.click(rankButton);

    const status = await screen.findByRole("status");
    expect(status).toHaveTextContent("boom");
    expect(document.activeElement).toBe(rankButton);
  });
});

describe("RunComparison overlay x axis", () => {
  it("labels the overlay's x axis the same as the single-run chart", async () => {
    const rect = { width: 600, height: 300, top: 0, left: 0, bottom: 300, right: 600 } as DOMRect;
    const original = HTMLElement.prototype.getBoundingClientRect;
    HTMLElement.prototype.getBoundingClientRect = () => rect;
    try {
      vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
      vi.mocked(openTrainingStream).mockImplementation((_root, _runId, onMessage) => {
        onMessage({ type: "metric", row: { epoch: 1, loss: 0.3 } } as never);
        return () => {};
      });
      render(<RunComparison marked={MARKED} projectRoot="/proj" />);
      expect(await screen.findByText("epoch/step")).toBeInTheDocument();
    } finally {
      HTMLElement.prototype.getBoundingClientRect = original;
    }
  });
});

describe("RunComparison rank chooser's numeric-only filter", () => {
  it("never offers a key whose stamped value is a string, such as a selection label", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          {
            experiment_id: "exp-a",
            registry: [
              {
                name: "exp-a",
                metrics: { val_map99: 0.7, val_selection: "held-out" },
                metrics_source: "trainer",
                registered_at: null,
              },
            ],
          },
          { experiment_id: "exp-b", registry: [] },
        ],
      }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    const rankSelect = await screen.findByRole("combobox", { name: "Rank by metric" });
    const optionTexts = within(rankSelect)
      .getAllByRole("option")
      .map((o) => o.textContent);
    expect(optionTexts).toContain("val_map99");
    expect(optionTexts).not.toContain("val_selection");
  });
});

describe("RunComparison rank answer, a stamped-value exclusion", () => {
  it("names a marked run as not ranked when its checkpoint never stamped the ranked metric", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          {
            experiment_id: "exp-a",
            registry: [
              {
                name: "exp-a",
                metrics: { val_loss: 0.2 },
                metrics_source: "trainer",
                registered_at: null,
              },
            ],
          },
          {
            experiment_id: "exp-b",
            registry: [
              {
                name: "exp-b",
                metrics: { val_map99: 0.6 },
                metrics_source: "trainer",
                registered_at: null,
              },
            ],
          },
        ],
      }),
    );
    vi.spyOn(trainingApi, "metricDirections").mockResolvedValue({
      higher_is_better: { map99: true },
    });
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue({
      name: "exp-b",
      experiment_id: "exp-b",
      metrics: { val_map99: 0.6 },
      metrics_source: "trainer",
      higher_is_better: true,
      direction_source: "declared",
      excluded_unverified: [],
    });

    render(<RunComparison marked={MARKED} projectRoot={null} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: "val_map99" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    expect(await screen.findByText(/exp-a: not ranked: no val_map99 stamped/)).toBeInTheDocument();
  });
});

describe("RunComparison disabled Rank's reason", () => {
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

  it("links the metric-not-chosen reason to the Rank control with aria-describedby", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(registeredResult());
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    await screen.findByRole("combobox", { name: "Rank by metric" });

    const rankButton = screen.getByRole("button", { name: "Rank" });
    expect(rankButton).toBeDisabled();
    const describedBy = rankButton.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy as string)).toHaveTextContent(
      "choose a metric before ranking",
    );
  });

  it("says the registry is unreadable for a marked run beside a disabled Rank, instead of an empty chooser", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          {
            experiment_id: "exp-a",
            registry_error: "registry unreadable: simulated decode failure",
          },
          { experiment_id: "exp-b", registry: [] },
        ],
      }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);

    const reason = await screen.findByText(
      "registry unreadable for a marked run; see the checkpoints above",
    );
    expect(screen.queryByRole("combobox", { name: "Rank by metric" })).not.toBeInTheDocument();
    const rankButton = screen.getByRole("button", { name: "Rank" });
    expect(rankButton).toBeDisabled();
    expect(rankButton).toHaveAttribute("aria-describedby", reason.id);
  });
});

describe("RunComparison unrecorded-field wording", () => {
  it("reads the builder cell as unrecorded, not a third wording, when the config names none", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          { experiment_id: "exp-a", model: null },
          { experiment_id: "exp-b", model: null },
        ],
      }),
    );
    render(<RunComparison marked={MARKED} projectRoot={null} />);
    const label = await screen.findByText("Builder");
    const row = label.closest("tr") as HTMLElement;
    expect(within(row).queryByText("no builder recorded")).not.toBeInTheDocument();
    expect(within(row).getAllByText("unrecorded").length).toBeGreaterThan(0);
  });
});
