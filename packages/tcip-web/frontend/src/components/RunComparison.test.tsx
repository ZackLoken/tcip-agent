import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { CompareResult, MetricRow } from "@/api/training";
import { trainingApi } from "@/api/training";
import { RunComparison, type MarkedRun } from "@/components/RunComparison";
import {
  compareBestAnswer,
  compareEntry,
  compareResult,
  metricDirections,
  producedRefusal,
  refusalDetail,
  registryEntry,
} from "@/test/training";

/** The ranking the generator received, by the metric its completed run's checkpoint carries. */
const BEST = compareBestAnswer({});
const METRIC = BEST.ranking_basis;

/** A metric the produced direction table names no direction for. */
const UNDECLARED = "val_map99";

beforeEach(() => {
  // Every render fetches the declared-direction table once on mount; a test that cares about
  // its content overrides this default with its own mockResolvedValue.
  vi.spyOn(trainingApi, "metricDirections").mockResolvedValue(metricDirections({}));
});

afterEach(cleanup);

const MARKED: MarkedRun[] = [{ experimentId: "exp-a" }, { experimentId: "exp-b" }];

/** The produced comparison holding a produced column per marked run, with `change` laid over it. */
function baseResult(change: Partial<CompareResult>): CompareResult {
  return compareResult({
    experiments: MARKED.map(({ experimentId }) => compareEntry({ experiment_id: experimentId })),
    ...change,
  });
}

/** The comparison in which exp-a carries the produced registered checkpoint, under its own name
 *  and carrying `metrics`, and exp-b registered nothing. */
function oneRegisteredResult(metrics: MetricRow | null = BEST.metrics): CompareResult {
  return baseResult({
    experiments: [
      compareEntry({
        experiment_id: "exp-a",
        registry: [registryEntry({ name: "exp-a", metrics })],
      }),
      compareEntry({ experiment_id: "exp-b" }),
    ],
  });
}

/** The produced ranking, won by exp-a. */
const EXP_A_BEST = compareBestAnswer({ name: "exp-a", experiment_id: "exp-a" });

describe("RunComparison data lines", () => {
  it("reads the images line as same source images when the tool says so", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({ same_dataset_fingerprint: true }),
    );
    render(<RunComparison marked={MARKED} />);
    expect(await screen.findByText("same source images")).toBeInTheDocument();
  });

  it("reads the images line as not the same when the tool says so", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({ same_dataset_fingerprint: false }),
    );
    render(<RunComparison marked={MARKED} />);
    expect(await screen.findByText("not the same source images")).toBeInTheDocument();
  });

  it("names which columns carry no record when the images line reads not comparable", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          compareEntry({ experiment_id: "exp-a" }),
          compareEntry({ experiment_id: "exp-b", dataset_fingerprint: null }),
        ],
        same_dataset_fingerprint: null,
      }),
    );
    render(<RunComparison marked={MARKED} />);
    expect(await screen.findByText(/not comparable/)).toBeInTheDocument();
    expect(screen.getByText(/exp-b carry no fingerprint/)).toBeInTheDocument();
  });

  it("reads the partition line from each column's own split state", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          compareEntry({
            experiment_id: "exp-a",
            split: { case: "bound", selection_dir: "splits/d1", seed: 7 },
          }),
          compareEntry({ experiment_id: "exp-b" }),
        ],
      }),
    );
    render(<RunComparison marked={MARKED} />);
    expect(await screen.findByText(/bound to splits\/d1/)).toBeInTheDocument();
    expect(
      screen.getByText(new RegExp(`drawn again \\(seed ${compareEntry({}).split?.seed}\\)`)),
    ).toBeInTheDocument();
  });
});

describe("RunComparison on a change of the marked set", () => {
  it("reads as reading, never a stale absence claim, for a newly marked column", async () => {
    const compare = vi.spyOn(trainingApi, "compare").mockResolvedValue(baseResult({}));
    const { rerender } = render(<RunComparison marked={[MARKED[0]]} />);
    await waitFor(() => expect(screen.getAllByText("exp-a").length).toBeGreaterThan(0));

    compare.mockImplementation(() => new Promise(() => {})); // never resolves for the new set
    rerender(<RunComparison marked={MARKED} />);

    expect(await screen.findByText("Loading comparison...")).toBeInTheDocument();
  });
});

describe("RunComparison rank control", () => {
  it("shows the direction toggle as soon as an undeclared metric is chosen, before any Rank press", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult({ [UNDECLARED]: 0.7 }));
    const compareBest = vi.spyOn(trainingApi, "compareBest");

    render(<RunComparison marked={MARKED} />);
    expect(screen.queryByText("higher is better")).not.toBeInTheDocument();

    fireEvent.change(await screen.findByRole("combobox"), {
      target: { value: UNDECLARED },
    });
    expect(await screen.findByText("higher is better")).toBeInTheDocument();
    expect(screen.getByText("lower is better")).toBeInTheDocument();
    expect(compareBest).not.toHaveBeenCalled();
  });

  it("clears a pending rank refusal when a direction is chosen, since it answers what the refusal asked for", async () => {
    const { message } = refusalDetail("unverified_refusal");
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult({ [UNDECLARED]: 0.7 }));
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      await producedRefusal("unverified_refusal"),
    );

    render(<RunComparison marked={MARKED} />);
    fireEvent.change(await screen.findByRole("combobox"), {
      target: { value: UNDECLARED },
    });
    fireEvent.click(await screen.findByRole("button", { name: "higher is better" }));
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));
    expect(await screen.findByRole("status")).toHaveTextContent(message);

    fireEvent.click(screen.getByRole("button", { name: "lower is better" }));
    expect(screen.queryByText(message)).not.toBeInTheDocument();
  });

  it("shows the unverified switch only after the tool says every candidate was unverified", async () => {
    // The ranked metric's direction is declared, so Rank is reachable without first picking a
    // direction, keeping this test scoped to the unverified switch alone.
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      await producedRefusal("unverified_refusal"),
    );

    render(<RunComparison marked={MARKED} />);
    const select = await screen.findByRole("combobox");
    fireEvent.change(select, { target: { value: METRIC } });
    expect(
      screen.queryByText("include checkpoints the trainer did not stamp"),
    ).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Rank" }));
    expect(
      await screen.findByText("include checkpoints the trainer did not stamp"),
    ).toBeInTheDocument();
  });

  it("renders the projected best-model answer once ranking succeeds", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue(EXP_A_BEST);

    render(<RunComparison marked={MARKED} />);
    const select = await screen.findByRole("combobox");
    fireEvent.change(select, { target: { value: METRIC } });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    const status = await screen.findByRole("status");
    expect(within(status).getByText("exp-a", { selector: "span.font-mono" })).toBeInTheDocument();
    expect(
      within(status).getByText(
        new RegExp(`${BEST.metrics_source} / ${BEST.direction_source} direction`),
      ),
    ).toBeInTheDocument();
  });
});

describe("RunComparison rank exclusions", () => {
  it("names the tool's own no-checkpoint sentence beside a disabled Rank instead of an empty chooser", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(baseResult({}));
    render(<RunComparison marked={MARKED} />);
    expect(
      await screen.findByText("none of the marked experiments registered a checkpoint"),
    ).toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Rank" })).toBeDisabled();
  });

  it("lists a marked run left out for having no registered checkpoint, beside excluded_unverified", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue(EXP_A_BEST);

    render(<RunComparison marked={MARKED} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: METRIC },
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
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue(
      compareBestAnswer({ name: "exp-a", experiment_id: "exp-a", metrics: { [METRIC]: 0.71234 } }),
    );

    render(<RunComparison marked={MARKED} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: METRIC },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    expect(await screen.findByText(new RegExp(`${METRIC}: 0.712`))).toBeInTheDocument();
    expect(screen.queryByText("(exp-a)")).not.toBeInTheDocument();
  });

  it("resets rankResult, rankError, the direction choice, needsUnverifiedOption and rankMetric when the marked set changes", async () => {
    const { message } = refusalDetail("absent_metric_refusal");
    const compare = vi
      .spyOn(trainingApi, "compare")
      .mockResolvedValue(oneRegisteredResult({ [UNDECLARED]: 0.7 }));
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      await producedRefusal("absent_metric_refusal"),
    );

    const { rerender } = render(<RunComparison marked={MARKED} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: UNDECLARED },
    });
    // The metric has no declared direction: the toggle appears immediately, and a direction
    // must be picked before Rank is reachable at all.
    expect(await screen.findByText("higher is better")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "higher is better" }));
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));
    expect(await screen.findByText(message)).toBeInTheDocument();

    compare.mockResolvedValue(
      baseResult({
        experiments: [
          compareEntry({
            experiment_id: "exp-c",
            registry: [registryEntry({ name: "exp-c", metrics: { [UNDECLARED]: 0.5 } })],
          }),
          compareEntry({ experiment_id: "exp-d" }),
        ],
      }),
    );
    rerender(<RunComparison marked={[{ experimentId: "exp-c" }, { experimentId: "exp-d" }]} />);

    await waitFor(() => {
      expect(screen.queryByText("higher is better")).not.toBeInTheDocument();
      expect(screen.queryByText(message)).not.toBeInTheDocument();
    });
    expect(await screen.findByRole("combobox", { name: "Rank by metric" })).toHaveValue("");
  });
});

describe("RunComparison per-column absence and loading", () => {
  it("reads 'not in the comparison answer' for a marked column the answer omits, never 'reading'", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({ experiments: [compareEntry({ experiment_id: "exp-a" })] }),
    );
    render(<RunComparison marked={MARKED} />);
    expect((await screen.findAllByText("not in the comparison answer")).length).toBeGreaterThan(0);
    expect(screen.queryByText("reading")).not.toBeInTheDocument();
  });

  it("renders the Images statement in one table row", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({ same_dataset_fingerprint: true }),
    );
    render(<RunComparison marked={MARKED} />);
    const label = await screen.findByText("Images");
    const row = label.closest("tr") as HTMLElement;
    expect(within(row).getByText("same source images")).toBeInTheDocument();
  });

  it("uses one wording for an absent value and four significant figures for a value", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          compareEntry({
            experiment_id: "exp-a",
            registry: [registryEntry({ name: "exp-a", metrics: {}, metrics_source: null })],
            last_logged_metrics: { epoch: 1, val_map50: 0.712345 },
          }),
          compareEntry({ experiment_id: "exp-b" }),
        ],
      }),
    );
    render(<RunComparison marked={MARKED} />);
    expect(await screen.findByText("0.7123")).toBeInTheDocument();
    expect(screen.queryByText("0.712345")).not.toBeInTheDocument();
    expect((await screen.findAllByText("unrecorded")).length).toBeGreaterThan(0);
    expect(screen.queryByText("no metrics")).not.toBeInTheDocument();
  });
});

describe("RunComparison rank metric chooser", () => {
  it("lists declared-direction keys first, groups the rest under 'no declared direction'", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      oneRegisteredResult({ [UNDECLARED]: 0.2, [METRIC]: 0.5 }),
    );

    render(<RunComparison marked={MARKED} />);
    const rankSelect = await screen.findByRole("combobox", { name: "Rank by metric" });
    await waitFor(() =>
      expect(
        within(rankSelect).getByRole("group", { name: "no declared direction" }),
      ).toBeInTheDocument(),
    );
    const optionTexts = within(rankSelect)
      .getAllByRole("option")
      .map((o) => o.textContent);
    expect(optionTexts).toEqual(["Choose a metric...", METRIC, UNDECLARED]);
  });

  it("never calls compareBest before Rank is pressed", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    const compareBest = vi.spyOn(trainingApi, "compareBest");

    render(<RunComparison marked={MARKED} />);
    const rankSelect = await screen.findByRole("combobox", { name: "Rank by metric" });
    await waitFor(() => expect(within(rankSelect).getAllByRole("option")).toHaveLength(2));

    expect(compareBest).not.toHaveBeenCalled();
  });
});

describe("RunComparison ranking direction control", () => {
  it("gives the direction choice the row group's own role, aria-pressed and focus-visible styling", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult({ [UNDECLARED]: 0.7 }));

    render(<RunComparison marked={MARKED} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: UNDECLARED },
    });

    const group = await screen.findByRole("group", { name: "Ranking direction" });
    const higher = within(group).getByRole("button", { name: "higher is better" });
    expect(higher).toHaveAttribute("aria-pressed", "false");
    expect(higher.className).toContain("focus-visible:ring-tcip-accent/70");
    fireEvent.click(higher);
    expect(higher).toHaveAttribute("aria-pressed", "true");
  });

  it("renders a rank refusal in a status region without moving focus off the Rank control", async () => {
    // The ranked metric's direction is declared, so Rank is reachable right after choosing it;
    // the refusal under test here is unrelated to direction.
    const { message } = refusalDetail("absent_metric_refusal");
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      await producedRefusal("absent_metric_refusal"),
    );

    render(<RunComparison marked={MARKED} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: METRIC },
    });
    const rankButton = screen.getByRole("button", { name: "Rank" });
    rankButton.focus();
    fireEvent.click(rankButton);

    const status = await screen.findByRole("status");
    expect(status).toHaveTextContent(message);
    expect(document.activeElement).toBe(rankButton);
  });
});

describe("RunComparison rank chooser's numeric-only filter", () => {
  it("never offers a key whose stamped value is a string, such as a selection label", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      oneRegisteredResult({ [METRIC]: 0.7, val_selection: "held-out" }),
    );
    render(<RunComparison marked={MARKED} />);
    const rankSelect = await screen.findByRole("combobox", { name: "Rank by metric" });
    const optionTexts = within(rankSelect)
      .getAllByRole("option")
      .map((o) => o.textContent);
    expect(optionTexts).toContain(METRIC);
    expect(optionTexts).not.toContain("val_selection");
  });
});

describe("RunComparison rank answer, a stamped-value exclusion", () => {
  it("names a marked run as not ranked when its checkpoint never stamped the ranked metric", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          compareEntry({
            experiment_id: "exp-a",
            registry: [registryEntry({ name: "exp-a", metrics: { [UNDECLARED]: 0.2 } })],
          }),
          compareEntry({
            experiment_id: "exp-b",
            registry: [registryEntry({ name: "exp-b" })],
          }),
        ],
      }),
    );
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue(
      compareBestAnswer({ name: "exp-b", experiment_id: "exp-b" }),
    );

    render(<RunComparison marked={MARKED} />);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: METRIC },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    expect(
      await screen.findByText(new RegExp(`exp-a: not ranked: no ${METRIC} stamped`)),
    ).toBeInTheDocument();
  });
});

describe("RunComparison disabled Rank's reason", () => {
  it("links the metric-not-chosen reason to the Rank control with aria-describedby", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(oneRegisteredResult());
    render(<RunComparison marked={MARKED} />);
    await screen.findByRole("combobox", { name: "Rank by metric" });

    const rankButton = screen.getByRole("button", { name: "Rank" });
    expect(rankButton).toBeDisabled();
    const describedBy = rankButton.getAttribute("aria-describedby");
    expect(describedBy).toBeTruthy();
    expect(document.getElementById(describedBy as string)).toHaveTextContent(
      "choose a metric before ranking",
    );
  });
});

describe("RunComparison unrecorded-field wording", () => {
  it("reads the builder cell as unrecorded, not a third wording, when the config names none", async () => {
    vi.spyOn(trainingApi, "compare").mockResolvedValue(
      baseResult({
        experiments: [
          compareEntry({ experiment_id: "exp-a", model: null }),
          compareEntry({ experiment_id: "exp-b", model: null }),
        ],
      }),
    );
    render(<RunComparison marked={MARKED} />);
    const label = await screen.findByText("Builder");
    const row = label.closest("tr") as HTMLElement;
    expect(within(row).queryByText("no builder recorded")).not.toBeInTheDocument();
    expect(within(row).getAllByText("unrecorded").length).toBeGreaterThan(0);
  });
});
