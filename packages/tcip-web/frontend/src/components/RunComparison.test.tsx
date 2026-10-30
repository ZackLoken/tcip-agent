import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { trainingApi } from "@/api/training";
import type { Comparison } from "@/api/types.generated";
import { NO_MARKED_CHECKPOINT, RunComparison, type MarkedRun } from "@/components/RunComparison";
import { formatNumber } from "@/lib/numberFormat";
import { thrown } from "@/test/refusals";
import { COMPARISONS, RANKINGS, readColumns, TRAINING_REFUSALS } from "@/test/training";

/** The ranking the generator received, by the one metric its completed run's checkpoint carries. */
const BEST = RANKINGS.compare;
const [METRIC] = Object.keys(BEST.facts.metrics);

/** The registered column of the produced comparison whose checkpoint carries a declared metric,
 *  an undeclared one and a string. */
const [RICH] = readColumns(COMPARISONS.rich).filter((c) => c.registry.length > 0);

/** The direction the produced rich comparison serves for each metric key, `null` where none is
 *  declared. */
const RICH_DIRECTIONS = COMPARISONS.rich.higher_is_better;
const [DECLARED_KEYS, UNDECLARED_KEYS] = [true, false].map((declared) =>
  Object.keys(RICH_DIRECTIONS).filter((key) => (RICH_DIRECTIONS[key] !== null) === declared),
);

/** A metric the produced rich comparison serves with no declared direction. */
const [UNDECLARED] = UNDECLARED_KEYS;

afterEach(cleanup);

/** The id each column of `comparison` was asked for, read or not. */
function idsOf(comparison: Comparison): string[] {
  return comparison.experiments.map((c) => ("run" in c ? c.run.experiment_id : c.experiment_id));
}

/** The runs `comparison` compared, marked in its columns' order. */
function markedOf(comparison: Comparison): MarkedRun[] {
  return idsOf(comparison).map((experimentId) => ({ experimentId }));
}

/** Render the comparison of the runs the produced `comparison` compared, which it answers. */
function compared(comparison: Comparison) {
  const compare = vi.spyOn(trainingApi, "compare").mockResolvedValue(comparison);
  const view = render(<RunComparison marked={markedOf(comparison)} />);
  return { compare, ...view };
}

describe("RunComparison data lines", () => {
  it("reads the images line as same source images when the tool says so", async () => {
    expect(COMPARISONS.same.dataset_fingerprints).toBe("same");
    compared(COMPARISONS.same);
    expect(await screen.findByText("same source images")).toBeInTheDocument();
  });

  it("reads the images line as not the same when the tool says so", async () => {
    expect(COMPARISONS.compare.dataset_fingerprints).toBe("different");
    compared(COMPARISONS.compare);
    expect(await screen.findByText("not the same source images")).toBeInTheDocument();
  });

  it("names which columns it could not read when the images line reads not comparable", async () => {
    const [, unread] = idsOf(COMPARISONS.unread);
    expect(COMPARISONS.unread.dataset_fingerprints).toBe("unread");
    compared(COMPARISONS.unread);
    expect(
      await screen.findByText(`not comparable: ${unread} could not be read`),
    ).toBeInTheDocument();
  });

  it("names which columns carry no fingerprint when the tool says one is unset", async () => {
    const [, loose] = idsOf(COMPARISONS.unset);
    expect(COMPARISONS.unset.dataset_fingerprints).toBe("unset");
    compared(COMPARISONS.unset);
    expect(
      await screen.findByText(`not comparable: ${loose} carry no fingerprint`),
    ).toBeInTheDocument();
  });

  it("reads the partition line from each column's own split state", async () => {
    const [bound, drawn] = readColumns(COMPARISONS.bound).map((c) => c.split);
    compared(COMPARISONS.bound);
    expect(
      await screen.findByText(new RegExp(`bound to ${String(bound.selection_dir)}`)),
    ).toBeInTheDocument();
    expect(
      screen.getByText(new RegExp(`drawn again \\(seed ${String(drawn.seed)}\\)`)),
    ).toBeInTheDocument();
  });
});

describe("RunComparison on a change of the marked set", () => {
  it("reads as reading, never a stale absence claim, for a newly marked column", async () => {
    const [only] = idsOf(COMPARISONS.one);
    const { compare, rerender } = compared(COMPARISONS.one);
    await waitFor(() => expect(screen.getAllByText(only).length).toBeGreaterThan(0));

    compare.mockImplementation(() => new Promise(() => {})); // never resolves for the new set
    rerender(<RunComparison marked={markedOf(COMPARISONS.same)} />);

    expect(await screen.findByText("Loading comparison...")).toBeInTheDocument();
  });
});

describe("RunComparison rank control", () => {
  it("shows the direction toggle as soon as an undeclared metric is chosen, before any Rank press", async () => {
    const compareBest = vi.spyOn(trainingApi, "compareBest");

    compared(COMPARISONS.rich);
    expect(screen.queryByText("higher is better")).not.toBeInTheDocument();

    fireEvent.change(await screen.findByRole("combobox"), {
      target: { value: UNDECLARED },
    });
    expect(await screen.findByText("higher is better")).toBeInTheDocument();
    expect(screen.getByText("lower is better")).toBeInTheDocument();
    expect(compareBest).not.toHaveBeenCalled();
  });

  it("clears a pending rank refusal when a direction is chosen, since it answers what the refusal asked for", async () => {
    const { message } = TRAINING_REFUSALS.unverified_refusal.body.detail;
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      await thrown(TRAINING_REFUSALS.unverified_refusal),
    );

    compared(COMPARISONS.rich);
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
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      await thrown(TRAINING_REFUSALS.unverified_refusal),
    );

    compared(COMPARISONS.compare);
    const select = await screen.findByRole("combobox");
    fireEvent.change(select, { target: { value: METRIC } });
    expect(screen.queryByText("include unverified checkpoints")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Rank" }));
    expect(await screen.findByText("include unverified checkpoints")).toBeInTheDocument();
  });

  it("renders the projected best-model answer once ranking succeeds", async () => {
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue(BEST);

    compared(COMPARISONS.compare);
    const select = await screen.findByRole("combobox");
    fireEvent.change(select, { target: { value: METRIC } });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    const status = await screen.findByRole("status");
    expect(
      within(status).getByText(BEST.entry.name, { selector: "span.font-mono" }),
    ).toBeInTheDocument();
    expect(
      within(status).getByText(
        new RegExp(`${BEST.facts.metrics_source} / ${BEST.direction_source} direction`),
      ),
    ).toBeInTheDocument();
  });
});

describe("RunComparison rank exclusions", () => {
  it("says no column carries a registered checkpoint beside a disabled Rank instead of an empty chooser", async () => {
    expect(readColumns(COMPARISONS.same).every((c) => c.registry.length === 0)).toBe(true);
    compared(COMPARISONS.same);
    expect(await screen.findByText(NO_MARKED_CHECKPOINT)).toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Rank" })).toBeDisabled();
  });

  it("says once, in its column, that a marked run the ranking answers as unregistered has no checkpoint", async () => {
    const [unregistered] = readColumns(COMPARISONS.compare)
      .filter((c) => c.registry.length === 0)
      .map((c) => c.run.experiment_id);
    expect(BEST.unregistered).toEqual([unregistered]);
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue(BEST);

    compared(COMPARISONS.compare);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: METRIC },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    expect(await screen.findByRole("status")).not.toHaveTextContent("no registered checkpoint");
    const column = idsOf(COMPARISONS.compare).indexOf(unregistered);
    const entries = screen.getByText("Entries").closest("tr") as HTMLElement;
    expect(within(entries).getAllByRole("cell")[column]).toHaveTextContent(
      "no registered checkpoint",
    );
    expect(screen.getAllByText("no registered checkpoint")).toHaveLength(1);
  });
});

describe("RunComparison rank answer", () => {
  it("shows the winner's own value for the ranked metric, and its name once when it equals the experiment id", async () => {
    const winner = RANKINGS.rich;
    expect(winner.entry.name).toBe(winner.entry.experiment_id);
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue(winner);

    compared(COMPARISONS.rich);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: METRIC },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    // The winner's value carries more than four significant figures, which the answer rounds.
    const value = winner.facts.metrics[METRIC] as number;
    expect(
      await screen.findByText(new RegExp(`${METRIC}: ${formatNumber(value)}`)),
    ).toBeInTheDocument();
    expect(screen.queryByText(new RegExp(`${METRIC}: ${value}`))).not.toBeInTheDocument();
    expect(screen.queryByText(`(${winner.entry.name})`)).not.toBeInTheDocument();
  });

  it("resets rankResult, rankError, the direction choice, needsUnverifiedOption and rankMetric when the marked set changes", async () => {
    const { message } = TRAINING_REFUSALS.absent_metric_refusal.body.detail;
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      await thrown(TRAINING_REFUSALS.absent_metric_refusal),
    );

    const { compare, rerender } = compared(COMPARISONS.rich);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: UNDECLARED },
    });
    // The metric has no declared direction: the toggle appears immediately, and a direction
    // must be picked before Rank is reachable at all.
    expect(await screen.findByText("higher is better")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "higher is better" }));
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));
    expect(await screen.findByText(message)).toBeInTheDocument();

    compare.mockResolvedValue(COMPARISONS.unranked);
    rerender(<RunComparison marked={markedOf(COMPARISONS.unranked)} />);

    await waitFor(() => {
      expect(screen.queryByText("higher is better")).not.toBeInTheDocument();
      expect(screen.queryByText(message)).not.toBeInTheDocument();
    });
    expect(await screen.findByRole("combobox", { name: "Rank by metric" })).toHaveValue("");
  });
});

describe("RunComparison per-column absence and loading", () => {
  it("reads 'not in the comparison answer' for a marked column the answer omits, never 'reading'", async () => {
    // The answer of one run's comparison, for a marked set holding a second run.
    vi.spyOn(trainingApi, "compare").mockResolvedValue(COMPARISONS.one);
    render(<RunComparison marked={markedOf(COMPARISONS.same)} />);
    expect((await screen.findAllByText("not in the comparison answer")).length).toBeGreaterThan(0);
    expect(screen.queryByText("reading")).not.toBeInTheDocument();
  });

  it("renders the Images statement in one table row", async () => {
    compared(COMPARISONS.same);
    const label = await screen.findByText("Images");
    const row = label.closest("tr") as HTMLElement;
    expect(within(row).getByText("same source images")).toBeInTheDocument();
  });

  it("uses one wording for an absent value and four significant figures for a value", async () => {
    // The unstamped column's checkpoint carries no metric while its log carries one with more
    // than four significant figures.
    const [unstamped] = readColumns(COMPARISONS.unstamped);
    const logged = unstamped.last_logged_metrics?.[METRIC] as number;
    compared(COMPARISONS.unstamped);
    expect(await screen.findByText(formatNumber(logged))).toBeInTheDocument();
    expect(screen.queryByText(String(logged))).not.toBeInTheDocument();
    expect((await screen.findAllByText("unrecorded")).length).toBeGreaterThan(0);
    expect(screen.queryByText("no metrics")).not.toBeInTheDocument();
  });
});

describe("RunComparison rank metric chooser", () => {
  it("lists declared-direction keys first, groups the rest under 'no declared direction'", async () => {
    compared(COMPARISONS.rich);
    const rankSelect = await screen.findByRole("combobox", { name: "Rank by metric" });
    await waitFor(() =>
      expect(
        within(rankSelect).getByRole("group", { name: "no declared direction" }),
      ).toBeInTheDocument(),
    );
    const optionTexts = within(rankSelect)
      .getAllByRole("option")
      .map((o) => o.textContent);
    expect(optionTexts).toEqual(["Choose a metric...", ...DECLARED_KEYS, ...UNDECLARED_KEYS]);
  });

  it("never calls compareBest before Rank is pressed", async () => {
    const compareBest = vi.spyOn(trainingApi, "compareBest");

    compared(COMPARISONS.compare);
    const rankSelect = await screen.findByRole("combobox", { name: "Rank by metric" });
    await waitFor(() => expect(within(rankSelect).getAllByRole("option")).toHaveLength(2));

    expect(compareBest).not.toHaveBeenCalled();
  });
});

describe("RunComparison ranking direction control", () => {
  it("gives the direction choice the row group's own role, aria-pressed and focus-visible styling", async () => {
    compared(COMPARISONS.rich);
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
    const { message } = TRAINING_REFUSALS.absent_metric_refusal.body.detail;
    vi.spyOn(trainingApi, "compareBest").mockRejectedValue(
      await thrown(TRAINING_REFUSALS.absent_metric_refusal),
    );

    compared(COMPARISONS.compare);
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
    const [stringKey] = Object.entries(RICH.registry[0].facts.metrics)
      .filter(([, value]) => typeof value === "string")
      .map(([key]) => key);
    compared(COMPARISONS.rich);
    const rankSelect = await screen.findByRole("combobox", { name: "Rank by metric" });
    const optionTexts = within(rankSelect)
      .getAllByRole("option")
      .map((o) => o.textContent);
    expect(optionTexts.slice(1).sort()).toEqual(Object.keys(RICH_DIRECTIONS).sort());
    expect(optionTexts).not.toContain(stringKey);
  });
});

describe("RunComparison rank answer, a stamped-value exclusion", () => {
  it("names a checkpoint the ranking skipped for never stamping the ranked metric, with why", async () => {
    const ranking = RANKINGS.unranked;
    const skipped = ranking.skipped.find((s) => s.unstamped);
    if (skipped === undefined) throw new Error("the produced ranking skipped no unstamped entry");
    expect(skipped.unverified).toBe(false);
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue(ranking);

    compared(COMPARISONS.unranked);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: ranking.metric },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    expect(
      await screen.findByText(
        new RegExp(`${skipped.name}: not ranked: no ${ranking.metric} stamped`),
      ),
    ).toBeInTheDocument();
  });

  it("names every fact a checkpoint was left out for when both hold, by the metric ranked", async () => {
    const ranking = RANKINGS.verified;
    const [both] = ranking.skipped.filter((s) => s.unverified && s.unstamped);
    const directions = COMPARISONS.verified.higher_is_better;
    // A declared metric other than the one ranked by: the sentence names the answer's metric.
    const [chosen] = Object.keys(directions).filter(
      (key) => directions[key] !== null && key !== ranking.metric,
    );
    vi.spyOn(trainingApi, "compareBest").mockResolvedValue(ranking);

    compared(COMPARISONS.verified);
    fireEvent.change(await screen.findByRole("combobox", { name: "Rank by metric" }), {
      target: { value: chosen },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rank" }));

    expect(await screen.findByRole("status")).toHaveTextContent(
      `${both.name}: not ranked: unverified, its metrics are not the trainer's; no ${ranking.metric} stamped`,
    );
  });
});

describe("RunComparison disabled Rank's reason", () => {
  it("links the metric-not-chosen reason to the Rank control with aria-describedby", async () => {
    compared(COMPARISONS.compare);
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
  it("reads the builder cell as unrecorded, not a third wording, for a run it could not read", async () => {
    compared(COMPARISONS.unread);
    const label = await screen.findByText("Builder");
    const row = label.closest("tr") as HTMLElement;
    expect(within(row).queryByText("no builder recorded")).not.toBeInTheDocument();
    expect(within(row).getAllByText("unrecorded").length).toBeGreaterThan(0);
  });
});
