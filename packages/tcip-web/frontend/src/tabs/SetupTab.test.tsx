import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import { resultsApi } from "@/api/inference";
import { type CaptureInvariances, type ServedMapping } from "@/api/types.generated";
import { answerPhrase, invarianceName } from "@/lib/invariances";
import { openable } from "@/lib/openProject";
import { useStore } from "@/store";
import { SetupTab } from "@/tabs/SetupTab";
import { RESULTS_ANSWERS } from "@/test/deliveryRecords";
import { PROJECT_FIXTURES } from "@/test/projectRecords";
import { thrown } from "@/test/refusals";
import { openTestProject, RAW_SELECTION } from "@/test/store";
import { CONFIRM_ANSWERS, TRAIT_LISTINGS } from "@/test/traitRecords";

const initialStoreState = useStore.getState();

const { listing: LISTING, correction: CORRECTION, refusal: REFUSAL } = PROJECT_FIXTURES;
/** The listed project the produced correction corrected. */
const CORRECTED_ID = CORRECTION.record.id;
const CORRECTED = LISTING.projects.filter(openable).find((p) => p.record.id === CORRECTED_ID)!;
/** An invariances block's answers, field by field. */
const answers = (block: CaptureInvariances) =>
  Object.entries(block) as [string, boolean | number][];

/** A text matcher for each innermost element whose text holds `text`. */
function innermostHolding(text: string) {
  const holds = (element: Element | null) => Boolean(element?.textContent?.includes(text));
  return (_content: string, element: Element | null) =>
    holds(element) && !Array.from(element?.children ?? []).some(holds);
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  openTestProject(RAW_SELECTION);
  vi.spyOn(resultsApi, "traits").mockResolvedValue({ traits: [], unreadable: [], definitions: {} });
  vi.spyOn(resultsApi, "listPlantMappings").mockResolvedValue(RESULTS_ANSWERS.no_mapping_names);
  vi.spyOn(api.projects, "list").mockResolvedValue(LISTING);
});

describe("SetupTab capture invariances", () => {
  const ROTATION = "rotation (degrees)";

  beforeEach(() => {
    expect(PROJECT_FIXTURES.corrected_open.project.id).toBe(CORRECTED_ID);
    useStore.setState({ openProject: PROJECT_FIXTURES.corrected_open.project });
  });

  it("shows the six answers the open project's record carries", async () => {
    render(<SetupTab />);

    await screen.findByLabelText(ROTATION);
    for (const [name, value] of answers(CORRECTED.record.invariances)) {
      if (typeof value === "boolean") {
        expect(screen.getByLabelText(name.replaceAll("_", " "))).toHaveValue(value ? "yes" : "no");
      } else {
        expect(screen.getByLabelText(ROTATION)).toHaveValue(value);
      }
    }
    expect(screen.getByRole("button", { name: "Saved" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Save" })).not.toBeInTheDocument();
  });

  it("posts the whole block with the person's name and names what changed", async () => {
    const correct = vi.spyOn(api.projects, "invariances").mockResolvedValue(CORRECTION);
    useStore.setState({ user: "breeder" });

    render(<SetupTab />);
    await screen.findByLabelText(ROTATION);
    for (const [name, value] of answers(CORRECTION.record.invariances)) {
      fireEvent.change(
        screen.getByLabelText(typeof value === "boolean" ? name.replaceAll("_", " ") : ROTATION),
        { target: { value: typeof value === "boolean" ? (value ? "yes" : "no") : String(value) } },
      );
    }
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(correct).toHaveBeenCalledWith({
        id: CORRECTED_ID,
        invariances: CORRECTION.record.invariances,
        user: "breeder",
      }),
    );
    const before = CORRECTED.record.invariances;
    const changed = answers(CORRECTION.record.invariances)
      .filter(([name, value]) => before[name as keyof CaptureInvariances] !== value)
      .map(([name, value]) => {
        const field = name as keyof CaptureInvariances;
        return `${invarianceName(field)} ${answerPhrase(before[field])} to ${answerPhrase(value)}`;
      });
    expect(changed).not.toHaveLength(0);
    await waitFor(() =>
      expect(useStore.getState().toasts.map((t) => t.message)).toContain(
        `Saved: ${changed.join(", ")}.`,
      ),
    );
  });

  it("shows the door's refusal text in the panel", async () => {
    const refused = await thrown(REFUSAL);
    vi.spyOn(api.projects, "invariances").mockRejectedValue(refused);

    render(<SetupTab />);
    fireEvent.change(await screen.findByLabelText(ROTATION), { target: { value: "400" } });
    fireEvent.click(screen.getByRole("button", { name: "Save" }));

    expect(await screen.findByText(refused.message)).toBeInTheDocument();
  });
});

afterEach(cleanup);

describe("SetupTab trait revisions", () => {
  const LISTING = TRAIT_LISTINGS.setup;
  const [{ trait: TRAIT, revisions }] = LISTING.traits;
  const [FIRST, SECOND] = revisions;
  const TRAIT_ROW = `trait-${TRAIT}`;
  const statementOf = (revision: typeof FIRST) =>
    revision.entry.operationalizations.per_image_count.statement;

  it("lists every revision, shows the latest, and names the revision deliveries read", async () => {
    vi.spyOn(resultsApi, "traits").mockResolvedValue(LISTING);

    render(<SetupTab />);
    const row = await screen.findByTestId(TRAIT_ROW);

    const listed = within(row).getByRole("list", { name: `${TRAIT} revisions` });
    expect(
      within(listed)
        .getAllByRole("button")
        .map((b) => b.textContent),
    ).toEqual([`Revision ${FIRST.number}`, `Revision ${SECOND.number}`]);
    expect(within(row).getByText(`Confirmed, revision ${FIRST.number}`)).toBeInTheDocument();
    expect(within(row).getByText(statementOf(SECOND))).toBeInTheDocument();
    expect(within(row).getByText("Not confirmed")).toBeInTheDocument();

    fireEvent.click(within(listed).getByRole("button", { name: `Revision ${FIRST.number}` }));
    expect(within(row).getByText(statementOf(FIRST))).toBeInTheDocument();
    expect(
      within(row).getByText(new RegExp(`Confirmed by ${String(FIRST.confirmed_by)}`)),
    ).toBeInTheDocument();
  });

  it("confirms the revision shown with that revision's own hash", async () => {
    const traits = vi.spyOn(resultsApi, "traits").mockResolvedValue(LISTING);
    const confirmSpy = vi
      .spyOn(resultsApi, "confirmTraitRevision")
      .mockResolvedValue(CONFIRM_ANSWERS.confirmed);

    useStore.setState({ user: "breeder" });
    render(<SetupTab />);
    const row = await screen.findByTestId(TRAIT_ROW);
    fireEvent.click(within(row).getByRole("button", { name: "Confirm" }));

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    expect(confirmSpy.mock.calls[0][0]).toEqual({
      trait: TRAIT,
      revision: SECOND.number,
      entry_sha256: SECOND.entry_sha256,
      user: "breeder",
      confirmed: true,
    });
    await waitFor(() => expect(traits).toHaveBeenCalledTimes(2));
  });

  it("withdraws the confirmation of the revision shown", async () => {
    vi.spyOn(resultsApi, "traits").mockResolvedValue(LISTING);
    const confirmSpy = vi
      .spyOn(resultsApi, "confirmTraitRevision")
      .mockResolvedValue(CONFIRM_ANSWERS.withdrawn);

    render(<SetupTab />);
    const row = await screen.findByTestId(TRAIT_ROW);
    fireEvent.click(within(row).getByRole("button", { name: `Revision ${FIRST.number}` }));
    expect(within(row).queryByRole("button", { name: "Confirm" })).toBeNull();
    fireEvent.click(within(row).getByRole("button", { name: "Withdraw" }));

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    expect(confirmSpy.mock.calls[0][0]).toMatchObject({
      revision: FIRST.number,
      entry_sha256: FIRST.entry_sha256,
      confirmed: false,
    });
  });

  it("shows the door's own sentence when it answers that the revision moved", async () => {
    const moved = CONFIRM_ANSWERS.revision_moved;
    vi.spyOn(resultsApi, "traits").mockResolvedValue(LISTING);
    vi.spyOn(resultsApi, "confirmTraitRevision").mockRejectedValue(await thrown(moved));

    render(<SetupTab />);
    const row = await screen.findByTestId(TRAIT_ROW);
    fireEvent.click(within(row).getByRole("button", { name: "Confirm" }));

    expect(
      await within(row).findByText(moved.body.detail.message, { exact: false }),
    ).toBeInTheDocument();
  });

  it("shows a committed confirmation's audit warning without calling it a failure", async () => {
    vi.spyOn(resultsApi, "traits").mockResolvedValue(LISTING);
    const unrecorded = CONFIRM_ANSWERS.unrecorded;
    vi.spyOn(resultsApi, "confirmTraitRevision").mockResolvedValue(unrecorded);

    render(<SetupTab />);
    const row = await screen.findByTestId(TRAIT_ROW);
    fireEvent.click(within(row).getByRole("button", { name: "Confirm" }));

    expect(await screen.findByText(unrecorded.audit_warning!)).toBeInTheDocument();
  });

  it("marks a trait with no confirmed revision as not confirmed", async () => {
    const listing = TRAIT_LISTINGS.unconfirmed;
    vi.spyOn(resultsApi, "traits").mockResolvedValue(listing);

    render(<SetupTab />);
    const row = await screen.findByTestId(`trait-${listing.traits[0].trait}`);

    expect(within(row).getAllByText("Not confirmed").length).toBeGreaterThan(0);
    expect(within(row).queryByText(/Confirmed, revision/)).toBeNull();
  });

  it("names a trait whose record will not read", async () => {
    const listing = TRAIT_LISTINGS.unreadable;
    vi.spyOn(resultsApi, "traits").mockResolvedValue(listing);
    const [{ trait, reason }] = listing.unreadable;

    render(<SetupTab />);

    expect(await screen.findByText(innermostHolding(`${trait}: ${reason}`))).toBeInTheDocument();
  });
});

describe("SetupTab plant-mapping build: match radius", () => {
  const {
    mapping_built: BUILT,
    mapping_one_placed: ONE_PLACED,
    mapping_stated: STATED,
    mapping_unpositioned: UNPOSITIONED,
    mapping_loaded: LOADED,
  } = RESULTS_ANSWERS;

  function fillAndBuild() {
    fireEvent.change(screen.getByLabelText("Mapping"), {
      target: { value: "valley-2026" },
    });
    fireEvent.change(screen.getByLabelText("Plant registry"), {
      target: { value: "valley-plants" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Build" }));
  }

  async function buildAnswering(mapping: ServedMapping) {
    vi.spyOn(resultsApi, "buildPlantMapping").mockResolvedValue(mapping);

    render(<SetupTab />);
    await waitFor(() => expect(resultsApi.listPlantMappings).toHaveBeenCalled());
    fillAndBuild();
    await waitFor(() => expect(resultsApi.buildPlantMapping).toHaveBeenCalled());
  }

  /** Each line of the shown mapping holding `text`, found once the build answered. */
  async function linesHolding(text: string) {
    return screen.findAllByText(innermostHolding(text));
  }

  it("names the photographs' positioning error for source h_pos_err", async () => {
    await buildAnswering(BUILT);
    const [radius] = Object.values(BUILT.match_radius);
    expect(radius.source).toBe("h_pos_err");
    expect(
      await linesHolding(
        `radius ${radius.value.toFixed(2)} m from the photographs' positioning error`,
      ),
    ).not.toHaveLength(0);
  });

  it("names the stated radius with its reason for source stated", async () => {
    await buildAnswering(STATED);
    const [radius] = Object.values(STATED.match_radius);
    expect(
      await linesHolding(`radius ${radius.value.toFixed(2)} m stated: ${radius.rationale}`),
    ).not.toHaveLength(0);
  });

  it("sends a stated radius with its reason, and none until one is typed", async () => {
    await buildAnswering(BUILT);
    expect(vi.mocked(resultsApi.buildPlantMapping).mock.calls[0][0].match_radius).toBeNull();

    fireEvent.change(screen.getByLabelText("Match radius (m)"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("Radius source"), {
      target: { value: "the receiver's spec sheet" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Build" }));
    await waitFor(() =>
      expect(resultsApi.buildPlantMapping).toHaveBeenLastCalledWith(
        expect.objectContaining({
          match_radius: { value: 4, rationale: "the receiver's spec sheet" },
        }),
      ),
    );
  });

  it("sends supersede only once the checkbox is checked", async () => {
    await buildAnswering(STATED);

    expect(resultsApi.buildPlantMapping).toHaveBeenCalledWith(
      expect.objectContaining({ supersede: false }),
    );

    fireEvent.click(screen.getByLabelText(/replace a mapping a delivery used/i));
    fireEvent.click(screen.getByRole("button", { name: "Build" }));
    await waitFor(() =>
      expect(resultsApi.buildPlantMapping).toHaveBeenLastCalledWith(
        expect.objectContaining({ supersede: true }),
      ),
    );
  });

  it("shows a citing-events 409 beside the supersede checkbox rather than only a toast", async () => {
    const cited = RESULTS_ANSWERS.mapping_cited;
    vi.spyOn(resultsApi, "buildPlantMapping").mockRejectedValue(await thrown(cited));

    render(<SetupTab />);
    await waitFor(() => expect(resultsApi.listPlantMappings).toHaveBeenCalled());
    fillAndBuild();

    expect(await screen.findByText(cited.body.detail)).toBeInTheDocument();
  });

  it("adopts nothing and shows the panel message when the receipt could not be written", async () => {
    const gap = RESULTS_ANSWERS.mapping_audit_gap;
    vi.spyOn(resultsApi, "buildPlantMapping").mockRejectedValue(await thrown(gap));

    render(<SetupTab />);
    await waitFor(() => expect(resultsApi.listPlantMappings).toHaveBeenCalled());
    fillAndBuild();

    expect(
      await screen.findByText(new RegExp(gap.body.detail.message.slice(0, 40))),
    ).toBeInTheDocument();
  });

  it("renders the mapping-wide unattributed line when the total is nonzero", async () => {
    await buildAnswering(ONE_PLACED);

    expect(
      await screen.findByText(new RegExp(`^${ONE_PLACED.images_unattributed} captures? matched`)),
    ).toBeInTheDocument();
  });

  it("renders no mapping-wide line when the total is zero", async () => {
    await buildAnswering(BUILT);

    expect(BUILT.images_unattributed).toBe(0);
    expect(screen.queryByText(/matched no plant/)).not.toBeInTheDocument();
  });

  it("renders 'no distances' for a date with no recorded mean", async () => {
    await buildAnswering(UNPOSITIONED);

    expect(await linesHolding("unmatched, no distances")).toHaveLength(
      Object.values(UNPOSITIONED.per_date).filter((d) => d.avg_distance_m === null).length,
    );
  });

  /** The summary line of `mapping`'s first date. */
  function firstDateLine(mapping: ServedMapping): string {
    const [{ n_mapped, n_images, n_unattributed }] = Object.values(mapping.per_date);
    return `${n_mapped} of ${n_images} mapped, ${n_unattributed} unmatched`;
  }

  it("loading an already-built mapping by name shows its own summary", async () => {
    vi.spyOn(resultsApi, "listPlantMappings").mockResolvedValue(RESULTS_ANSWERS.mapping_names);
    vi.spyOn(resultsApi, "loadPlantMapping").mockResolvedValue(LOADED);

    render(<SetupTab />);
    await waitFor(() => expect(resultsApi.listPlantMappings).toHaveBeenCalled());
    fireEvent.change(screen.getByLabelText("Mapping"), {
      target: { value: LOADED.name },
    });

    expect(await linesHolding(firstDateLine(LOADED))).not.toHaveLength(0);
    expect(resultsApi.loadPlantMapping).toHaveBeenCalledWith({ name: LOADED.name });
  });

  it("clears the loaded summary once the typed name no longer matches a stored mapping", async () => {
    vi.spyOn(resultsApi, "listPlantMappings").mockResolvedValue(RESULTS_ANSWERS.mapping_names);
    vi.spyOn(resultsApi, "loadPlantMapping").mockResolvedValue(LOADED);

    render(<SetupTab />);
    await waitFor(() => expect(resultsApi.listPlantMappings).toHaveBeenCalled());
    const nameInput = screen.getByLabelText("Mapping");
    fireEvent.change(nameInput, { target: { value: LOADED.name } });

    expect(await linesHolding(firstDateLine(LOADED))).not.toHaveLength(0);

    fireEvent.change(nameInput, { target: { value: `${LOADED.name}-draft` } });

    expect(screen.queryByText(new RegExp(firstDateLine(LOADED)))).not.toBeInTheDocument();
    expect(screen.queryByText(/radius \d/)).not.toBeInTheDocument();
  });
});

describe("SetupTab heading", () => {
  it("names the tab for a screen reader", async () => {
    render(<SetupTab />);
    expect(await screen.findByRole("heading", { level: 1, name: "Setup" })).toBeInTheDocument();
  });
});
