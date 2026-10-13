import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { StructuredRefusalError } from "@/api/http";
import { resultsApi, type ServedPlantMapping } from "@/api/inference";
import type { MatchRadius } from "@/api/types.generated";
import { useStore } from "@/store";
import { SetupTab } from "@/tabs/SetupTab";
import { openTestProject } from "@/test/store";
import { TRAIT_LISTINGS } from "@/test/traitRecords";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  openTestProject({ dataset_root: "C:/data" });
  vi.spyOn(resultsApi, "traits").mockResolvedValue({ traits: [], unreadable: [], definitions: {} });
  vi.spyOn(resultsApi, "listPlantMappings").mockResolvedValue({ names: [] });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("SetupTab trait revisions", () => {
  const LISTING = TRAIT_LISTINGS.setup;
  const [FIRST, SECOND] = LISTING.traits[0].revisions;

  it("lists every revision, shows the latest, and names the revision deliveries read", async () => {
    vi.spyOn(resultsApi, "traits").mockResolvedValue(LISTING);

    render(<SetupTab />);
    const row = await screen.findByTestId("trait-stem");

    const revisions = within(row).getByRole("list", { name: "stem revisions" });
    expect(
      within(revisions)
        .getAllByRole("button")
        .map((b) => b.textContent),
    ).toEqual(["Revision 1", "Revision 2"]);
    expect(within(row).getByText("Deliveries read revision 1")).toBeInTheDocument();
    expect(within(row).getByText("Only stems above the graft union.")).toBeInTheDocument();
    expect(within(row).getByText("Not confirmed")).toBeInTheDocument();

    fireEvent.click(within(revisions).getByRole("button", { name: "Revision 1" }));
    expect(within(row).getByText("One stem per detected box.")).toBeInTheDocument();
    expect(within(row).getByText(/Confirmed by user:grower/)).toBeInTheDocument();
  });

  it("confirms the revision shown with that revision's own hash", async () => {
    const traits = vi.spyOn(resultsApi, "traits").mockResolvedValue(LISTING);
    const confirmSpy = vi
      .spyOn(resultsApi, "confirmTraitRevision")
      .mockResolvedValue({ ...SECOND, audit_warning: null });

    useStore.setState({ user: "breeder" });
    render(<SetupTab />);
    const row = await screen.findByTestId("trait-stem");
    fireEvent.click(within(row).getByRole("button", { name: "Confirm revision 2" }));

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    expect(confirmSpy.mock.calls[0][0]).toEqual({
      trait: "stem",
      revision: 2,
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
      .mockResolvedValue({ ...FIRST, audit_warning: null });

    render(<SetupTab />);
    const row = await screen.findByTestId("trait-stem");
    fireEvent.click(within(row).getByRole("button", { name: "Revision 1" }));
    expect(within(row).queryByRole("button", { name: /confirm revision/i })).toBeNull();
    fireEvent.click(within(row).getByRole("button", { name: /withdraw this confirmation/i }));

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    expect(confirmSpy.mock.calls[0][0]).toMatchObject({
      revision: 1,
      entry_sha256: FIRST.entry_sha256,
      confirmed: false,
    });
  });

  it("says the revision was not the one shown when the door answers 409", async () => {
    vi.spyOn(resultsApi, "traits").mockResolvedValue(LISTING);
    vi.spyOn(resultsApi, "confirmTraitRevision").mockRejectedValue(
      new StructuredRefusalError(
        { message: "revision 2 of 'stem' holds another entry", record: { revisions: [] } },
        409,
        "revision 2 of 'stem' holds another entry",
      ),
    );

    render(<SetupTab />);
    const row = await screen.findByTestId("trait-stem");
    fireEvent.click(within(row).getByRole("button", { name: "Confirm revision 2" }));

    expect(await within(row).findByText(/not the one shown/)).toBeInTheDocument();
  });

  it("shows a committed confirmation's audit warning without calling it a failure", async () => {
    vi.spyOn(resultsApi, "traits").mockResolvedValue(LISTING);
    vi.spyOn(resultsApi, "confirmTraitRevision").mockResolvedValue({
      ...SECOND,
      audit_warning: "committed and unrecorded, do not blind-retry",
    });

    render(<SetupTab />);
    const row = await screen.findByTestId("trait-stem");
    fireEvent.click(within(row).getByRole("button", { name: "Confirm revision 2" }));

    expect(
      await screen.findByText(/committed and unrecorded, do not blind-retry/),
    ).toBeInTheDocument();
  });

  it("says nothing delivers under a trait with no confirmed revision", async () => {
    vi.spyOn(resultsApi, "traits").mockResolvedValue(TRAIT_LISTINGS.unconfirmed);

    render(<SetupTab />);
    const row = await screen.findByTestId("trait-subject_a");

    expect(within(row).getByText(/nothing delivers under this trait/)).toBeInTheDocument();
  });

  it("names a trait whose record will not read", async () => {
    vi.spyOn(resultsApi, "traits").mockResolvedValue({
      traits: [],
      unreadable: [{ trait: "leaf_area", reason: "delivers: off-vocab ['leaf_size']" }],
      definitions: {},
    });

    render(<SetupTab />);

    expect(await screen.findByText(/leaf_area: delivers: off-vocab/)).toBeInTheDocument();
  });
});

describe("SetupTab plant-mapping build: match radius", () => {
  function fillAndBuild() {
    fireEvent.change(screen.getByPlaceholderText("valley-2026"), {
      target: { value: "valley-2026" },
    });
    fireEvent.change(screen.getByPlaceholderText("valley-plants"), {
      target: { value: "valley-plants" },
    });
    fireEvent.click(screen.getByRole("button", { name: /build \+ save mapping/i }));
  }

  async function buildWithRadius(radius: MatchRadius) {
    vi.spyOn(resultsApi, "buildPlantMapping").mockResolvedValue({
      name: "valley-2026",
      unreadable: {},
      summary: {
        per_date: {
          "2026-01-01": { n_images: 3, n_mapped: 2, n_unattributed: 1, avg_distance_m: 1.4 },
        },
        totals: { n_dates: 1, n_images: 3, n_mapped: 2, n_unattributed: 1 },
      },
      match_radius: { "2026-01-01": radius },
    });

    render(<SetupTab />);
    await waitFor(() => expect(resultsApi.listPlantMappings).toHaveBeenCalled());
    fillAndBuild();
    await waitFor(() => expect(resultsApi.buildPlantMapping).toHaveBeenCalled());
  }

  it("names the photographs' positioning error for source h_pos_err", async () => {
    await buildWithRadius({ value: 2.5, source: "h_pos_err", rationale: null });
    expect(
      await screen.findByText(
        /matched within 2\.50 m \(the largest positioning error its photographs report\)/,
      ),
    ).toBeInTheDocument();
  });

  it("names the stated radius with its reason for source stated", async () => {
    await buildWithRadius({ value: 4, source: "stated", rationale: "the receiver's spec sheet" });
    expect(
      await screen.findByText(/matched within 4\.00 m \(stated: the receiver's spec sheet\)/),
    ).toBeInTheDocument();
  });

  it("sends a stated radius with its reason, and none until one is typed", async () => {
    await buildWithRadius({ value: 2.5, source: "h_pos_err", rationale: null });
    expect(vi.mocked(resultsApi.buildPlantMapping).mock.calls[0][0]).not.toHaveProperty(
      "match_radius",
    );

    fireEvent.change(screen.getByLabelText("Match radius (m)"), { target: { value: "4" } });
    fireEvent.change(screen.getByLabelText("How the radius is known"), {
      target: { value: "the receiver's spec sheet" },
    });
    fireEvent.click(screen.getByRole("button", { name: /build \+ save mapping/i }));
    await waitFor(() =>
      expect(resultsApi.buildPlantMapping).toHaveBeenLastCalledWith(
        expect.objectContaining({
          match_radius: { value: 4, rationale: "the receiver's spec sheet" },
        }),
      ),
    );
  });

  it("sends supersede only once the checkbox is checked", async () => {
    await buildWithRadius({ value: 3, source: "stated", rationale: "known" });

    expect(resultsApi.buildPlantMapping).toHaveBeenCalledWith(
      expect.objectContaining({ supersede: false }),
    );

    fireEvent.click(screen.getByLabelText(/supersede a mapping a delivery event still cites/i));
    fireEvent.click(screen.getByRole("button", { name: /build \+ save mapping/i }));
    await waitFor(() =>
      expect(resultsApi.buildPlantMapping).toHaveBeenLastCalledWith(
        expect.objectContaining({ supersede: true }),
      ),
    );
  });

  it("shows a citing-events 409 beside the supersede checkbox rather than only a toast", async () => {
    vi.spyOn(resultsApi, "buildPlantMapping").mockRejectedValue(
      new StructuredRefusalError(
        { message: "plant mapping 'valley' is cited by delivery event(s) ['evt-1']" },
        409,
        "plant mapping 'valley' is cited by delivery event(s) ['evt-1']",
      ),
    );

    render(<SetupTab />);
    await waitFor(() => expect(resultsApi.listPlantMappings).toHaveBeenCalled());
    fillAndBuild();

    expect(await screen.findByText(/cited by delivery event\(s\)/)).toBeInTheDocument();
  });

  it("adopts nothing and shows the panel message when the receipt could not be written", async () => {
    const gapMessage =
      "plant_mapping_built completed and its audit entry could not be written: the log refused";
    vi.spyOn(resultsApi, "buildPlantMapping").mockRejectedValue(
      new StructuredRefusalError(
        { error: "audit_entry_not_written", message: gapMessage, committed: null },
        409,
        gapMessage,
      ),
    );

    render(<SetupTab />);
    await waitFor(() => expect(resultsApi.listPlantMappings).toHaveBeenCalled());
    fillAndBuild();

    expect(await screen.findByText(new RegExp(gapMessage.slice(0, 40)))).toBeInTheDocument();
  });

  function buildMappingWith(
    perDate: Record<
      string,
      {
        n_images: number;
        n_mapped: number;
        n_unattributed: number;
        avg_distance_m: number | null;
      }
    >,
    totalsUnattributed: number,
  ) {
    const totalImages = Object.values(perDate).reduce((sum, d) => sum + d.n_images, 0);
    const totalMapped = Object.values(perDate).reduce((sum, d) => sum + d.n_mapped, 0);
    vi.spyOn(resultsApi, "buildPlantMapping").mockResolvedValue({
      name: "valley-2026",
      unreadable: {},
      summary: {
        per_date: perDate,
        totals: {
          n_dates: Object.keys(perDate).length,
          n_images: totalImages,
          n_mapped: totalMapped,
          n_unattributed: totalsUnattributed,
        },
      },
      match_radius: {},
    });
  }

  async function buildFromInputs() {
    render(<SetupTab />);
    await waitFor(() => expect(resultsApi.listPlantMappings).toHaveBeenCalled());
    fillAndBuild();
  }

  it("renders the mapping-wide unattributed line when the total is nonzero", async () => {
    buildMappingWith(
      { "2026-01-01": { n_images: 3, n_mapped: 2, n_unattributed: 1, avg_distance_m: 1.4 } },
      1,
    );
    await buildFromInputs();

    expect(
      await screen.findByText(/1 captures across this mapping's dates are attributed to no plant/),
    ).toBeInTheDocument();
  });

  it("renders no mapping-wide line when the total is zero", async () => {
    buildMappingWith(
      { "2026-01-01": { n_images: 2, n_mapped: 2, n_unattributed: 0, avg_distance_m: 1.4 } },
      0,
    );
    await buildFromInputs();

    await waitFor(() => expect(resultsApi.buildPlantMapping).toHaveBeenCalled());
    expect(
      screen.queryByText(/attributed to no plant \(no readable position/),
    ).not.toBeInTheDocument();
  });

  it("renders 'no distances' for a date with no recorded mean", async () => {
    buildMappingWith(
      { "2026-01-01": { n_images: 2, n_mapped: 0, n_unattributed: 2, avg_distance_m: null } },
      2,
    );
    await buildFromInputs();

    expect(await screen.findByText(/avg no distances/)).toBeInTheDocument();
  });

  const LOADED: ServedPlantMapping = {
    name: "valley-2026",
    unreadable: {},
    summary: {
      per_date: {
        "2026-01-01": { n_images: 4, n_mapped: 4, n_unattributed: 0, avg_distance_m: 0.9 },
      },
      totals: { n_dates: 1, n_images: 4, n_mapped: 4, n_unattributed: 0 },
    },
    match_radius: { "2026-01-01": { value: 2, source: "h_pos_err", rationale: null } },
  };

  it("loading an already-built mapping by name shows its own summary", async () => {
    vi.spyOn(resultsApi, "listPlantMappings").mockResolvedValue({ names: ["valley-2026"] });
    vi.spyOn(resultsApi, "loadPlantMapping").mockResolvedValue(LOADED);

    render(<SetupTab />);
    await waitFor(() => expect(resultsApi.listPlantMappings).toHaveBeenCalled());
    fireEvent.change(screen.getByPlaceholderText("valley-2026"), {
      target: { value: "valley-2026" },
    });

    expect(await screen.findByText(/mapped 4 of 4, 0 attributed to no plant/)).toBeInTheDocument();
    expect(resultsApi.loadPlantMapping).toHaveBeenCalledWith("valley-2026");
  });

  it("clears the loaded summary once the typed name no longer matches a stored mapping", async () => {
    vi.spyOn(resultsApi, "listPlantMappings").mockResolvedValue({ names: ["valley-2026"] });
    vi.spyOn(resultsApi, "loadPlantMapping").mockResolvedValue(LOADED);

    render(<SetupTab />);
    await waitFor(() => expect(resultsApi.listPlantMappings).toHaveBeenCalled());
    const nameInput = screen.getByPlaceholderText("valley-2026");
    fireEvent.change(nameInput, { target: { value: "valley-2026" } });

    expect(await screen.findByText(/mapped 4 of 4, 0 attributed to no plant/)).toBeInTheDocument();

    fireEvent.change(nameInput, { target: { value: "valley-2026-draft" } });

    expect(screen.queryByText(/mapped 4 of 4, 0 attributed to no plant/)).not.toBeInTheDocument();
    expect(screen.queryByText(/matched within \d/)).not.toBeInTheDocument();
  });
});

describe("SetupTab heading", () => {
  it("names the tab for a screen reader", async () => {
    render(<SetupTab />);
    expect(await screen.findByRole("heading", { level: 1, name: "Setup" })).toBeInTheDocument();
  });
});
