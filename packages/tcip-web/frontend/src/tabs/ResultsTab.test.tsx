import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import { reportedExport, resultsApi } from "@/api/inference";
import type {
  CountExport,
  DeliveryEvents,
  PhenologyAnswer,
  SavedExport,
} from "@/api/types.generated";
import { useStore } from "@/store";
import { ResultsTab } from "@/tabs/ResultsTab";
import { mockDatasetTree } from "@/test/datasetTree";
import {
  ACKNOWLEDGED_COUNT_EXPORT,
  COUNT_EXPORT,
  CURVES_EXPORT,
  DELIVERY_LISTINGS,
  DELIVERY_RECORDS,
  RESULTS_ANSWERS,
} from "@/test/deliveryRecords";
import { fetchAnswers, thrown } from "@/test/refusals";
import { DATASET_TREES, DS_BUCKET, DS_DATE, openTestProject, RAW_SELECTION } from "@/test/store";
import { TRAIT_LISTINGS } from "@/test/traitRecords";

const initialStoreState = useStore.getState();

// The produced phenology measurement of two plants, shown unvalidated.
const MEASURED = RESULTS_ANSWERS.phenology;

// The produced measurement of one plant as the gate answers it once an assessment answers for
// every bucket.
const VALIDATED = RESULTS_ANSWERS.phenology_validated;
const [VALIDATED_ROW] = VALIDATED.milestones;

/** The trait the produced results listing confirms. */
const [{ trait: TRAIT }] = TRAIT_LISTINGS.results.traits;

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  openTestProject(RAW_SELECTION);
  // One trait whose confirmed revision declares milestone fractions, which is what the
  // curve/milestone panels render for.
  vi.spyOn(resultsApi, "traits").mockResolvedValue(TRAIT_LISTINGS.results);
  vi.spyOn(resultsApi, "listPlantMappings").mockResolvedValue(RESULTS_ANSWERS.mapping_names);
  // The delivery-events panel loads with the tab; a test about anything else has no records.
  vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue(RESULTS_ANSWERS.no_deliveries);
});

afterEach(cleanup);

describe("ResultsTab structured predictions-by-date picker", () => {
  it("delivers only the bucket the breeder picks per date, and offers none for a date with none", async () => {
    mockDatasetTree();
    const [first, second] = DATASET_TREES.ds.captures.map((c) => c.name);
    const [, picked] = DATASET_TREES.ds.buckets_by_date[first];
    expect(DATASET_TREES.ds.buckets_by_date[second]).toEqual([]);
    const measurementSpy = vi
      .spyOn(resultsApi, "phenologyMeasurement")
      .mockResolvedValue(VALIDATED);

    render(<ResultsTab />);
    await waitFor(() => expect(api.dataset.tree).toHaveBeenCalledWith(RAW_SELECTION.dataset_root));
    // The trait resolves from the project's registered traits asynchronously; wait for it before
    // computing, or the compute click races the fetch and refuses (no trait resolved yet).
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());

    // No date starts on a bucket: a date with two published waits for the breeder's choice.
    await waitFor(() => expect(screen.getByText(first)).toBeInTheDocument());
    const selects = screen.getAllByTitle(
      /The bucket whose predictions to use for this date|No bucket is published over this date/,
    ) as HTMLSelectElement[];
    expect(selects).toHaveLength(2);
    expect(selects[0].value).toBe("");
    expect(selects[1]).toBeDisabled();

    fireEvent.change(selects[0], { target: { value: picked } });
    fireEvent.change(screen.getByLabelText("Plants"), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(measurementSpy).toHaveBeenCalled());
    // The name is the one the tree response supplied.
    expect(measurementSpy.mock.calls[0][0].buckets).toEqual([picked]);
    expect(measurementSpy.mock.calls[0][0].dataset_root).toBe(RAW_SELECTION.dataset_root);
  });

  it("shows the tree's label_problem beside the date list without blocking it", async () => {
    const { troubled } = DATASET_TREES;
    mockDatasetTree(troubled);
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue(VALIDATED);

    render(<ResultsTab />);

    expect(troubled.label_problem).toEqual(expect.any(String));
    expect(await screen.findByText(String(troubled.label_problem))).toBeInTheDocument();
    expect(screen.getByText(troubled.captures[0].name)).toBeInTheDocument();
  });

  it("dropping a date to '(skip)' excludes it from the computed predictions map", async () => {
    mockDatasetTree();
    const measurementSpy = vi
      .spyOn(resultsApi, "phenologyMeasurement")
      .mockResolvedValue(VALIDATED);

    render(<ResultsTab />);
    // The trait resolves from the project's registered traits asynchronously; wait for it before
    // computing, or the compute click races the fetch and refuses (no trait resolved yet).
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());

    const select = screen.getByTitle("The bucket whose predictions to use for this date");
    fireEvent.change(select, { target: { value: DS_BUCKET } });
    fireEvent.change(select, { target: { value: "" } });
    fireEvent.change(screen.getByLabelText("Plants"), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(measurementSpy).toHaveBeenCalled());
    expect(measurementSpy.mock.calls[0][0].buckets).toEqual([]);
  });
});

describe("ResultsTab unreadable trait visibility", () => {
  it("names an unreadable trait", async () => {
    const listing = TRAIT_LISTINGS.unreadable;
    vi.spyOn(resultsApi, "traits").mockResolvedValue(listing);

    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());

    const [{ trait }] = listing.unreadable;
    expect(await screen.findByText(new RegExp(`will not read: ${trait}`))).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /setup/i })).not.toBeInTheDocument();
  });

  it("renders nothing extra when every trait record reads", async () => {
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());

    expect(screen.queryByText(/will not read/)).not.toBeInTheDocument();
  });

  it("hides phenology while no revision is confirmed", async () => {
    const listing = TRAIT_LISTINGS.unconfirmed;
    vi.spyOn(resultsApi, "traits").mockResolvedValue(listing);

    render(<ResultsTab />);

    expect(
      await screen.findByText(`No revision of ${listing.traits[0].trait} is confirmed.`),
    ).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Phenology" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /setup/i })).not.toBeInTheDocument();
  });
});

describe("ResultsTab evidence gate", () => {
  async function renderAndCompute() {
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    fireEvent.change(screen.getByTitle("The bucket whose predictions to use for this date"), {
      target: { value: DS_BUCKET },
    });
    fireEvent.change(screen.getByLabelText("Plants"), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: "Compute" }));
  }

  it("answers a delivery refusal with the gate's own sentence", async () => {
    mockDatasetTree();
    const refusal = RESULTS_ANSWERS.delivery_refusal;
    vi.spyOn(resultsApi, "phenologyMeasurement").mockRejectedValue(await thrown(refusal));

    await renderAndCompute();

    expect(await screen.findByText(refusal.body.detail.message)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /show unvalidated numbers/i })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /ask the agent/i })).not.toBeInTheDocument();
  });

  it("replaces the disabled export with an acknowledge-and-export flow while unvalidated", async () => {
    useStore.setState({ user: "breeder" });
    mockDatasetTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue(MEASURED);
    const downloadCsv = vi.spyOn(resultsApi, "downloadCsv").mockResolvedValue({
      blob: new Blob([CURVES_EXPORT.file]),
      exported: reportedExport<SavedExport>(new Headers(CURVES_EXPORT.headers)),
    });

    await renderAndCompute();
    const [first] = MEASURED.milestones;
    await waitFor(() => expect(screen.getByText(first.plant_id)).toBeInTheDocument());

    expect(screen.queryByRole("button", { name: /curves csv/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /milestones csv/i })).not.toBeInTheDocument();
    // Each row has to say so too; the banner explaining it scrolls out of view.
    expect(screen.getAllByText("unvalidated")).toHaveLength(MEASURED.milestones.length);
    expect(screen.queryByText("valid")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /acknowledge and export/i }));
    expect(screen.getByRole("button", { name: /curves csv/i })).toBeDisabled();

    fireEvent.change(screen.getByPlaceholderText(/reason for delivering unvalidated/i), {
      target: { value: "calibration is not ready yet" },
    });
    fireEvent.click(screen.getByRole("button", { name: /curves csv/i }));

    await waitFor(() => expect(downloadCsv).toHaveBeenCalled());
    expect(downloadCsv.mock.calls[0][0].user).toBe("breeder");
    expect(downloadCsv.mock.calls[0][0].acknowledgment).toEqual({
      reason: "calibration is not ready yet",
      result_sha256: MEASURED.unvalidated?.digests.curves,
    });
  });

  it("opens both CSV doors once the same rows arrive on validated evidence", async () => {
    mockDatasetTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue(VALIDATED);

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText(VALIDATED_ROW.plant_id)).toBeInTheDocument());

    expect(screen.getByRole("button", { name: /curves csv/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /milestones csv/i })).toBeEnabled();
    expect(screen.getByText("valid")).toBeInTheDocument();
  });

  it("toasts the second-delivery sentence naming the saved path when the CSV export's audit line is lost", async () => {
    mockDatasetTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue(VALIDATED);
    const gap = RESULTS_ANSWERS.export_audit_gap.body.detail;
    vi.spyOn(resultsApi, "downloadCsv").mockRejectedValue(
      await thrown(RESULTS_ANSWERS.export_audit_gap),
    );

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText(VALIDATED_ROW.plant_id)).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /curves csv/i }));

    await waitFor(() => {
      const toast = useStore.getState().toasts.at(-1)?.message ?? "";
      expect(toast).toContain(gap.committed.saved_path);
      expect(toast).toContain("already written");
      expect(toast).toContain("second delivery");
      expect(toast).toContain(gap.message);
    });
  });

  it("shows the row count only after a compute", async () => {
    mockDatasetTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue(VALIDATED);

    render(<ResultsTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    expect(screen.getByRole("heading", { name: "Milestones" })).toBeInTheDocument();
    expect(screen.queryByText(/\d+ rows?$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Not validated:/)).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Plants"), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(screen.getByText(VALIDATED_ROW.plant_id)).toBeInTheDocument());

    expect(screen.getByRole("heading", { name: /Milestones\s*1 row/ })).toBeInTheDocument();
    expect(screen.queryByText(/positive-state fraction is the share/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/acknowledged and exported unvalidated/i)).not.toBeInTheDocument();
  });

  it("clears the rows when a compute fails", async () => {
    mockDatasetTree();
    const measurement = vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue(VALIDATED);

    await renderAndCompute();
    const cell = { name: VALIDATED_ROW.plant_id };
    await waitFor(() => expect(screen.getByRole("cell", cell)).toBeInTheDocument());

    const refusal = RESULTS_ANSWERS.unknown_mapping;
    measurement.mockRejectedValue(await thrown(refusal));
    fireEvent.click(screen.getByRole("button", { name: "Compute" }));

    expect(await screen.findByText(refusal.body.detail)).toBeInTheDocument();
    expect(screen.queryByRole("cell", cell)).not.toBeInTheDocument();
  });

  it("renders the delivery-scoped unattributed count beside the measurement", async () => {
    mockDatasetTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue(VALIDATED);
    const { dates_delivered: dates, images_unattributed: unattributed } = VALIDATED.plant_mapping;
    expect(unattributed).toBeGreaterThan(0);

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText(VALIDATED_ROW.plant_id)).toBeInTheDocument());

    expect(
      await screen.findByText(
        new RegExp(`Delivered dates ${dates.join(", ")}: ${unattributed} attributed to no plant`),
      ),
    ).toBeInTheDocument();
  });
});

describe("ResultsTab onset table validity marker", () => {
  // The produced measurements of one plant each: one missing an image, one never detected, and
  // one past its last observation.
  const [MISSING_IMAGE, UNOBSERVED, PAST_LAST] = RESULTS_ANSWERS.milestones;

  /** Compute `answer`, a measurement of one plant; return its row. */
  async function computeWithOnsetRow(answer: PhenologyAnswer) {
    const [row] = answer.milestones;
    mockDatasetTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue(answer);

    render(<ResultsTab />);
    // The trait resolves from the project's registered traits asynchronously; wait for it before
    // computing, or the compute click races the fetch and refuses (no trait resolved yet).
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("Plants"), { target: { value: row.plant_id } });
    fireEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(screen.getByText(row.plant_id)).toBeInTheDocument());
    return row;
  }

  it("marks a fully-classified, fully-observed plant valid", async () => {
    await computeWithOnsetRow(VALIDATED);
    expect(screen.getByText("valid")).toBeInTheDocument();
    expect(screen.queryByText("incomplete")).not.toBeInTheDocument();
    expect(screen.queryByText("no observations")).not.toBeInTheDocument();
  });

  it("marks a plant with an unclassified or missing-image date incomplete, not silently blank", async () => {
    const row = await computeWithOnsetRow(MISSING_IMAGE);
    expect(row.n_dates_missing_images).toBeGreaterThan(0);
    expect(screen.getByText("incomplete")).toBeInTheDocument();
    expect(screen.queryByText("valid")).not.toBeInTheDocument();
    expect(screen.queryByText("no observations")).not.toBeInTheDocument();
  });

  it("marks a fully-classified, fully-observed plant with zero detections as 'no observations', not 'valid'", async () => {
    await computeWithOnsetRow(UNOBSERVED);
    expect(screen.getByText("no observations")).toBeInTheDocument();
    expect(screen.queryByText("valid")).not.toBeInTheDocument();
    expect(screen.queryByText("incomplete")).not.toBeInTheDocument();
  });

  it("renders a right-censored milestone with its own marker, not the interpolated one", async () => {
    const row = await computeWithOnsetRow(PAST_LAST);
    const crossings = Object.values(row.milestones).filter(
      (crossing) => crossing?.bound === "right_censored",
    );
    expect(crossings).not.toHaveLength(0);
    for (const crossing of crossings) {
      expect(screen.getAllByText(crossing?.date ?? "")).not.toHaveLength(0);
    }
    const markers = screen.getAllByTitle("After last observation");
    expect(markers).toHaveLength(crossings.length);
    for (const marker of markers) expect(marker).toHaveTextContent(">");
    expect(screen.queryByTitle("Interpolated")).not.toBeInTheDocument();
  });
});

describe("ResultsTab meaning refusals", () => {
  it("routes a refusal by its kind to its own panel, never the assessment flow", async () => {
    mockDatasetTree();
    const refusal = RESULTS_ANSWERS.operationalization_refusal;
    vi.spyOn(resultsApi, "phenologyMeasurement").mockRejectedValue(await thrown(refusal));

    render(<ResultsTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("Plants"), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: "Compute" }));

    expect(await screen.findByText(refusal.body.detail.message)).toBeInTheDocument();
    expect(screen.queryByText(RESULTS_ANSWERS.delivery_refusal.body.detail.message)).toBeNull();
    expect(
      screen.queryByRole("button", { name: /show unvalidated numbers/i }),
    ).not.toBeInTheDocument();
  });

  it("renders a structured refusal from the CSV download instead of stringifying it", async () => {
    mockDatasetTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue(VALIDATED);
    const refusal = RESULTS_ANSWERS.operationalization_refusal;
    // exportCsv is left unmocked: the refusal has to survive its own blob path.
    fetchAnswers(refusal.status, refusal.body);

    render(<ResultsTab />);
    await waitFor(() => expect(screen.getByText(DS_DATE)).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("Plants"), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: "Compute" }));
    await waitFor(() => expect(screen.getByRole("button", { name: /curves csv/i })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: /curves csv/i }));

    expect(await screen.findByText(refusal.body.detail.message)).toBeInTheDocument();
    expect(screen.queryByText(/\[object Object\]/)).not.toBeInTheDocument();
    // The assessment flow answers a different refusal family and must not be offered for this one.
    expect(
      screen.queryByRole("button", { name: /ask the agent to assess this/i }),
    ).not.toBeInTheDocument();
  });
});

describe("ResultsTab delivery events (read-only)", () => {
  /** Render the tab over the produced one-record `listing`; return its record's row. */
  async function servedRow(listing: DeliveryEvents): Promise<HTMLElement> {
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue(listing);
    render(<ResultsTab />);
    const [served] = listing.records;
    return screen.findByTestId(`delivery-${served.event.event_id}`);
  }

  it("lists what shipped, with each bucket's gate finding and no confirm/withdraw controls", async () => {
    const record = DELIVERY_RECORDS.validated.event;
    const row = await servedRow(DELIVERY_LISTINGS.validated);

    expect(within(row).getByText(record.trait)).toBeInTheDocument();
    expect(
      within(row).getByText(`revision ${record.trait_revision}, ${record.delivery_kind}`),
    ).toBeInTheDocument();
    expect(within(row).getByText(record.door)).toBeInTheDocument();
    expect(within(row).getByText(record.produced_at)).toBeInTheDocument();
    expect(within(row).getByText(record.output_path)).toBeInTheDocument();
    for (const bucket of record.buckets) {
      expect(
        within(row).getByText(`${bucket.bucket}: validated by assessment ${bucket.assessment_id}`),
      ).toBeInTheDocument();
    }
    expect(within(row).queryByRole("button", { name: /confirm/i })).not.toBeInTheDocument();
    expect(within(row).queryByRole("button", { name: /withdraw/i })).not.toBeInTheDocument();
    expect(within(row).queryByRole("button", { name: /correction/i })).not.toBeInTheDocument();
  });

  it("renders the acknowledging breeder's name and reason, and each bucket's missing assessment", async () => {
    const record = DELIVERY_RECORDS.acknowledged_mapping.event;
    const row = await servedRow(DELIVERY_LISTINGS.acknowledged_mapping);

    expect(within(row).getByText(record.acknowledgment.acknowledged_by)).toBeInTheDocument();
    expect(within(row).getByText(record.acknowledgment.reason)).toBeInTheDocument();
    for (const bucket of record.buckets) {
      expect(
        within(row).getByText(`${bucket.bucket}: not validated (${bucket.reason})`),
      ).toBeInTheDocument();
    }
  });

  it("renders nothing extra when this project has no deliveries yet", async () => {
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.deliveryEvents).toHaveBeenCalled());

    expect(screen.getByText("No deliveries.")).toBeInTheDocument();
  });

  it("renders only the load error when the listing is refused, not the empty state", async () => {
    vi.spyOn(resultsApi, "deliveryEvents").mockRejectedValue(
      await thrown(RESULTS_ANSWERS.unreadable_deliveries),
    );

    render(<ResultsTab />);

    await waitFor(() =>
      expect(screen.getByText(/could not load what this project has shipped/i)).toBeInTheDocument(),
    );
    expect(screen.queryByText("No deliveries.")).not.toBeInTheDocument();
  });

  it("renders the plant mapping's dates_delivered, images_unattributed and plant_attribution", async () => {
    const pm = DELIVERY_RECORDS.acknowledged_mapping.event.plant_mapping;
    const row = await servedRow(DELIVERY_LISTINGS.acknowledged_mapping);

    expect(
      within(row).getByText(
        `Delivered dates ${pm.dates_delivered.join(", ")}: ${pm.images_unattributed} ` +
          `attributed to no plant (${pm.plant_attribution}-level attribution)`,
      ),
    ).toBeInTheDocument();
    expect(within(row).queryByText(/archived as/)).not.toBeInTheDocument();
  });

  it("renders the orthomosaic door's own registry disclosure, not the walked-mapping form", async () => {
    const pm = DELIVERY_RECORDS.registry.event.plant_mapping;
    const row = await servedRow(DELIVERY_LISTINGS.registry);

    expect(
      within(row).getByText(
        `Plant registry ${pm.plant_registry.name}: ${pm.detections_unattributed} ` +
          "detection(s) attributed to no plant on the delivered raster " +
          `(${pm.plant_attribution}-level attribution)`,
      ),
    ).toBeInTheDocument();
    expect(
      within(row).getByText(`Outside the raster: ${pm.plants_outside_raster.join(", ")}`),
    ).toBeInTheDocument();
    expect(within(row).queryByText(/Delivered dates/)).not.toBeInTheDocument();
  });

  it("renders the orthomosaic door's own canopy-segment disclosure, narrowed before the registry form", async () => {
    const record = DELIVERY_RECORDS.canopy.event;
    const pm = record.plant_mapping;
    const row = await servedRow(DELIVERY_LISTINGS.canopy);

    expect(within(row).getByTestId("canopy-disclosure")).toBeInTheDocument();
    expect(
      within(row).getByText(
        `Canopy segments (${pm.plant_registry.name}): ${record.population.length} registry ` +
          `plant(s) delivered, ${pm.segments_without_plant} segment(s) with no plant`,
      ),
    ).toBeInTheDocument();
    expect(
      within(row).getByText(`No segment: ${pm.plants_without_segment.join(", ")}`),
    ).toBeInTheDocument();
    expect(
      within(row).getByText(
        `Within the position error: ${pm.plants_within_position_error.join(", ")}`,
      ),
    ).toBeInTheDocument();
    expect(
      within(row).getByText(`Outside the raster: ${pm.plants_outside_raster.join(", ")}`),
    ).toBeInTheDocument();
    expect(
      within(row).getByText(
        `Ambiguous detection: ${pm.plants_with_ambiguous_detections.join(", ")}`,
      ),
    ).toBeInTheDocument();
    expect(within(row).queryByText(/Plant registry/)).not.toBeInTheDocument();
    expect(within(row).queryByText(/Delivered dates/)).not.toBeInTheDocument();
  });

  it("renders the archived key beside a cited mapping once a rebuild has moved past it", async () => {
    const served = DELIVERY_RECORDS.archived_mapping;
    const row = await servedRow(DELIVERY_LISTINGS.archived_mapping);

    expect(
      within(row).getByText(new RegExp(`archived as ${served.resolved_key}`)),
    ).toBeInTheDocument();
  });
});

describe("ResultsTab count export", () => {
  // A completed download makes and revokes a Blob URL; these stand in for both so no test
  // allocates one.
  beforeEach(() => {
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:mock-url");
    vi.spyOn(URL, "revokeObjectURL").mockReturnValue(undefined);
  });

  // Every field in the count-export panel sits immediately after its own <label>, the DOM
  // relationship this reads.
  function controlFollowing(panel: HTMLElement, labelText: string): HTMLElement {
    const control = within(panel).getByText(labelText).nextElementSibling;
    expect(control).toBeInstanceOf(HTMLElement);
    return control as HTMLElement;
  }

  async function renderCountPanel(): Promise<HTMLElement> {
    mockDatasetTree();
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(api.dataset.tree).toHaveBeenCalled());
    const heading = await screen.findByRole("heading", { name: "Export" });
    const panel = heading.closest(".tcip-panel");
    expect(panel).toBeInstanceOf(HTMLElement);
    // The Trait and Prediction bucket selects only fill in once their own async loads resolve.
    await within(panel as HTMLElement).findByText(TRAIT);
    await within(panel as HTMLElement).findByText(`${DS_DATE} (${DS_BUCKET})`);
    return panel as HTMLElement;
  }

  /** The produced download `exported`: the file the count-export door sent, the export it
   *  reported read as the client reads it. */
  function downloaded(exported: typeof COUNT_EXPORT) {
    return {
      blob: new Blob([exported.file]),
      exported: reportedExport<CountExport>(new Headers(exported.headers)),
    };
  }

  function chooseBucket(panel: HTMLElement) {
    fireEvent.change(controlFollowing(panel, "Predictions"), {
      target: { value: DS_BUCKET },
    });
  }

  it("posts a per_image_count delivery with the selected bucket and trait", async () => {
    const panel = await renderCountPanel();
    const downloadCountCsv = vi
      .spyOn(resultsApi, "downloadCountCsv")
      .mockResolvedValue(downloaded(COUNT_EXPORT));

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: TRAIT } });
    fireEvent.change(controlFollowing(panel, "File name"), { target: { value: "counts.csv" } });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    await waitFor(() => expect(downloadCountCsv).toHaveBeenCalled());
    expect(downloadCountCsv.mock.calls[0][0]).not.toHaveProperty("project_root");
    expect(downloadCountCsv.mock.calls[0][0]).toMatchObject({
      delivery: {
        kind: "per_image_count",
        dataset_root: RAW_SELECTION.dataset_root,
        bucket: DS_BUCKET,
        trait: TRAIT,
      },
      filename: "counts.csv",
    });
  });

  it("posts an orthomosaic_plant_counts delivery with its own fields", async () => {
    const panel = await renderCountPanel();
    const downloadCountCsv = vi
      .spyOn(resultsApi, "downloadCountCsv")
      .mockResolvedValue(downloaded(COUNT_EXPORT));

    fireEvent.change(controlFollowing(panel, "Kind"), {
      target: { value: "orthomosaic_plant_counts" },
    });
    chooseBucket(panel);
    fireEvent.change(within(panel).getByLabelText("Plants"), {
      target: { value: "PLOT-01\nPLOT-02" },
    });
    fireEvent.change(controlFollowing(panel, "Plant registry"), {
      target: { value: "reg" },
    });
    fireEvent.change(controlFollowing(panel, "Phenotype"), {
      target: { value: "stem_count" },
    });
    fireEvent.change(controlFollowing(panel, "File name"), {
      target: { value: "plant_counts.csv" },
    });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    await waitFor(() => expect(downloadCountCsv).toHaveBeenCalled());
    expect(downloadCountCsv.mock.calls[0][0]).toMatchObject({
      delivery: {
        kind: "orthomosaic_plant_counts",
        dataset_root: RAW_SELECTION.dataset_root,
        bucket: DS_BUCKET,
        plant_registry: "reg",
        delivered_phenotype: "stem_count",
        plants: ["PLOT-01", "PLOT-02"],
      },
      filename: "plant_counts.csv",
    });
  });

  it("drops the position error with the canopy subject it belongs to", async () => {
    const panel = await renderCountPanel();
    const downloadCountCsv = vi
      .spyOn(resultsApi, "downloadCountCsv")
      .mockResolvedValue(downloaded(COUNT_EXPORT));

    fireEvent.change(controlFollowing(panel, "Kind"), {
      target: { value: "orthomosaic_plant_counts" },
    });
    chooseBucket(panel);
    fireEvent.change(within(panel).getByLabelText("Plants"), {
      target: { value: "PLOT-01" },
    });
    fireEvent.change(controlFollowing(panel, "Plant registry"), {
      target: { value: "reg" },
    });
    fireEvent.change(controlFollowing(panel, "Phenotype"), {
      target: { value: "stem_count" },
    });
    const canopy = within(panel).getByLabelText("Canopy subject");
    fireEvent.change(canopy, { target: { value: "canopy" } });
    fireEvent.change(within(panel).getByLabelText("Position error (m)"), {
      target: { value: "0.7" },
    });
    fireEvent.change(canopy, { target: { value: "" } });
    fireEvent.change(controlFollowing(panel, "File name"), {
      target: { value: "plant_counts.csv" },
    });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    await waitFor(() => expect(downloadCountCsv).toHaveBeenCalled());
    const delivery = downloadCountCsv.mock.calls[0][0].delivery;
    expect(delivery).toMatchObject({ position_error_m: null, canopy_subject: "" });
  });

  it("shows the second-delivery sentence naming the saved path when the count export's audit line is lost", async () => {
    const panel = await renderCountPanel();
    const gap = RESULTS_ANSWERS.count_export_audit_gap.body.detail;
    vi.spyOn(resultsApi, "downloadCountCsv").mockRejectedValue(
      await thrown(RESULTS_ANSWERS.count_export_audit_gap),
    );

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: TRAIT } });
    fireEvent.change(controlFollowing(panel, "File name"), { target: { value: "counts.csv" } });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    const errorText = await within(panel).findByText((content) =>
      content.startsWith(`The file is already written at ${gap.committed.saved_path},`),
    );
    expect(errorText.textContent).toContain("second delivery");
    expect(errorText.textContent).toContain(gap.message);
  });

  it("decodes a delivery refusal and offers the acknowledgment controls", async () => {
    const panel = await renderCountPanel();
    const refusal = RESULTS_ANSWERS.count_delivery_refusal;
    vi.spyOn(resultsApi, "downloadCountCsv").mockRejectedValue(await thrown(refusal));

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: TRAIT } });
    fireEvent.change(controlFollowing(panel, "File name"), { target: { value: "counts.csv" } });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    expect(await within(panel).findByText(refusal.body.detail.message)).toBeInTheDocument();
    expect(
      within(panel).getByPlaceholderText(/reason for delivering unvalidated/i),
    ).toBeInTheDocument();
    expect(
      within(panel).getByRole("button", { name: /acknowledge and export/i }),
    ).toBeInTheDocument();
  });

  it("downloads and reports the export the door reported", async () => {
    const panel = await renderCountPanel();
    const acknowledged = downloaded(ACKNOWLEDGED_COUNT_EXPORT);
    vi.spyOn(resultsApi, "downloadCountCsv").mockResolvedValue(acknowledged);
    const {
      saved_path: savedTo,
      validated,
      acknowledged_by: acknowledgedBy,
    } = acknowledged.exported;

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: TRAIT } });
    fireEvent.change(controlFollowing(panel, "File name"), { target: { value: "counts.csv" } });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    const report = await within(panel).findByText((content) =>
      content.startsWith(`Saved to ${savedTo}.`),
    );
    expect(validated).toBe(false);
    expect(report.textContent).toContain(" Not validated.");
    expect(report.textContent).toContain(` Acknowledged by ${acknowledgedBy}.`);
  });

  it("disables the Export button while a post is in flight", async () => {
    const panel = await renderCountPanel();
    const download =
      Promise.withResolvers<Awaited<ReturnType<typeof resultsApi.downloadCountCsv>>>();
    vi.spyOn(resultsApi, "downloadCountCsv").mockReturnValue(download.promise);

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: TRAIT } });
    fireEvent.change(controlFollowing(panel, "File name"), { target: { value: "counts.csv" } });
    const button = within(panel).getByRole("button", { name: /^export$/i });
    fireEvent.click(button);

    await waitFor(() => expect(button).toBeDisabled());

    download.resolve(downloaded(COUNT_EXPORT));
    await waitFor(() =>
      expect(within(panel).getByRole("button", { name: /^export$/i })).toBeEnabled(),
    );
  });
});

describe("ResultsTab heading", () => {
  it("renders exactly one top-level heading naming the tab", async () => {
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Results");
  });
});
