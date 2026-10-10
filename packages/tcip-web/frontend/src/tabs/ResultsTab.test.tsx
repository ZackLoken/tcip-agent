import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { api } from "@/api/client";
import { StructuredRefusalError } from "@/api/http";
import {
  isCanopySegmentDisclosure,
  isPlantMappingDisclosure,
  isPlantRegistryDisclosure,
  resultsApi,
  type DeliveryEventRecord,
} from "@/api/inference";
import { useStore } from "@/store";
import { ResultsTab } from "@/tabs/ResultsTab";
import { mockDatasetTree } from "@/test/datasetTree";
import { DELIVERY_RECORDS } from "@/test/deliveryRecords";
import { openTestProject } from "@/test/store";
import { TRAIT_LISTINGS } from "@/test/traitRecords";

const initialStoreState = useStore.getState();

// The measurement door returns the delivery gate's finding beside its rows, so a mock that omits
// it would be describing a response the server cannot produce.
const VALIDATED = {
  validated: true,
  unvalidated_reason: null,
  trait: "subject_a",
  trait_revision: 1,
  trait_revision_sha256: "b".repeat(64),
  positive_class_assessed: true,
  captures_unverified: [],
  plant_csvs_unverified: [],
  dates_delivered: [],
  images_unattributed: 0,
  result_sha256: null,
  require_all_dates_complete: true,
};

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  openTestProject({ dataset_root: "C:/data" });
  // One trait whose confirmed revision declares milestone fractions, which is what the
  // curve/milestone panels render for.
  vi.spyOn(resultsApi, "traits").mockResolvedValue(TRAIT_LISTINGS.results);
  vi.spyOn(resultsApi, "listPlantMappings").mockResolvedValue({ names: ["valley"] });
  // The delivery-events panel loads with the tab; a test about anything else has no records.
  vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [] });
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe("ResultsTab structured predictions-by-date picker", () => {
  it("delivers only the bucket the breeder picks per date, and offers none for a date with none", async () => {
    mockDatasetTree({
      dates_with_images: ["2026-01-01", "2026-01-08"],
      buckets_by_date: {
        "2026-01-01": ["baseline/2026-01-01", "v2/2026-01-01"],
        "2026-01-08": [],
      },
    });
    const measurementSpy = vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [], n_plants: 0 },
      milestones: { rows: [], columns: [] },
      ...VALIDATED,
    });

    render(<ResultsTab />);
    await waitFor(() => expect(api.dataset.tree).toHaveBeenCalledWith("C:/data"));
    // The trait resolves from the project's registered traits asynchronously; wait for it before
    // computing, or the compute click races the fetch and refuses (no trait resolved yet).
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());

    // No date starts on a bucket: two published over one date are never chosen between silently.
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    const selects = screen.getAllByTitle(
      /The bucket whose predictions to use for this date|No bucket is published over this date/,
    ) as HTMLSelectElement[];
    expect(selects).toHaveLength(2);
    expect(selects[0].value).toBe("");
    expect(selects[1]).toBeDisabled();

    fireEvent.change(selects[0], { target: { value: "v2/2026-01-01" } });
    fireEvent.change(screen.getByLabelText(/plants to measure/i), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
    await waitFor(() => expect(measurementSpy).toHaveBeenCalled());
    // The name is the one the tree response supplied, never one the tab assembled.
    expect(measurementSpy.mock.calls[0][0].buckets).toEqual(["v2/2026-01-01"]);
    expect(measurementSpy.mock.calls[0][0].dataset_root).toBe("C:/data");
  });

  it("shows the tree's label_problem beside the date list without blocking it", async () => {
    mockDatasetTree({
      label_problem: "label_documents['2026-01-08', 'IMG_0000'] under C:/data: is a list",
    });
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [], n_plants: 0 },
      milestones: { rows: [], columns: [] },
      ...VALIDATED,
    });

    render(<ResultsTab />);

    expect(
      await screen.findByText("label_documents['2026-01-08', 'IMG_0000'] under C:/data: is a list"),
    ).toBeInTheDocument();
    expect(screen.getByText("2026-01-01")).toBeInTheDocument();
  });

  it("dropping a date to '(skip)' excludes it from the computed predictions map", async () => {
    mockDatasetTree();
    const measurementSpy = vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [], n_plants: 0 },
      milestones: { rows: [], columns: [] },
      ...VALIDATED,
    });

    render(<ResultsTab />);
    // The trait resolves from the project's registered traits asynchronously; wait for it before
    // computing, or the compute click races the fetch and refuses (no trait resolved yet).
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());

    const select = screen.getByTitle("The bucket whose predictions to use for this date");
    fireEvent.change(select, { target: { value: "baseline/2026-01-01" } });
    fireEvent.change(select, { target: { value: "" } });
    fireEvent.change(screen.getByLabelText(/plants to measure/i), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
    await waitFor(() => expect(measurementSpy).toHaveBeenCalled());
    expect(measurementSpy.mock.calls[0][0].buckets).toEqual([]);
  });
});

describe("ResultsTab unreadable trait visibility", () => {
  it("names an unreadable trait and sends the breeder to the Setup tab", async () => {
    vi.spyOn(resultsApi, "traits").mockResolvedValue({
      ...TRAIT_LISTINGS.results,
      unreadable: [{ trait: "leaf_area", reason: "delivers: off-vocab ['leaf_size']" }],
    });

    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());

    expect(await screen.findByText(/will not read: leaf_area/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /open the setup tab/i }));
    expect(useStore.getState().gui.active_tab).toBe("setup");
  });

  it("renders nothing extra when every trait record reads", async () => {
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());

    expect(screen.queryByText(/will not read/)).not.toBeInTheDocument();
  });

  it("sends the breeder to the Setup tab when no revision of the trait is confirmed", async () => {
    vi.spyOn(resultsApi, "traits").mockResolvedValue(TRAIT_LISTINGS.unconfirmed);

    render(<ResultsTab />);

    expect(await screen.findByText(/No revision of subject_a is confirmed/)).toBeInTheDocument();
    expect(screen.queryByText(/Per-plant phenology curves/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /open the setup tab/i }));
    expect(useStore.getState().gui.active_tab).toBe("setup");
  });
});

describe("ResultsTab evidence gate", () => {
  const CURVE_ROW = {
    plant_id: "P1",
    accession: null,
    date: "2026-01-01",
    n_images: 1,
    n_total: 10,
    n_positive: 3,
    n_unclassified: 0,
    n_missing: 0,
    ratio: 0.3,
  };
  const ONSET_ROW = {
    plant_id: "P1",
    accession: null,
    n_dates: 2,
    n_dates_unclassified: 0,
    n_dates_missing_images: 0,
    n_observed_dates: 2,
    complete: true,
    stage_50per_date: "2026-02-01",
  };
  const STAGE_COLUMNS = [{ date: "stage_50per_date", bound: "stage_50per_date_bound" }];
  const UNVALIDATED = {
    validated: false,
    unvalidated_reason:
      "no assessment answers for bucket 'baseline/2026-01-01': assess best against a " +
      "held-out reference selection and publish its predictions under that assessment.",
    trait: "subject_a",
    trait_revision: 1,
    trait_revision_sha256: "b".repeat(64),
    positive_class_assessed: true,
    captures_unverified: [],
    plant_csvs_unverified: [],
    dates_delivered: [],
    images_unattributed: 0,
    result_sha256: { curves: "c".repeat(64), milestones: "d".repeat(64) },
    require_all_dates_complete: true,
  };

  async function renderAndCompute() {
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    fireEvent.change(screen.getByTitle("The bucket whose predictions to use for this date"), {
      target: { value: "baseline/2026-01-01" },
    });
    fireEvent.change(screen.getByLabelText(/plants to measure/i), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
  }

  it("hands a delivery refusal to the assessment flow, not the raw error line", async () => {
    mockDatasetTree({ subjects: [] });
    const message = UNVALIDATED.unvalidated_reason;
    vi.spyOn(resultsApi, "phenologyMeasurement").mockRejectedValue(
      new StructuredRefusalError({ kind: "delivery", message }, 400, message),
    );

    await renderAndCompute();

    expect(
      await screen.findByRole("button", { name: /ask the agent to assess this/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /show unvalidated numbers/i })).toBeInTheDocument();
  });

  it("replaces the disabled export with an acknowledge-and-export flow while unvalidated", async () => {
    useStore.setState({ user: "breeder" });
    mockDatasetTree({ subjects: [] });
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [CURVE_ROW], n_plants: 1 },
      milestones: { rows: [ONSET_ROW], columns: STAGE_COLUMNS },
      ...UNVALIDATED,
    });
    const downloadCsv = vi.spyOn(resultsApi, "downloadCsv").mockResolvedValue(new Blob(["x"]));

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());

    expect(screen.queryByRole("button", { name: /curves csv/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /milestones csv/i })).not.toBeInTheDocument();
    // The row itself has to say so too; the banner explaining it scrolls out of view.
    expect(screen.getByText("unvalidated")).toBeInTheDocument();
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
      result_sha256: "c".repeat(64),
    });
  });

  it("opens both CSV doors once the same rows arrive on validated evidence", async () => {
    mockDatasetTree({ subjects: [] });
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [CURVE_ROW], n_plants: 1 },
      milestones: { rows: [ONSET_ROW], columns: STAGE_COLUMNS },
      ...VALIDATED,
    });

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());

    expect(screen.getByRole("button", { name: /curves csv/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /milestones csv/i })).toBeEnabled();
    expect(screen.getByText("valid")).toBeInTheDocument();
  });

  it("toasts the second-delivery sentence naming the saved path when the CSV export's audit line is lost", async () => {
    mockDatasetTree({ subjects: [] });
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [CURVE_ROW], n_plants: 1 },
      milestones: { rows: [ONSET_ROW], columns: STAGE_COLUMNS },
      ...VALIDATED,
    });
    const message = "results.export_csv completed and its audit entry could not be written";
    vi.spyOn(resultsApi, "downloadCsv").mockRejectedValue(
      new StructuredRefusalError(
        {
          error: "audit_entry_not_written",
          message,
          committed: { saved_path: "C:/proj/results_export/x.csv" },
        },
        409,
        message,
      ),
    );

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /curves csv/i }));

    await waitFor(() => {
      const toast = useStore.getState().toasts.at(-1)?.message ?? "";
      expect(toast).toContain("C:/proj/results_export/x.csv");
      expect(toast).toContain("already written");
      expect(toast).toContain("second delivery");
      expect(toast).toContain(message);
    });
  });

  it("states which delivery kinds the export controls actually cover", async () => {
    mockDatasetTree({ subjects: [] });
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [CURVE_ROW], n_plants: 1 },
      milestones: { rows: [ONSET_ROW], columns: STAGE_COLUMNS },
      ...VALIDATED,
    });

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());

    expect(
      screen.getByText(/a phenology milestone delivery here, and a per-image or per-plant/i),
    ).toBeInTheDocument();
  });

  it("renders the delivery-scoped unattributed count beside the measurement", async () => {
    mockDatasetTree({ subjects: [] });
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [CURVE_ROW], n_plants: 1 },
      milestones: { rows: [ONSET_ROW], columns: STAGE_COLUMNS },
      ...VALIDATED,
      dates_delivered: ["2026-01-01"],
      images_unattributed: 3,
    });

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());

    expect(
      await screen.findByText(/Delivered dates 2026-01-01: 3 attributed to no plant/),
    ).toBeInTheDocument();
  });
});

describe("ResultsTab onset table validity marker", () => {
  async function computeWithOnsetRows(
    rows: Awaited<ReturnType<typeof resultsApi.phenologyMeasurement>>["milestones"]["rows"],
  ) {
    mockDatasetTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: {
        rows: [
          {
            plant_id: "P1",
            accession: null,
            date: "2026-01-01",
            n_images: 1,
            n_total: 10,
            n_positive: 3,
            n_unclassified: 0,
            n_missing: 0,
            ratio: 0.3,
          },
        ],
        n_plants: 1,
      },
      milestones: {
        rows,
        columns: Object.keys(rows[0] ?? {})
          .filter((key) => key.endsWith("_date"))
          .map((date) => ({ date, bound: `${date}_bound` })),
      },
      ...VALIDATED,
    });

    render(<ResultsTab />);
    // The trait resolves from the project's registered traits asynchronously; wait for it before
    // computing, or the compute click races the fetch and refuses (no trait resolved yet).
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText(/plants to measure/i), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());
  }

  it("marks a fully-classified, fully-observed plant valid", async () => {
    await computeWithOnsetRows([
      {
        plant_id: "P1",
        accession: null,
        n_dates: 1,
        n_dates_unclassified: 0,
        n_dates_missing_images: 0,
        n_observed_dates: 1,
        complete: true,
        subject_a_50per_date: "2026-02-01",
      },
    ]);
    expect(screen.getByText("valid")).toBeInTheDocument();
    expect(screen.queryByText("incomplete")).not.toBeInTheDocument();
    expect(screen.queryByText("no observations")).not.toBeInTheDocument();
  });

  it("marks a plant with an unclassified or missing-image date incomplete, not silently blank", async () => {
    await computeWithOnsetRows([
      {
        plant_id: "P1",
        accession: null,
        n_dates: 1,
        n_dates_unclassified: 1,
        n_dates_missing_images: 0,
        n_observed_dates: 0,
        complete: false,
        subject_a_50per_date: null,
      },
    ]);
    expect(screen.getByText("incomplete")).toBeInTheDocument();
    expect(screen.queryByText("valid")).not.toBeInTheDocument();
    expect(screen.queryByText("no observations")).not.toBeInTheDocument();
  });

  it("marks a fully-classified, fully-observed plant with zero detections as 'no observations', not 'valid'", async () => {
    await computeWithOnsetRows([
      {
        plant_id: "P1",
        accession: null,
        n_dates: 2,
        n_dates_unclassified: 0,
        n_dates_missing_images: 0,
        n_observed_dates: 0,
        complete: true,
        subject_a_50per_date: null,
      },
    ]);
    expect(screen.getByText("no observations")).toBeInTheDocument();
    expect(screen.queryByText("valid")).not.toBeInTheDocument();
    expect(screen.queryByText("incomplete")).not.toBeInTheDocument();
  });

  it("renders a right-censored milestone with its own marker, not the interpolated one", async () => {
    await computeWithOnsetRows([
      {
        plant_id: "P1",
        accession: null,
        n_dates: 1,
        n_dates_unclassified: 0,
        n_dates_missing_images: 0,
        n_observed_dates: 1,
        complete: true,
        subject_a_95per_date: "2026-03-12",
        subject_a_95per_date_bound: "right_censored",
      },
    ]);
    expect(screen.getByText("2026-03-12")).toBeInTheDocument();
    const marker = screen.getByTitle(
      "Right-censored: the last observation still hadn't met this target, so the true date, if any, is after this one.",
    );
    expect(marker).toHaveTextContent(">");
    expect(screen.queryByTitle("Interpolated between two observed dates.")).not.toBeInTheDocument();
  });
});

describe("ResultsTab meaning refusals", () => {
  it("routes a refusal by its kind even when its text reads like the calibration one", async () => {
    mockDatasetTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockRejectedValue(
      new StructuredRefusalError(
        {
          kind: "operationalization",
          message: "an unvalidated number is not what this refusal is about",
        },
        400,
        "an unvalidated number is not what this refusal is about",
      ),
    );

    render(<ResultsTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText(/plants to measure/i), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));

    expect(
      await screen.findByText(/an unvalidated number is not what this refusal is about/),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /open the setup tab/i })).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /ask the agent to assess this/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /show unvalidated numbers/i }),
    ).not.toBeInTheDocument();
  });

  it("renders a structured refusal from the CSV download instead of stringifying it", async () => {
    mockDatasetTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: {
        rows: [
          {
            plant_id: "P1",
            accession: null,
            date: "2026-01-01",
            n_images: 1,
            n_total: 10,
            n_positive: 3,
            n_unclassified: 0,
            n_missing: 0,
            ratio: 0.3,
          },
        ],
        n_plants: 1,
      },
      milestones: { rows: [], columns: [] },
      ...VALIDATED,
    });
    // exportCsv is left unmocked: the refusal has to survive its own blob path, not a stub.
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: false,
        status: 400,
        statusText: "Bad Request",
        json: async () => ({
          detail: {
            kind: "operationalization",
            message: "stated but not confirmed by the breeder",
          },
        }),
      } as Response),
    );

    render(<ResultsTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText(/plants to measure/i), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
    await waitFor(() => expect(screen.getByRole("button", { name: /curves csv/i })).toBeEnabled());
    fireEvent.click(screen.getByRole("button", { name: /curves csv/i }));

    expect(await screen.findByText(/stated but not confirmed by the breeder/)).toBeInTheDocument();
    expect(screen.queryByText(/\[object Object\]/)).not.toBeInTheDocument();
    // The assessment flow answers a different refusal family and must not be offered for this one.
    expect(
      screen.queryByRole("button", { name: /ask the agent to assess this/i }),
    ).not.toBeInTheDocument();
    vi.unstubAllGlobals();
  });
});

describe("ResultsTab delivery events (read-only)", () => {
  async function servedRow(record: DeliveryEventRecord): Promise<HTMLElement> {
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [record] });
    render(<ResultsTab />);
    return screen.findByTestId(`delivery-${record.event_id}`);
  }

  it("lists what shipped, with each bucket's gate finding and no confirm/withdraw controls", async () => {
    const record = DELIVERY_RECORDS.validated;
    const row = await servedRow(record);

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
    const record = DELIVERY_RECORDS.acknowledged_mapping;
    const row = await servedRow(record);

    if (!record.acknowledgment) throw new Error("the fixture delivery was not acknowledged");
    expect(within(row).getByText(record.acknowledgment.acknowledged_by)).toBeInTheDocument();
    expect(within(row).getByText(record.acknowledgment.reason)).toBeInTheDocument();
    for (const bucket of record.buckets) {
      expect(
        within(row).getByText(`${bucket.bucket}: not validated (${bucket.reason})`),
      ).toBeInTheDocument();
    }
  });

  it("renders nothing extra when this project has no deliveries yet", async () => {
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [] });

    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.deliveryEvents).toHaveBeenCalled());

    expect(screen.getByText(/nothing has shipped from this project yet/i)).toBeInTheDocument();
  });

  it("renders only the load error when the listing is refused, not the empty state", async () => {
    vi.spyOn(resultsApi, "deliveryEvents").mockRejectedValue(
      new Error(
        "delivery event 'old-shaped' does not validate against the current delivery_events shape",
      ),
    );

    render(<ResultsTab />);

    await waitFor(() =>
      expect(screen.getByText(/could not load what this project has shipped/i)).toBeInTheDocument(),
    );
    expect(
      screen.queryByText(/nothing has shipped from this project yet/i),
    ).not.toBeInTheDocument();
  });

  it("renders the plant mapping's dates_delivered, images_unattributed and plant_attribution", async () => {
    const record = DELIVERY_RECORDS.acknowledged_mapping;
    const pm = record.plant_mapping;
    if (!pm || !isPlantMappingDisclosure(pm)) throw new Error("the fixture cites no mapping");
    const row = await servedRow(record);

    expect(
      within(row).getByText(
        `Delivered dates ${pm.dates_delivered.join(", ")}: ${pm.images_unattributed} ` +
          `attributed to no plant (${pm.plant_attribution}-level attribution)`,
      ),
    ).toBeInTheDocument();
    expect(within(row).queryByText(/archived as/)).not.toBeInTheDocument();
  });

  it("renders the orthomosaic door's own registry disclosure, not the walked-mapping form", async () => {
    const record = DELIVERY_RECORDS.registry;
    const pm = record.plant_mapping;
    if (!pm || !isPlantRegistryDisclosure(pm)) throw new Error("the fixture names no registry");
    const row = await servedRow(record);

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
    const record = DELIVERY_RECORDS.canopy;
    const pm = record.plant_mapping;
    if (!pm || !isCanopySegmentDisclosure(pm)) throw new Error("the fixture names no segments");
    const row = await servedRow(record);

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
    const record = DELIVERY_RECORDS.archived_mapping;
    const row = await servedRow(record);

    expect(
      within(row).getByText(new RegExp(`archived as ${record.plant_mapping_resolved_key}`)),
    ).toBeInTheDocument();
  });
});

describe("ResultsTab count export", () => {
  // jsdom carries no Blob-URL support at all; every export path that reaches a completed
  // download needs these stubbed, the same way a real browser's URL object provides them.
  beforeEach(() => {
    Object.defineProperty(URL, "createObjectURL", {
      value: vi.fn(() => "blob:mock-url"),
      writable: true,
      configurable: true,
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      value: vi.fn(),
      writable: true,
      configurable: true,
    });
  });

  // Every field in the count-export panel sits immediately after its own <label>, the DOM
  // relationship this reads rather than a positional guess at render order.
  function controlFollowing(panel: HTMLElement, labelText: string): HTMLElement {
    const label = within(panel).getByText(labelText);
    const control = label.nextElementSibling;
    if (!control) throw new Error(`no control follows the ${labelText} label`);
    return control as HTMLElement;
  }

  async function renderCountPanel(): Promise<HTMLElement> {
    mockDatasetTree({ subjects: [] });
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(api.dataset.tree).toHaveBeenCalled());
    const heading = await screen.findByText("Count export");
    const panel = heading.closest(".tcip-panel");
    if (!panel) throw new Error("Count export panel not found");
    // The Trait and Prediction bucket selects only fill in once their own async loads resolve.
    await within(panel as HTMLElement).findByText("subject_a");
    await within(panel as HTMLElement).findByText("2026-01-01 (baseline/2026-01-01)");
    return panel as HTMLElement;
  }

  function chooseBucket(panel: HTMLElement) {
    fireEvent.change(controlFollowing(panel, "Prediction bucket"), {
      target: { value: "baseline/2026-01-01" },
    });
  }

  it("posts a per_image_count delivery with the selected bucket and trait", async () => {
    const panel = await renderCountPanel();
    const downloadCountCsv = vi.spyOn(resultsApi, "downloadCountCsv").mockResolvedValue({
      blob: new Blob(["x"]),
      headers: {
        savedTo: "C:/proj/results_export/counts.csv",
        validated: true,
        acknowledgedBy: "",
      },
    });

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: "subject_a" } });
    fireEvent.change(controlFollowing(panel, "Filename"), { target: { value: "counts.csv" } });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    await waitFor(() => expect(downloadCountCsv).toHaveBeenCalled());
    expect(downloadCountCsv.mock.calls[0][0]).not.toHaveProperty("project_root");
    expect(downloadCountCsv.mock.calls[0][0]).toMatchObject({
      delivery: {
        kind: "per_image_count",
        dataset_root: "C:/data",
        bucket: "baseline/2026-01-01",
        trait: "subject_a",
      },
      filename: "counts.csv",
    });
  });

  it("posts an orthomosaic_plant_counts delivery with its own fields", async () => {
    const panel = await renderCountPanel();
    const downloadCountCsv = vi.spyOn(resultsApi, "downloadCountCsv").mockResolvedValue({
      blob: new Blob(["x"]),
      headers: {
        savedTo: "C:/proj/results_export/plant_counts.csv",
        validated: true,
        acknowledgedBy: "",
      },
    });

    fireEvent.change(controlFollowing(panel, "Kind"), {
      target: { value: "orthomosaic_plant_counts" },
    });
    chooseBucket(panel);
    fireEvent.change(within(panel).getByLabelText(/plants to deliver/i), {
      target: { value: "PLOT-01\nPLOT-02" },
    });
    fireEvent.change(controlFollowing(panel, "Plant registry (registered by name)"), {
      target: { value: "reg" },
    });
    fireEvent.change(controlFollowing(panel, "Delivered phenotype"), {
      target: { value: "stem_count" },
    });
    fireEvent.change(controlFollowing(panel, "Filename"), {
      target: { value: "plant_counts.csv" },
    });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    await waitFor(() => expect(downloadCountCsv).toHaveBeenCalled());
    expect(downloadCountCsv.mock.calls[0][0]).toMatchObject({
      delivery: {
        kind: "orthomosaic_plant_counts",
        dataset_root: "C:/data",
        bucket: "baseline/2026-01-01",
        plant_registry: "reg",
        delivered_phenotype: "stem_count",
        plants: ["PLOT-01", "PLOT-02"],
      },
      filename: "plant_counts.csv",
    });
  });

  it("drops the position error with the canopy subject it belongs to", async () => {
    const panel = await renderCountPanel();
    const downloadCountCsv = vi.spyOn(resultsApi, "downloadCountCsv").mockResolvedValue({
      blob: new Blob(["x"]),
      headers: {
        savedTo: "C:/proj/results_export/plant_counts.csv",
        validated: true,
        acknowledgedBy: "",
      },
    });

    fireEvent.change(controlFollowing(panel, "Kind"), {
      target: { value: "orthomosaic_plant_counts" },
    });
    chooseBucket(panel);
    fireEvent.change(within(panel).getByLabelText(/plants to deliver/i), {
      target: { value: "PLOT-01" },
    });
    fireEvent.change(controlFollowing(panel, "Plant registry (registered by name)"), {
      target: { value: "reg" },
    });
    fireEvent.change(controlFollowing(panel, "Delivered phenotype"), {
      target: { value: "stem_count" },
    });
    const canopy = within(panel).getByPlaceholderText("canopy_subject (optional)");
    fireEvent.change(canopy, { target: { value: "canopy" } });
    fireEvent.change(within(panel).getByLabelText("position_error_m"), {
      target: { value: "0.7" },
    });
    fireEvent.change(canopy, { target: { value: "" } });
    fireEvent.change(controlFollowing(panel, "Filename"), {
      target: { value: "plant_counts.csv" },
    });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    await waitFor(() => expect(downloadCountCsv).toHaveBeenCalled());
    const delivery = downloadCountCsv.mock.calls[0][0].delivery;
    expect(delivery).not.toHaveProperty("position_error_m");
    expect(delivery).not.toHaveProperty("canopy_subject");
  });

  it("shows the second-delivery sentence naming the saved path when the count export's audit line is lost", async () => {
    const panel = await renderCountPanel();
    const message = "results.export_count_csv completed and its audit entry could not be written";
    vi.spyOn(resultsApi, "downloadCountCsv").mockRejectedValue(
      new StructuredRefusalError(
        {
          error: "audit_entry_not_written",
          message,
          committed: {
            saved_path: "C:/proj/results_export/counts.csv",
          },
        },
        409,
        message,
      ),
    );

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: "subject_a" } });
    fireEvent.change(controlFollowing(panel, "Filename"), { target: { value: "counts.csv" } });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    const errorText = await within(panel).findByText(/C:\/proj\/results_export\/counts\.csv/);
    expect(errorText.textContent).toContain("already written");
    expect(errorText.textContent).toContain("second delivery");
    expect(errorText.textContent).toContain(message);
  });

  it("decodes a delivery refusal and offers the acknowledgment controls", async () => {
    const panel = await renderCountPanel();
    const message =
      "no assessment answers for bucket 'baseline/2026-01-01': assess best against a " +
      "held-out reference selection and publish its predictions under that assessment.";
    vi.spyOn(resultsApi, "downloadCountCsv").mockRejectedValue(
      new StructuredRefusalError(
        { kind: "delivery", message, result_sha256: "c".repeat(64) },
        400,
        message,
      ),
    );

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: "subject_a" } });
    fireEvent.change(controlFollowing(panel, "Filename"), { target: { value: "counts.csv" } });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    expect(await within(panel).findByText(message)).toBeInTheDocument();
    expect(
      within(panel).getByPlaceholderText(/reason for delivering unvalidated/i),
    ).toBeInTheDocument();
    expect(
      within(panel).getByRole("button", { name: /acknowledge and export/i }),
    ).toBeInTheDocument();
  });

  it("downloads with the headers and reports from them", async () => {
    const panel = await renderCountPanel();
    vi.spyOn(resultsApi, "downloadCountCsv").mockResolvedValue({
      blob: new Blob(["x"]),
      headers: {
        savedTo: "C:/proj/results_export/counts.csv",
        validated: false,
        acknowledgedBy: "user:breeder",
      },
    });

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: "subject_a" } });
    fireEvent.change(controlFollowing(panel, "Filename"), { target: { value: "counts.csv" } });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    expect(
      await within(panel).findByText(/Saved to C:\/proj\/results_export\/counts\.csv/),
    ).toBeInTheDocument();
    expect(within(panel).getByText(/Not validated\./)).toBeInTheDocument();
    expect(within(panel).getByText(/Acknowledged by user:breeder/)).toBeInTheDocument();
  });

  it("disables the Export button while a post is in flight", async () => {
    const panel = await renderCountPanel();
    let resolveDownload!: (value: {
      blob: Blob;
      headers: Awaited<ReturnType<typeof resultsApi.downloadCountCsv>>["headers"];
    }) => void;
    vi.spyOn(resultsApi, "downloadCountCsv").mockReturnValue(
      new Promise((resolve) => {
        resolveDownload = resolve;
      }),
    );

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: "subject_a" } });
    fireEvent.change(controlFollowing(panel, "Filename"), { target: { value: "counts.csv" } });
    const button = within(panel).getByRole("button", { name: /^export$/i });
    fireEvent.click(button);

    expect(await within(panel).findByRole("button", { name: /exporting/i })).toBeDisabled();

    resolveDownload({
      blob: new Blob(["x"]),
      headers: {
        savedTo: "C:/proj/results_export/counts.csv",
        validated: true,
        acknowledgedBy: "",
      },
    });
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
