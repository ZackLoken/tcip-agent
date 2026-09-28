import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { api } from "@/api/client";
import { StructuredRefusalError } from "@/api/http";
import { resultsApi, type DeliveryEventRecord } from "@/api/inference";
import { useStore } from "@/store";
import { ResultsTab } from "@/tabs/ResultsTab";
import { TRAIT_LISTINGS } from "@/test/traitRecords";

const initialStoreState = useStore.getState();

// Every Results door now returns the reconciled evidence beside its rows, so a mock that omits it
// would be describing a response the server cannot produce.
const VALIDATED = {
  validated: { operating_point: "validated_held_out", classifier: "validated_held_out" },
  validated_raw: { operating_point: "validated_held_out", classifier: "validated_held_out" },
  has_unvalidated_dimensions: false,
  validity_detail: {},
  positive_class_assessed: true,
  captures_unverified: [],
  plant_csvs_unverified: [],
  dates_delivered: [],
  images_unattributed: 0,
};

function setupDataset() {
  useStore.setState((s) => ({
    gui: {
      ...s.gui,
      dataset: {
        ...s.gui.dataset,
        project_root: "C:/proj",
        dataset_root: "C:/data",
      },
    },
  }));
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  setupDataset();
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
  it("defaults each date to its first model with predictions, and skips a date with none", async () => {
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01", "2026-01-08"],
      subjects: ["subject_a"],
      model_names: ["baseline", "v2"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline", "v2"], "2026-01-08": [] },
      prediction_dirs: {
        "2026-01-01": {
          baseline: "C:/data/predictions/baseline/2026-01-01",
          v2: "C:/data/predictions/v2/2026-01-01",
        },
        "2026-01-08": {
          baseline: "C:/data/predictions/baseline/2026-01-08",
          v2: "C:/data/predictions/v2/2026-01-08",
        },
      },
      label_problem: null,
    });
    const measurementSpy = vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [], n_plants: 0, positive_class_id: 1 },
      milestones: { rows: [] },
      ...VALIDATED,
    });

    render(<ResultsTab />);
    await waitFor(() => expect(api.dataset.tree).toHaveBeenCalledWith("C:/data"));
    // The trait resolves from the project's registered traits asynchronously; wait for it before
    // computing, or the compute click races the fetch and refuses (no trait resolved yet).
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());

    // The classified date defaults to "baseline" (first with predictions); the empty date shows
    // its own disabled "no predictions" option, never silently reusing the other date's model.
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    const selects = screen.getAllByTitle(
      /Model whose predictions to use for this date|No model has predictions for this date/,
    ) as HTMLSelectElement[];
    expect(selects).toHaveLength(2);
    expect(selects[0].value).toBe("baseline");
    expect(selects[1]).toBeDisabled();

    fireEvent.change(screen.getByLabelText(/plants to measure/i), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
    await waitFor(() => expect(measurementSpy).toHaveBeenCalled());
    // The dir is the one the tree response supplied for that (date, model), not a path the tab
    // assembled: a client-built convention is exactly what stopped matching the writers.
    expect(measurementSpy.mock.calls[0][0].predictions_by_date).toEqual({
      "2026-01-01": "C:/data/predictions/baseline/2026-01-01",
    });
  });

  it("shows the tree's label_problem beside the date list without blocking it", async () => {
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: ["subject_a"],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
      prediction_dirs: { "2026-01-01": { baseline: "C:/data/predictions/baseline/2026-01-01" } },
      label_problem: "C:/data/annotations/2026-01-08/IMG_0000.json does not decode as JSON",
    });
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [], n_plants: 0, positive_class_id: 1 },
      milestones: { rows: [] },
      ...VALIDATED,
    });

    render(<ResultsTab />);

    expect(
      await screen.findByText(
        "C:/data/annotations/2026-01-08/IMG_0000.json does not decode as JSON",
      ),
    ).toBeInTheDocument();
    expect(screen.getByText("2026-01-01")).toBeInTheDocument();
  });

  it("dropping a date to '(skip)' excludes it from the computed predictions map", async () => {
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: ["subject_a"],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
      prediction_dirs: { "2026-01-01": { baseline: "C:/data/predictions/baseline/2026-01-01" } },
      label_problem: null,
    });
    const measurementSpy = vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [], n_plants: 0, positive_class_id: 1 },
      milestones: { rows: [] },
      ...VALIDATED,
    });

    render(<ResultsTab />);
    // The trait resolves from the project's registered traits asynchronously; wait for it before
    // computing, or the compute click races the fetch and refuses (no trait resolved yet).
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());

    fireEvent.change(screen.getByTitle("Model whose predictions to use for this date"), {
      target: { value: "" },
    });
    fireEvent.change(screen.getByLabelText(/plants to measure/i), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
    await waitFor(() => expect(measurementSpy).toHaveBeenCalled());
    expect(measurementSpy.mock.calls[0][0].predictions_by_date).toEqual({});
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
    stage_50per_date: "2026-02-01",
  };
  const UNVALIDATED = {
    validated: { operating_point: "false", classifier: "validated_held_out" },
    validated_raw: { operating_point: "false", classifier: "validated_held_out" },
    has_unvalidated_dimensions: true,
    validity_detail: {},
    positive_class_assessed: true,
    captures_unverified: [],
    plant_csvs_unverified: [],
    dates_delivered: [],
    images_unattributed: 0,
  };

  function mockTree() {
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: [],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
      prediction_dirs: { "2026-01-01": { baseline: "C:/data/predictions/baseline/2026-01-01" } },
      label_problem: null,
    });
  }

  async function renderAndCompute() {
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText(/plants to measure/i), { target: { value: "P1" } });
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
  }

  it("hands an unvalidated-evidence refusal to the calibration flow, not the raw error line", async () => {
    mockTree();
    // Opens like the backend refusal message, which the tab must classify as a refusal.
    vi.spyOn(resultsApi, "phenologyMeasurement").mockRejectedValue(
      new Error(
        "phenology delivery requires a validated classifier and count operating point, " +
          "reconciled from the prediction buckets' own sidecars (never a caller-asserted " +
          "string). Unvalidated: ['operating_point'] (operating_point='missing', " +
          "classifier='validated_held_out').",
      ),
    );

    await renderAndCompute();

    expect(
      await screen.findByRole("button", { name: /ask the agent to calibrate this/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /show unvalidated numbers/i })).toBeInTheDocument();
  });

  it("replaces the disabled export with an acknowledge-and-export flow while unvalidated", async () => {
    useStore.setState({ user: "breeder" });
    mockTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [CURVE_ROW], n_plants: 1, positive_class_id: 1 },
      milestones: { rows: [ONSET_ROW] },
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
    expect(downloadCsv.mock.calls[0][0].acknowledgment).toEqual({
      reason: "calibration is not ready yet",
    });
  });

  it("disables the acknowledged-export buttons with a stated reason when no user is set", async () => {
    mockTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [CURVE_ROW], n_plants: 1, positive_class_id: 1 },
      milestones: { rows: [ONSET_ROW] },
      ...UNVALIDATED,
    });

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());

    fireEvent.click(screen.getByRole("button", { name: /acknowledge and export/i }));
    fireEvent.change(screen.getByPlaceholderText(/reason for delivering unvalidated/i), {
      target: { value: "calibration is not ready yet" },
    });

    expect(screen.getByText(/set your name on the workspace page/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /curves csv/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /milestones csv/i })).toBeDisabled();
  });

  it("opens both CSV doors once the same rows arrive on validated evidence", async () => {
    mockTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [CURVE_ROW], n_plants: 1, positive_class_id: 1 },
      milestones: { rows: [ONSET_ROW] },
      ...VALIDATED,
    });

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());

    expect(screen.getByRole("button", { name: /curves csv/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /milestones csv/i })).toBeEnabled();
    expect(screen.getByText("valid")).toBeInTheDocument();
  });

  it("toasts the second-delivery sentence naming the saved path when the CSV export's audit line is lost", async () => {
    mockTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [CURVE_ROW], n_plants: 1, positive_class_id: 1 },
      milestones: { rows: [ONSET_ROW] },
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
      expect(toast).toContain("supersede_delivery");
      expect(toast).toContain(message);
    });
  });

  it("states which delivery kinds the export controls actually cover", async () => {
    mockTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [CURVE_ROW], n_plants: 1, positive_class_id: 1 },
      milestones: { rows: [ONSET_ROW] },
      ...VALIDATED,
    });

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());

    expect(
      screen.getByText(/a phenology milestone delivery here, and a per-image or per-plant/i),
    ).toBeInTheDocument();
  });

  it("renders the delivery-scoped unattributed count beside the measurement", async () => {
    mockTree();
    vi.spyOn(resultsApi, "phenologyMeasurement").mockResolvedValue({
      curves: { rows: [CURVE_ROW], n_plants: 1, positive_class_id: 1 },
      milestones: { rows: [ONSET_ROW] },
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
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: ["subject_a"],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
      prediction_dirs: { "2026-01-01": { baseline: "C:/data/predictions/baseline/2026-01-01" } },
      label_problem: null,
    });
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
        positive_class_id: 1,
      },
      milestones: { rows },
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
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: ["subject_a"],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
      prediction_dirs: { "2026-01-01": { baseline: "C:/data/predictions/baseline/2026-01-01" } },
      label_problem: null,
    });
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
      screen.queryByRole("button", { name: /ask the agent to calibrate this/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /show unvalidated numbers/i }),
    ).not.toBeInTheDocument();
  });

  it("renders a structured refusal from the CSV download instead of stringifying it", async () => {
    const VALIDATED_EVIDENCE = {
      validated: { operating_point: "validated_held_out", classifier: "validated_held_out" },
      validated_raw: { operating_point: "validated_held_out", classifier: "validated_held_out" },
      has_unvalidated_dimensions: false,
      validity_detail: {},
      positive_class_assessed: true,
      captures_unverified: [],
      plant_csvs_unverified: [],
      dates_delivered: [],
      images_unattributed: 0,
    };
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: ["subject_a"],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
      prediction_dirs: { "2026-01-01": { baseline: "C:/data/predictions/baseline/2026-01-01" } },
      label_problem: null,
    });
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
        positive_class_id: 1,
      },
      milestones: { rows: [] },
      ...VALIDATED_EVIDENCE,
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
    // The calibration flow answers a different refusal family and must not be offered for this one.
    expect(
      screen.queryByRole("button", { name: /ask the agent to calibrate this/i }),
    ).not.toBeInTheDocument();
    vi.unstubAllGlobals();
  });
});

describe("ResultsTab delivery events (read-only)", () => {
  const DELIVERY_EVENT: DeliveryEventRecord = {
    event_id: "abc123",
    trait: "subject_a",
    trait_revision: 1,
    trait_revision_sha256: "b".repeat(64),
    delivery_kind: "state_crossing_dates",
    door: "results.export_csv",
    output_path: "C:/proj/results_export/subject_a_phenology.csv",
    output_sha256: "a".repeat(64),
    acknowledged_by: null,
    acknowledgment_reason: null,
    document_reconciliations: {
      operating_point: {
        validated: "false",
        on_disk_validated: false,
        missing_sidecars: ["C:/data/predictions/baseline/2026-01-08"],
        unvalidated_buckets: ["C:/data/predictions/baseline/2026-01-08"],
        binding_notes: {},
        bindings: {
          "C:/data/predictions/baseline/2026-01-01": {
            ok: true,
            claimed: true,
            experiment_id: "exp-1",
            producing_experiment_id: "exp-1",
            checkpoint_sha256: "abc",
            record_digest: "digest-1",
            note: "",
          },
          "C:/data/predictions/baseline/2026-01-08": {
            ok: false,
            claimed: false,
            experiment_id: null,
            producing_experiment_id: null,
            checkpoint_sha256: null,
            record_digest: null,
            note: "no stamp on this bucket",
          },
        },
        conf: null,
        confs: {},
        per_bucket: {},
      },
    },
    dimension_reconciliations: {},
    produced_at: "2026-02-03T12:00:00+00:00",
    plant_mapping: null,
    superseded: null,
  };

  it("lists what shipped, with real per-bucket verification evidence and no confirm/withdraw controls", async () => {
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [DELIVERY_EVENT] });

    render(<ResultsTab />);
    const row = await screen.findByTestId("delivery-abc123");

    expect(within(row).getByText("subject_a")).toBeInTheDocument();
    expect(within(row).getByText("revision 1, state_crossing_dates")).toBeInTheDocument();
    expect(within(row).getByText("results.export_csv")).toBeInTheDocument();
    expect(within(row).getByText(/2026-02-03T12:00:00\+00:00/)).toBeInTheDocument();
    expect(
      within(row).getByText("C:/proj/results_export/subject_a_phenology.csv"),
    ).toBeInTheDocument();
    expect(
      within(row).getByText("C:/data/predictions/baseline/2026-01-01: verified"),
    ).toBeInTheDocument();
    expect(
      within(row).getByText("C:/data/predictions/baseline/2026-01-08: no claim"),
    ).toBeInTheDocument();
    expect(within(row).queryByRole("button", { name: /confirm/i })).not.toBeInTheDocument();
    expect(within(row).queryByRole("button", { name: /withdraw/i })).not.toBeInTheDocument();
    expect(within(row).queryByRole("button", { name: /correction/i })).not.toBeInTheDocument();
  });

  it("renders the acknowledging breeder's name and reason on an acknowledged delivery", async () => {
    const acknowledged: DeliveryEventRecord = {
      ...DELIVERY_EVENT,
      event_id: "acked",
      acknowledged_by: "user:breeder",
      acknowledgment_reason: "calibration is not ready yet",
    };
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [acknowledged] });

    render(<ResultsTab />);
    const row = await screen.findByTestId("delivery-acked");

    expect(within(row).getByText("user:breeder")).toBeInTheDocument();
    expect(within(row).getByText("calibration is not ready yet")).toBeInTheDocument();
  });

  it("renders a stated 'no file' for a delivery event with no output path", async () => {
    const fileless: DeliveryEventRecord = {
      ...DELIVERY_EVENT,
      event_id: "fileless",
      output_path: null,
    };
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [fileless] });

    render(<ResultsTab />);
    const row = await screen.findByTestId("delivery-fileless");

    expect(within(row).getByText("no file")).toBeInTheDocument();
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
    const withMapping: DeliveryEventRecord = {
      ...DELIVERY_EVENT,
      event_id: "with-mapping",
      plant_mapping: {
        name: "valley",
        project_root: "C:/proj",
        dataset_id: "ds-1",
        dataset_root: "C:/data",
        built_at: "2026-02-01T00:00:00+00:00",
        record_sha256: "0".repeat(64),
        nn_tolerance_m: { value: 3, source: "stated" },
        capture_identity: {},
        captures_unverified: [],
        plant_csvs_unverified: [],
        dates_delivered: ["2026-01-01", "2026-01-08"],
        images_unattributed: 2,
        images_unattributed_scope: "delivered_dates",
        plant_attribution: "image",
      },
    };
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [withMapping] });

    render(<ResultsTab />);
    const row = await screen.findByTestId("delivery-with-mapping");

    expect(
      within(row).getByText(/Delivered dates 2026-01-01, 2026-01-08: 2 attributed to no plant/),
    ).toBeInTheDocument();
    expect(within(row).getByText(/image-level attribution/)).toBeInTheDocument();
  });

  it("renders the orthomosaic door's own registry disclosure, not the walked-mapping form", async () => {
    // Covers deliver_orthomosaic_plant_counts's PlantRegistryDisclosure: no dates_delivered or
    // record_sha256 to render, since no walked mapping exists for a whole-raster frame.
    const withRegistry: DeliveryEventRecord = {
      ...DELIVERY_EVENT,
      event_id: "with-registry",
      door: "deliver_orthomosaic_plant_counts",
      plant_mapping: {
        plant_registry: { name: "orchard-block", digest: "0".repeat(64) },
        project_root: "C:/proj",
        raster_identity: { width: 4096, height: 4096 },
        nn_tolerance_m: { value: 1.5, source: "grid_pitch" },
        detections_unattributed: 3,
        detections_unattributed_scope: "delivered_raster",
        plant_attribution: "detection",
        plants_outside_raster: ["plot9"],
      },
    };
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [withRegistry] });

    render(<ResultsTab />);
    const row = await screen.findByTestId("delivery-with-registry");

    expect(within(row).getByText(/Plant registry orchard-block:/)).toBeInTheDocument();
    expect(
      within(row).getByText(/3 detection\(s\) attributed to no plant on the delivered raster/),
    ).toBeInTheDocument();
    expect(within(row).getByText(/detection-level attribution/)).toBeInTheDocument();
    expect(within(row).getByText(/Outside the raster: plot9/)).toBeInTheDocument();
    expect(within(row).queryByText(/Delivered dates/)).not.toBeInTheDocument();
  });

  it("renders the orthomosaic door's own canopy-segment disclosure, narrowed before the registry form", async () => {
    const withCanopy: DeliveryEventRecord = {
      ...DELIVERY_EVENT,
      event_id: "with-canopy",
      door: "deliver_orthomosaic_plant_counts",
      plant_mapping: {
        plant_registry: { name: "orchard-block", digest: "0".repeat(64) },
        project_root: "C:/proj",
        raster_identity: { width: 4096, height: 4096 },
        canopy_segments: {
          path: "C:/proj/annotations/2024-06-01/mosaic.json",
          sha256: "1".repeat(64),
          subject: "canopy",
          n_segments: 3,
        },
        segment_ties: [
          { segment_index: 0, plot_name: "plot0", clearance_m: 0.8 },
          { segment_index: 1, plot_name: "plot1", clearance_m: 1.2 },
        ],
        segments_without_plant: 1,
        plants_outside_raster: ["plot9"],
        plants_without_segment: ["plot8"],
        plants_with_ambiguous_detections: ["plot1"],
        detections_unattributed: 2,
        detections_unattributed_by_source: {
          outside_segments: 1,
          overlapping_segments: 1,
          segment_without_plant: 0,
        },
        detections_unattributed_scope: "delivered_raster",
        plant_attribution: "segment",
      },
    };
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [withCanopy] });

    render(<ResultsTab />);
    const row = await screen.findByTestId("delivery-with-canopy");

    expect(within(row).getByTestId("canopy-disclosure")).toBeInTheDocument();
    expect(within(row).getByText(/Canopy segments \(orchard-block\): 1\/4/)).toBeInTheDocument();
    expect(within(row).getByText(/No segment: plot8/)).toBeInTheDocument();
    expect(within(row).getByText(/Outside the raster: plot9/)).toBeInTheDocument();
    expect(within(row).getByText(/Ambiguous detection: plot1/)).toBeInTheDocument();
    expect(within(row).queryByText(/Plant registry orchard-block:/)).not.toBeInTheDocument();
    expect(within(row).queryByText(/Delivered dates/)).not.toBeInTheDocument();
  });

  it("renders the archived key beside a cited mapping once a rebuild has moved past it", async () => {
    const withMapping: DeliveryEventRecord = {
      ...DELIVERY_EVENT,
      event_id: "moved-on",
      plant_mapping: {
        name: "valley",
        project_root: "C:/proj",
        dataset_id: "ds-1",
        dataset_root: "C:/data",
        built_at: "2026-02-01T00:00:00+00:00",
        record_sha256: "0".repeat(64),
        nn_tolerance_m: { value: 3, source: "stated" },
        capture_identity: {},
        captures_unverified: [],
        plant_csvs_unverified: [],
        dates_delivered: ["2026-01-01"],
        images_unattributed: 0,
        images_unattributed_scope: "delivered_dates",
        plant_attribution: "image",
      },
      plant_mapping_resolved_key: "valley@0123456789ab",
    };
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [withMapping] });

    render(<ResultsTab />);
    const row = await screen.findByTestId("delivery-moved-on");

    expect(within(row).getByText(/archived as valley@0123456789ab/)).toBeInTheDocument();
  });

  it("renders a supersession's reason and replacement", async () => {
    const superseded: DeliveryEventRecord = {
      ...DELIVERY_EVENT,
      event_id: "superseded-1",
      superseded: {
        superseded_event_id: "superseded-1",
        output_sha256: "abc",
        replacement_event_id: "replacement-1",
        reason: "a mis-stated crop was corrected upstream",
        superseded_by: "supersede_delivery",
        superseded_at: "2026-02-04T00:00:00+00:00",
      },
    };
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [superseded] });

    render(<ResultsTab />);
    const row = await screen.findByTestId("delivery-superseded-1");

    expect(
      within(row).getByText(/Superseded: a mis-stated crop was corrected upstream/),
    ).toBeInTheDocument();
    expect(within(row).getByText(/replaced by replacement-1/)).toBeInTheDocument();
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

  function mockCountTree() {
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: [],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
      prediction_dirs: { "2026-01-01": { baseline: "C:/data/predictions/baseline/2026-01-01" } },
      label_problem: null,
    });
  }

  // Every field in the count-export panel sits immediately after its own <label>, the DOM
  // relationship this reads rather than a positional guess at render order.
  function controlFollowing(panel: HTMLElement, labelText: string): HTMLElement {
    const label = within(panel).getByText(labelText);
    const control = label.nextElementSibling;
    if (!control) throw new Error(`no control follows the ${labelText} label`);
    return control as HTMLElement;
  }

  async function renderCountPanel(): Promise<HTMLElement> {
    mockCountTree();
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(api.dataset.tree).toHaveBeenCalled());
    const heading = await screen.findByText("Count export");
    const panel = heading.closest(".tcip-panel");
    if (!panel) throw new Error("Count export panel not found");
    // The Trait and Prediction bucket selects only fill in once their own async loads resolve.
    await within(panel as HTMLElement).findByText("subject_a");
    await within(panel as HTMLElement).findByText("2026-01-01 (baseline)");
    return panel as HTMLElement;
  }

  function chooseBucket(panel: HTMLElement) {
    fireEvent.change(controlFollowing(panel, "Prediction bucket"), {
      target: { value: "2026-01-01 baseline" },
    });
  }

  it("posts a per_image_count delivery with the selected bucket and trait", async () => {
    const panel = await renderCountPanel();
    const downloadCountCsv = vi.spyOn(resultsApi, "downloadCountCsv").mockResolvedValue({
      blob: new Blob(["x"]),
      headers: {
        savedTo: "C:/proj/results_export/counts.csv",
        unvalidatedDimensions: "",
        acknowledgedBy: "",
      },
    });

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: "subject_a" } });
    fireEvent.change(controlFollowing(panel, "Filename"), { target: { value: "counts.csv" } });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    await waitFor(() => expect(downloadCountCsv).toHaveBeenCalled());
    expect(downloadCountCsv.mock.calls[0][0]).toMatchObject({
      project_root: "C:/proj",
      delivery: {
        kind: "per_image_count",
        predictions_dir: "C:/data/predictions/baseline/2026-01-01",
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
        unvalidatedDimensions: "",
        acknowledgedBy: "",
      },
    });

    fireEvent.change(controlFollowing(panel, "Kind"), {
      target: { value: "orthomosaic_plant_counts" },
    });
    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Raster path"), {
      target: { value: "C:/data/mosaic.tif" },
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
      project_root: "C:/proj",
      delivery: {
        kind: "orthomosaic_plant_counts",
        predictions_dir: "C:/data/predictions/baseline/2026-01-01",
        raster_path: "C:/data/mosaic.tif",
        plant_registry: "reg",
        delivered_phenotype: "stem_count",
      },
      filename: "plant_counts.csv",
    });
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
    expect(errorText.textContent).toContain("supersede_delivery");
    expect(errorText.textContent).toContain(message);
  });

  it("decodes a delivery_gate refusal and offers the acknowledgment controls", async () => {
    useStore.setState({ user: "breeder" });
    const panel = await renderCountPanel();
    vi.spyOn(resultsApi, "downloadCountCsv").mockRejectedValue(
      new StructuredRefusalError(
        {
          kind: "delivery_gate",
          message: "delivery refused: unvalidated dimension(s) ['operating_point'].",
          unvalidated_dimensions: "operating_point",
        },
        400,
        "delivery_gate",
      ),
    );

    chooseBucket(panel);
    fireEvent.change(controlFollowing(panel, "Trait"), { target: { value: "subject_a" } });
    fireEvent.change(controlFollowing(panel, "Filename"), { target: { value: "counts.csv" } });
    fireEvent.click(within(panel).getByRole("button", { name: /^export$/i }));

    expect(
      await within(panel).findByText(/unvalidated dimension\(s\) \['operating_point'\]/),
    ).toBeInTheDocument();
    expect(
      within(panel).getByPlaceholderText(/reason for delivering unvalidated/i),
    ).toBeInTheDocument();
    expect(
      within(panel).getByRole("button", { name: /acknowledge and export/i }),
    ).toBeInTheDocument();
  });

  it("downloads with the four headers and reports from them", async () => {
    const panel = await renderCountPanel();
    vi.spyOn(resultsApi, "downloadCountCsv").mockResolvedValue({
      blob: new Blob(["x"]),
      headers: {
        savedTo: "C:/proj/results_export/counts.csv",
        unvalidatedDimensions: "operating_point",
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
    expect(within(panel).getByText(/Unvalidated: operating_point/)).toBeInTheDocument();
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
        unvalidatedDimensions: "",
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
