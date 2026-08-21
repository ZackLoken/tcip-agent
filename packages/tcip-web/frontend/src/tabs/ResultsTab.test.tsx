import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { api } from "@/api/client";
import { StructuredRefusalError } from "@/api/http";
import {
  resultsApi,
  type DeliveryEventRecord,
  type OperationalizationRecord,
  type TraitSpecStatementRecord,
} from "@/api/inference";
import { useStore } from "@/store";
import { ResultsTab } from "@/tabs/ResultsTab";

const initialStoreState = useStore.getState();

// Every Results door now returns the reconciled evidence beside its rows, so a mock that omits it
// would be describing a response the server cannot produce.
const VALIDATED = {
  validated: { operating_point: "validated_held_out", classifier: "validated_held_out" },
  provisional: false,
  validity_detail: {},
  positive_class_assessed: true,
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
  // ResultsTab resolves its trait from the project's own registered traits before it will
  // compute anything; every test fixture here is written against a single-trait project whose
  // spec declares milestone fractions, which is what the curve/milestone panels render for.
  vi.spyOn(resultsApi, "traits").mockResolvedValue({
    traits: ["subject_a"],
    milestone_fractions_by_trait: { subject_a: [0.5, 0.95] },
    invalid_specs: [],
  });
  // The operationalization, trait-spec, and delivery-events panels all load with the tab; a test
  // about anything else has no records.
  vi.spyOn(resultsApi, "operationalizations").mockResolvedValue({
    records: [],
    statement_fields: [],
  });
  vi.spyOn(resultsApi, "traitSpecStatements").mockResolvedValue({
    records: [],
    unresolved: [],
    statement_fields: [],
  });
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
    });
    const curvesSpy = vi.spyOn(resultsApi, "perPlantCurves").mockResolvedValue({
      rows: [],
      n_plants: 0,
      positive_class_id: 1,
      ...VALIDATED,
    });
    vi.spyOn(resultsApi, "onsetDates").mockResolvedValue({ rows: [], ...VALIDATED });

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

    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
    await waitFor(() => expect(curvesSpy).toHaveBeenCalled());
    // The dir is the one the tree response supplied for that (date, model), not a path the tab
    // assembled: a client-built convention is exactly what stopped matching the writers.
    expect(curvesSpy.mock.calls[0][0].predictions_by_date).toEqual({
      "2026-01-01": "C:/data/predictions/baseline/2026-01-01",
    });
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
    });
    const curvesSpy = vi.spyOn(resultsApi, "perPlantCurves").mockResolvedValue({
      rows: [],
      n_plants: 0,
      positive_class_id: 1,
      ...VALIDATED,
    });
    vi.spyOn(resultsApi, "onsetDates").mockResolvedValue({ rows: [], ...VALIDATED });

    render(<ResultsTab />);
    // The trait resolves from the project's registered traits asynchronously; wait for it before
    // computing, or the compute click races the fetch and refuses (no trait resolved yet).
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());

    fireEvent.change(screen.getByTitle("Model whose predictions to use for this date"), {
      target: { value: "" },
    });
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
    await waitFor(() => expect(curvesSpy).toHaveBeenCalled());
    expect(curvesSpy.mock.calls[0][0].predictions_by_date).toEqual({});
  });
});

describe("ResultsTab broken trait spec visibility", () => {
  it("names a broken spec file and its reason, not just a blank/empty tab", async () => {
    vi.spyOn(resultsApi, "traits").mockResolvedValue({
      traits: ["subject_a"],
      milestone_fractions_by_trait: { subject_a: [0.5, 0.95] },
      invalid_specs: [
        {
          file: "leaf_area.yml",
          reason: "delivers must be non-empty and all in crops.yml (off-vocab: ['leaf_size'])",
        },
      ],
    });

    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());

    expect(await screen.findByText(/leaf_area\.yml/)).toBeInTheDocument();
    expect(screen.getByText(/off-vocab: \['leaf_size'\]/)).toBeInTheDocument();
  });

  it("renders nothing extra when every spec loaded cleanly", async () => {
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());

    expect(screen.queryByText(/failed to load/)).not.toBeInTheDocument();
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
    provisional: true,
    validity_detail: {},
    positive_class_assessed: true,
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
    });
  }

  async function renderAndCompute() {
    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));
  }

  it("hands an unvalidated-evidence refusal to the calibration flow, not the raw error line", async () => {
    mockTree();
    // Opens like the backend refusal message, which the tab must classify as a refusal.
    vi.spyOn(resultsApi, "perPlantCurves").mockRejectedValue(
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
    expect(screen.getByRole("button", { name: /show provisional numbers/i })).toBeInTheDocument();
  });

  it("keeps both CSV doors shut while the displayed numbers are provisional", async () => {
    mockTree();
    vi.spyOn(resultsApi, "perPlantCurves").mockResolvedValue({
      rows: [CURVE_ROW],
      n_plants: 1,
      positive_class_id: 1,
      ...UNVALIDATED,
    });
    vi.spyOn(resultsApi, "onsetDates").mockResolvedValue({ rows: [ONSET_ROW], ...UNVALIDATED });

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());

    expect(screen.getByRole("button", { name: /curves csv/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /milestones csv/i })).toBeDisabled();
    // The row itself has to say so too; the banner explaining it scrolls out of view.
    expect(screen.getByText("provisional")).toBeInTheDocument();
    expect(screen.queryByText("valid")).not.toBeInTheDocument();
  });

  it("opens both CSV doors once the same rows arrive on validated evidence", async () => {
    mockTree();
    vi.spyOn(resultsApi, "perPlantCurves").mockResolvedValue({
      rows: [CURVE_ROW],
      n_plants: 1,
      positive_class_id: 1,
      ...VALIDATED,
    });
    vi.spyOn(resultsApi, "onsetDates").mockResolvedValue({ rows: [ONSET_ROW], ...VALIDATED });

    await renderAndCompute();
    await waitFor(() => expect(screen.getByText("P1")).toBeInTheDocument());

    expect(screen.getByRole("button", { name: /curves csv/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /milestones csv/i })).toBeEnabled();
    expect(screen.getByText("valid")).toBeInTheDocument();
  });
});

describe("ResultsTab onset table validity marker", () => {
  async function computeWithOnsetRows(
    rows: Awaited<ReturnType<typeof resultsApi.onsetDates>>["rows"],
  ) {
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: ["subject_a"],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
      prediction_dirs: { "2026-01-01": { baseline: "C:/data/predictions/baseline/2026-01-01" } },
    });
    vi.spyOn(resultsApi, "perPlantCurves").mockResolvedValue({
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
      ...VALIDATED,
    });
    vi.spyOn(resultsApi, "onsetDates").mockResolvedValue({ rows, ...VALIDATED });

    render(<ResultsTab />);
    // The trait resolves from the project's registered traits asynchronously; wait for it before
    // computing, or the compute click races the fetch and refuses (no trait resolved yet).
    await waitFor(() => expect(resultsApi.traits).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
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

describe("ResultsTab operationalization records", () => {
  // Every hashed field is non-empty here, so rendering is asserted against the hashed set itself.
  const COUNT_RECORD: OperationalizationRecord = {
    trait: "subject_b_total",
    delivery_kind: "per_plant_count_aggregate",
    statement: "Every isolated object of the subject on a plant, summed over that plant's images.",
    mechanism: "The detector's objects at the operating point the calibration holdout fixed.",
    measured_subject: "subject_b",
    delivered_phenotypes: ["subject_b_count"],
    delivered_value_keys: ["n_objects"],
    stated_by: "state_trait_operationalization",
    stated_at: "2026-02-01T10:00:00+00:00",
    relayed_note: "Answered in the packing shed rather than in the browser.",
    agent_client_name: null,
    agent_client_version: null,
    agent_session: null,
    terminal_session: null,
    harness_session: null,
    harness_effort_at_connect: null,
    confirmed_by: null,
    confirmed_at: null,
    identity_from_request: null,
    confirmed_current: false,
    superseded: [],
    delivers: [{ name: "subject_b_count", definition: "objects counted per plant" }],
    record_seen: "hash-of-the-displayed-record",
  };

  // What the list route answers, the server's own naming of the fields the record_seen hash covers.
  const LISTED = {
    records: [COUNT_RECORD],
    statement_fields: [
      "statement",
      "mechanism",
      "measured_subject",
      "delivered_phenotypes",
      "delivered_value_keys",
      "stated_by",
      "stated_at",
      "relayed_note",
    ],
  };

  const CONFIRMED_RECORD: OperationalizationRecord = {
    ...COUNT_RECORD,
    confirmed_by: "user:breeder",
    confirmed_at: "2026-02-02T09:00:00+00:00",
    identity_from_request: true,
    confirmed_current: true,
  };

  function rowFor(record: OperationalizationRecord) {
    return screen.findByTestId(`operationalization-${record.trait}::${record.delivery_kind}`);
  }

  it("shows every field the served list names as covered by a confirmation", async () => {
    vi.spyOn(resultsApi, "operationalizations").mockResolvedValue(LISTED);

    render(<ResultsTab />);
    const row = await rowFor(COUNT_RECORD);

    const values: Record<string, unknown> = COUNT_RECORD;
    for (const field of LISTED.statement_fields) {
      const value = values[field];
      expect(
        within(row).getByText(Array.isArray(value) ? value.join(", ") : String(value)),
      ).toBeInTheDocument();
    }
    // The crop vocabulary's own wording for what the trait delivers, beside the statement.
    expect(within(row).getByText(/objects counted per plant/)).toBeInTheDocument();
  });

  it("shows only what the served list names, so a field the server drops leaves the row", async () => {
    vi.spyOn(resultsApi, "operationalizations").mockResolvedValue({
      records: [COUNT_RECORD],
      statement_fields: ["statement"],
    });

    render(<ResultsTab />);
    const row = await rowFor(COUNT_RECORD);

    expect(within(row).getByText(COUNT_RECORD.statement)).toBeInTheDocument();
    expect(within(row).queryByText(COUNT_RECORD.mechanism)).not.toBeInTheDocument();
  });

  it("lists a record whose delivery kind this tab computes no view for", async () => {
    vi.spyOn(resultsApi, "operationalizations").mockResolvedValue(LISTED);

    render(<ResultsTab />);
    const row = await rowFor(COUNT_RECORD);

    expect(within(row).getByText("per_plant_count_aggregate")).toBeInTheDocument();
    expect(within(row).getByRole("button", { name: /confirm this record/i })).toBeEnabled();
  });

  it("confirms with the record_seen hash of what was displayed", async () => {
    vi.spyOn(resultsApi, "operationalizations").mockResolvedValue(LISTED);
    const confirmSpy = vi.spyOn(resultsApi, "confirmOperationalization").mockResolvedValue({
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      confirmed_fields: { count_objective: "every visible object" },
      audit_warning: null,
    });
    vi.spyOn(resultsApi, "operationalization").mockResolvedValue({
      ...COUNT_RECORD,
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      confirmed_current: true,
    });

    useStore.setState({ user: "breeder" });
    render(<ResultsTab />);
    const row = await rowFor(COUNT_RECORD);
    fireEvent.click(within(row).getByRole("button", { name: /confirm this record/i }));

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    // The app-set name rides the post, so the record's identity_from_request stays honest.
    expect(confirmSpy.mock.calls[0][0]).toEqual({
      project_root: "C:/proj",
      trait: "subject_b_total",
      delivery_kind: "per_plant_count_aggregate",
      record_seen: "hash-of-the-displayed-record",
      confirmed: true,
      user: "breeder",
    });
    expect(await within(row).findByText(/confirmed by user:breeder/i)).toBeInTheDocument();
  });

  it("re-renders what is on file when the record moved since it was displayed", async () => {
    const MOVED: OperationalizationRecord = {
      ...COUNT_RECORD,
      statement: "Only the objects on the plant's own leader, summed over that plant's images.",
      record_seen: "hash-of-the-record-on-file",
    };
    vi.spyOn(resultsApi, "operationalizations").mockResolvedValue(LISTED);
    vi.spyOn(resultsApi, "confirmOperationalization").mockRejectedValue(
      new StructuredRefusalError(
        { message: "the operationalization moved since it was read", record: MOVED },
        409,
        "the operationalization moved since it was read",
      ),
    );

    render(<ResultsTab />);
    const row = await rowFor(COUNT_RECORD);
    fireEvent.click(within(row).getByRole("button", { name: /confirm this record/i }));

    expect(await within(row).findByText(MOVED.statement)).toBeInTheDocument();
    expect(within(row).queryByText(COUNT_RECORD.statement)).not.toBeInTheDocument();
    expect(within(row).getByText(/changed since it was shown/i)).toBeInTheDocument();
  });

  it("offers no withdrawal for a record nobody has confirmed", async () => {
    vi.spyOn(resultsApi, "operationalizations").mockResolvedValue(LISTED);

    render(<ResultsTab />);
    const row = await rowFor(COUNT_RECORD);

    expect(within(row).queryByRole("button", { name: /withdraw/i })).not.toBeInTheDocument();
  });

  it("withdraws with confirmed false and the row comes back unconfirmed", async () => {
    vi.spyOn(resultsApi, "operationalizations").mockResolvedValue({
      records: [CONFIRMED_RECORD],
      statement_fields: LISTED.statement_fields,
    });
    const confirmSpy = vi.spyOn(resultsApi, "confirmOperationalization").mockResolvedValue({
      confirmed_by: null,
      confirmed_at: null,
      identity_from_request: null,
      confirmed_fields: null,
      audit_warning: null,
    });
    vi.spyOn(resultsApi, "operationalization").mockResolvedValue(COUNT_RECORD);

    render(<ResultsTab />);
    const row = await rowFor(CONFIRMED_RECORD);
    fireEvent.click(within(row).getByRole("button", { name: /withdraw this confirmation/i }));

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    expect(confirmSpy.mock.calls[0][0]).toEqual({
      project_root: "C:/proj",
      trait: "subject_b_total",
      delivery_kind: "per_plant_count_aggregate",
      record_seen: "hash-of-the-displayed-record",
      confirmed: false,
    });
    expect(await within(row).findByText("Not confirmed")).toBeInTheDocument();
    expect(within(row).getByRole("button", { name: /confirm this record/i })).toBeEnabled();
    expect(within(row).queryByRole("button", { name: /withdraw/i })).not.toBeInTheDocument();
  });

  it("re-renders what is on file when the record moved since the withdrawal was offered", async () => {
    const MOVED: OperationalizationRecord = {
      ...CONFIRMED_RECORD,
      statement: "Only the objects on the plant's own leader, summed over that plant's images.",
      record_seen: "hash-of-the-record-on-file",
    };
    vi.spyOn(resultsApi, "operationalizations").mockResolvedValue({
      records: [CONFIRMED_RECORD],
      statement_fields: LISTED.statement_fields,
    });
    vi.spyOn(resultsApi, "confirmOperationalization").mockRejectedValue(
      new StructuredRefusalError(
        { message: "the operationalization moved since it was read", record: MOVED },
        409,
        "the operationalization moved since it was read",
      ),
    );

    render(<ResultsTab />);
    const row = await rowFor(CONFIRMED_RECORD);
    fireEvent.click(within(row).getByRole("button", { name: /withdraw this confirmation/i }));

    expect(await within(row).findByText(MOVED.statement)).toBeInTheDocument();
    expect(within(row).queryByText(CONFIRMED_RECORD.statement)).not.toBeInTheDocument();
    expect(within(row).getByText(/changed since it was shown/i)).toBeInTheDocument();
  });

  it("routes a refusal by its kind even when its text reads like the calibration one", async () => {
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: ["subject_a"],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
      prediction_dirs: { "2026-01-01": { baseline: "C:/data/predictions/baseline/2026-01-01" } },
    });
    vi.spyOn(resultsApi, "perPlantCurves").mockRejectedValue(
      new StructuredRefusalError(
        {
          kind: "operationalization",
          state: 2,
          trait: "subject_a",
          delivery_kind: "state_crossing_dates",
          message: "an unvalidated number is not what this refusal is about",
        },
        400,
        "an unvalidated number is not what this refusal is about",
      ),
    );

    render(<ResultsTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /compute curves/i }));

    expect(
      await screen.findByText(/an unvalidated number is not what this refusal is about/),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /ask the agent to calibrate this/i }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /show provisional numbers/i }),
    ).not.toBeInTheDocument();
  });

  it("renders a structured refusal from the CSV download instead of stringifying it", async () => {
    const VALIDATED_EVIDENCE = {
      validated: { operating_point: "validated_held_out", classifier: "validated_held_out" },
      provisional: false,
      validity_detail: {},
      positive_class_assessed: true,
    };
    vi.spyOn(api.dataset, "tree").mockResolvedValue({
      dataset_root: "C:/data",
      dates_with_images: ["2026-01-01"],
      subjects: ["subject_a"],
      model_names: ["baseline"],
      subjects_by_date: {},
      models_by_date: { "2026-01-01": ["baseline"] },
      prediction_dirs: { "2026-01-01": { baseline: "C:/data/predictions/baseline/2026-01-01" } },
    });
    vi.spyOn(resultsApi, "perPlantCurves").mockResolvedValue({
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
      ...VALIDATED_EVIDENCE,
    });
    vi.spyOn(resultsApi, "onsetDates").mockResolvedValue({ rows: [], ...VALIDATED_EVIDENCE });
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
            state: 2,
            trait: "subject_a",
            delivery_kind: "state_crossing_dates",
            message: "stated but not confirmed by the breeder",
          },
        }),
      } as Response),
    );

    render(<ResultsTab />);
    await waitFor(() => expect(screen.getByText("2026-01-01")).toBeInTheDocument());
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

describe("ResultsTab trait-spec authoring statements", () => {
  // Every authored value below is distinct once rendered, so a `getByText` match inside the row
  // is unambiguous: two authored fields that both rendered "none" would collide.
  const TRAIT_SPEC_RECORD: TraitSpecStatementRecord = {
    trait: "subject_a",
    statement_fields: {
      delivers: ["subject_a_50per_date"],
      positive_class_name: "subject_a_open",
      milestone_fractions: [0.5, 0.95],
      milestone_on: "positive_fraction",
      majority_milestone: "95per",
      majority_provisional: true,
      phenology_prefix: "subj_a_col",
      majority_label: "most subject_a open",
      count_objective: "count_unbiased",
      count_bias_tolerance_frac: 0.1,
      count_error_tolerance: 2.5,
      classifier_agreement_floor: 0.41,
      ordinal_agreement_floor: 0.4,
      regression_skill_floor: 0.6,
      notes: "Authored from the packing-shed conversation.",
    },
    rationale: "Breeder said the 50% and 95% open crossing dates matter most for harvest timing.",
    stated_by: "author_trait_spec",
    stated_at: "2026-02-01T10:00:00+00:00",
    relayed_note: "Answered during the site visit.",
    agent_client_name: null,
    agent_client_version: null,
    agent_session: null,
    terminal_session: null,
    harness_session: null,
    harness_effort_at_connect: null,
    confirmed_by: null,
    confirmed_at: null,
    identity_from_request: null,
    confirmed_current: false,
    record_seen: "hash-of-the-displayed-trait-spec",
  };

  const LISTED_TRAIT_SPEC = {
    records: [TRAIT_SPEC_RECORD],
    unresolved: [],
    statement_fields: ["statement_fields", "rationale", "stated_by", "stated_at", "relayed_note"],
  };

  function rowFor(record: TraitSpecStatementRecord) {
    return screen.findByTestId(`trait-spec-statement-${record.trait}`);
  }

  it("shows the rationale and every authored field the served statement carries", async () => {
    vi.spyOn(resultsApi, "traitSpecStatements").mockResolvedValue(LISTED_TRAIT_SPEC);

    render(<ResultsTab />);
    const row = await rowFor(TRAIT_SPEC_RECORD);

    expect(within(row).getByText(TRAIT_SPEC_RECORD.rationale!)).toBeInTheDocument();
    expect(within(row).getByText(TRAIT_SPEC_RECORD.stated_by!)).toBeInTheDocument();
    expect(within(row).getByText(TRAIT_SPEC_RECORD.relayed_note!)).toBeInTheDocument();
    const authored: Record<string, unknown> = TRAIT_SPEC_RECORD.statement_fields!;
    for (const value of Object.values(authored)) {
      const text = Array.isArray(value) ? value.join(", ") : value === true ? "yes" : String(value);
      expect(within(row).getByText(text)).toBeInTheDocument();
    }
  });

  it("shows only what the served list names, so a field the server drops leaves the row", async () => {
    vi.spyOn(resultsApi, "traitSpecStatements").mockResolvedValue({
      records: [TRAIT_SPEC_RECORD],
      unresolved: [],
      statement_fields: ["rationale"],
    });

    render(<ResultsTab />);
    const row = await rowFor(TRAIT_SPEC_RECORD);

    expect(within(row).getByText(TRAIT_SPEC_RECORD.rationale!)).toBeInTheDocument();
    expect(within(row).queryByText(TRAIT_SPEC_RECORD.stated_by!)).not.toBeInTheDocument();
    expect(within(row).queryByText("subject_a_open")).not.toBeInTheDocument();
  });

  it("confirms with the record_seen hash of what was displayed and posts record_seen", async () => {
    vi.spyOn(resultsApi, "traitSpecStatements").mockResolvedValue(LISTED_TRAIT_SPEC);
    const confirmSpy = vi.spyOn(resultsApi, "confirmTraitSpecStatement").mockResolvedValue({
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      record_seen: "hash-of-the-displayed-trait-spec",
      audit_warning: null,
    });
    vi.spyOn(resultsApi, "traitSpecStatement").mockResolvedValue({
      ...TRAIT_SPEC_RECORD,
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      confirmed_current: true,
    });

    useStore.setState({ user: "breeder" });
    render(<ResultsTab />);
    const row = await rowFor(TRAIT_SPEC_RECORD);
    fireEvent.click(within(row).getByRole("button", { name: /confirm this record/i }));

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    expect(confirmSpy.mock.calls[0][0]).toEqual({
      project_root: "C:/proj",
      trait: "subject_a",
      record_seen: "hash-of-the-displayed-trait-spec",
      confirmed: true,
      user: "breeder",
    });
    expect(await within(row).findByText(/confirmed by user:breeder/i)).toBeInTheDocument();
  });

  it("re-renders what is on file when the statement moved since it was displayed", async () => {
    const MOVED: TraitSpecStatementRecord = {
      ...TRAIT_SPEC_RECORD,
      rationale: "Revised: only the 95% crossing date matters now.",
      record_seen: "hash-of-the-statement-on-file",
    };
    vi.spyOn(resultsApi, "traitSpecStatements").mockResolvedValue(LISTED_TRAIT_SPEC);
    vi.spyOn(resultsApi, "confirmTraitSpecStatement").mockRejectedValue(
      new StructuredRefusalError(
        {
          kind: "trait_spec_authoring",
          message: "the trait-spec statement moved since it was read",
          record: MOVED,
        },
        409,
        "the trait-spec statement moved since it was read",
      ),
    );

    render(<ResultsTab />);
    const row = await rowFor(TRAIT_SPEC_RECORD);
    fireEvent.click(within(row).getByRole("button", { name: /confirm this record/i }));

    expect(await within(row).findByText(MOVED.rationale!)).toBeInTheDocument();
    expect(within(row).queryByText(TRAIT_SPEC_RECORD.rationale!)).not.toBeInTheDocument();
    expect(within(row).getByText(/changed since it was shown/i)).toBeInTheDocument();
  });
});

describe("ResultsTab audit_warning banner (A8)", () => {
  const COUNT_RECORD: OperationalizationRecord = {
    trait: "subject_b_total",
    delivery_kind: "per_plant_count_aggregate",
    statement: "Every isolated object of the subject on a plant, summed over that plant's images.",
    mechanism: "The detector's objects at the operating point the calibration holdout fixed.",
    measured_subject: "subject_b",
    delivered_phenotypes: ["subject_b_count"],
    delivered_value_keys: ["n_objects"],
    stated_by: "state_trait_operationalization",
    stated_at: "2026-02-01T10:00:00+00:00",
    relayed_note: "",
    agent_client_name: null,
    agent_client_version: null,
    agent_session: null,
    terminal_session: null,
    harness_session: null,
    harness_effort_at_connect: null,
    confirmed_by: null,
    confirmed_at: null,
    identity_from_request: null,
    confirmed_current: false,
    superseded: [],
    delivers: [],
    record_seen: "hash-of-the-displayed-record",
  };

  it("renders a warning banner when the operationalization confirmation lands but its audit line does not", async () => {
    vi.spyOn(resultsApi, "operationalizations").mockResolvedValue({
      records: [COUNT_RECORD],
      statement_fields: ["statement"],
    });
    vi.spyOn(resultsApi, "confirmOperationalization").mockResolvedValue({
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      confirmed_fields: {},
      audit_warning: "committed and unrecorded, do not blind-retry",
    });
    vi.spyOn(resultsApi, "operationalization").mockResolvedValue({
      ...COUNT_RECORD,
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      confirmed_current: true,
    });

    render(<ResultsTab />);
    const row = await screen.findByTestId(
      `operationalization-${COUNT_RECORD.trait}::${COUNT_RECORD.delivery_kind}`,
    );
    fireEvent.click(within(row).getByRole("button", { name: /confirm this record/i }));

    expect(
      await within(row).findByText(/committed and unrecorded, do not blind-retry/i),
    ).toBeInTheDocument();
  });

  it("renders no warning banner when the operationalization confirmation's audit line lands cleanly", async () => {
    vi.spyOn(resultsApi, "operationalizations").mockResolvedValue({
      records: [COUNT_RECORD],
      statement_fields: ["statement"],
    });
    vi.spyOn(resultsApi, "confirmOperationalization").mockResolvedValue({
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      confirmed_fields: {},
      audit_warning: null,
    });
    vi.spyOn(resultsApi, "operationalization").mockResolvedValue({
      ...COUNT_RECORD,
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      confirmed_current: true,
    });

    render(<ResultsTab />);
    const row = await screen.findByTestId(
      `operationalization-${COUNT_RECORD.trait}::${COUNT_RECORD.delivery_kind}`,
    );
    fireEvent.click(within(row).getByRole("button", { name: /confirm this record/i }));

    await waitFor(() =>
      expect(within(row).getByText(/confirmed by user:breeder/i)).toBeInTheDocument(),
    );
    expect(screen.queryByText(/Warning:/i)).not.toBeInTheDocument();
  });

  const TRAIT_SPEC_RECORD: TraitSpecStatementRecord = {
    trait: "subject_a",
    statement_fields: { count_objective: "count_unbiased" },
    rationale: "Breeder said count-unbiased is right for this trait.",
    stated_by: "author_trait_spec",
    stated_at: "2026-02-01T10:00:00+00:00",
    relayed_note: "",
    agent_client_name: null,
    agent_client_version: null,
    agent_session: null,
    terminal_session: null,
    harness_session: null,
    harness_effort_at_connect: null,
    confirmed_by: null,
    confirmed_at: null,
    identity_from_request: null,
    confirmed_current: false,
    record_seen: "hash-of-the-displayed-trait-spec",
  };

  it("renders a warning banner when the trait-spec confirmation lands but its audit line does not", async () => {
    vi.spyOn(resultsApi, "traitSpecStatements").mockResolvedValue({
      records: [TRAIT_SPEC_RECORD],
      unresolved: [],
      statement_fields: ["rationale"],
    });
    vi.spyOn(resultsApi, "confirmTraitSpecStatement").mockResolvedValue({
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      record_seen: "hash-of-the-displayed-trait-spec",
      audit_warning: "committed and unrecorded, do not blind-retry",
    });
    vi.spyOn(resultsApi, "traitSpecStatement").mockResolvedValue({
      ...TRAIT_SPEC_RECORD,
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      confirmed_current: true,
    });

    render(<ResultsTab />);
    const row = await screen.findByTestId(`trait-spec-statement-${TRAIT_SPEC_RECORD.trait}`);
    fireEvent.click(within(row).getByRole("button", { name: /confirm this record/i }));

    expect(
      await within(row).findByText(/committed and unrecorded, do not blind-retry/i),
    ).toBeInTheDocument();
  });

  it("renders no warning banner when the trait-spec confirmation's audit line lands cleanly", async () => {
    vi.spyOn(resultsApi, "traitSpecStatements").mockResolvedValue({
      records: [TRAIT_SPEC_RECORD],
      unresolved: [],
      statement_fields: ["rationale"],
    });
    vi.spyOn(resultsApi, "confirmTraitSpecStatement").mockResolvedValue({
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      record_seen: "hash-of-the-displayed-trait-spec",
      audit_warning: null,
    });
    vi.spyOn(resultsApi, "traitSpecStatement").mockResolvedValue({
      ...TRAIT_SPEC_RECORD,
      confirmed_by: "user:breeder",
      confirmed_at: "2026-02-02T09:00:00+00:00",
      identity_from_request: true,
      confirmed_current: true,
    });

    render(<ResultsTab />);
    const row = await screen.findByTestId(`trait-spec-statement-${TRAIT_SPEC_RECORD.trait}`);
    fireEvent.click(within(row).getByRole("button", { name: /confirm this record/i }));

    await waitFor(() =>
      expect(within(row).getByText(/confirmed by user:breeder/i)).toBeInTheDocument(),
    );
    expect(screen.queryByText(/Warning:/i)).not.toBeInTheDocument();
  });
});

describe("ResultsTab delivery events (read-only)", () => {
  const DELIVERY_EVENT: DeliveryEventRecord = {
    event_id: "abc123",
    trait: "subject_a",
    delivery_kind: "state_crossing_dates",
    door: "results.per_plant_curves",
    output_path: null,
    documents: {
      "C:/data/predictions/baseline/2026-01-01": {
        ok: true,
        claimed: true,
        experiment_id: "exp-1",
        producing_experiment_id: "exp-1",
        checkpoint_sha256: "abc",
        record_digest: "digest-1",
        note: null,
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
    produced_at: "2026-02-03T12:00:00+00:00",
  };

  it("lists what shipped, with real per-bucket verification evidence and no confirm/withdraw controls", async () => {
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [DELIVERY_EVENT] });

    render(<ResultsTab />);
    const row = await screen.findByTestId("delivery-abc123");

    expect(within(row).getByText("subject_a")).toBeInTheDocument();
    expect(within(row).getByText("state_crossing_dates")).toBeInTheDocument();
    expect(within(row).getByText("results.per_plant_curves")).toBeInTheDocument();
    expect(within(row).getByText(/2026-02-03T12:00:00\+00:00/)).toBeInTheDocument();
    expect(within(row).getByText(/no file written/i)).toBeInTheDocument();
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

  it("renders nothing extra when this project has no deliveries yet", async () => {
    vi.spyOn(resultsApi, "deliveryEvents").mockResolvedValue({ records: [] });

    render(<ResultsTab />);
    await waitFor(() => expect(resultsApi.deliveryEvents).toHaveBeenCalled());

    expect(screen.getByText(/nothing has shipped from this project yet/i)).toBeInTheDocument();
  });
});
