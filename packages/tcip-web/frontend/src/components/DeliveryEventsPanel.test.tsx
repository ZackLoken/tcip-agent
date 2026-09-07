import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";

import type { DeliveryEventRecord } from "@/api/inference";
import { DeliveryEventsPanel } from "@/components/DeliveryEventsPanel";

const BASE: DeliveryEventRecord = {
  event_id: "evt-base",
  trait: "subject_a",
  delivery_kind: "state_crossing_dates",
  door: "results.export_csv",
  output_path: "C:/proj/results_export/subject_a_phenology.csv",
  output_sha256: "a".repeat(64),
  acknowledged_by: null,
  acknowledgement_reason: null,
  documents: {},
  produced_at: "2026-02-03T12:00:00+00:00",
  plant_mapping: null,
  superseded: null,
};

describe("DeliveryEventsPanel reconciled validity", () => {
  it("renders one line per recorded document and dimension reconciliation", () => {
    const record: DeliveryEventRecord = {
      ...BASE,
      event_id: "with-reconciliations",
      document_reconciliations: {
        operating_point: {
          validated: "held_out_annotations",
          on_disk_validated: true,
          missing_sidecars: [],
          unvalidated_buckets: [],
          binding_notes: {},
          bindings: {},
          conf: 0.4,
          confs: {},
          per_bucket: {},
        },
        classifier_operating_point: {
          validated: "held_out_annotations",
          on_disk_validated: true,
          missing_sidecars: [],
          unvalidated_buckets: ["C:/data/predictions/baseline/2026-01-08"],
          binding_notes: {},
          bindings: {},
          conf: null,
          confs: {},
          per_bucket: {},
          bound_validated: "false",
          delivery_note: "classifier_operating_point.json at ... floored by the binding",
        },
      },
      dimension_reconciliations: {
        tile_size: {
          operative: false,
          validated: null,
          per_bucket: {},
          unvalidated_buckets: [],
          binding_notes: {},
        },
      },
    };

    render(<DeliveryEventsPanel records={[record]} loadError={null} />);
    const row = screen.getByTestId("delivery-with-reconciliations");

    expect(within(row).getByText("Reconciled validity")).toBeInTheDocument();
    expect(
      within(row).getByText("operating_point: held_out_annotations, 0 unvalidated bucket(s)"),
    ).toBeInTheDocument();
    expect(
      within(row).getByText(
        "classifier_operating_point: held_out_annotations (bound: false), 1 unvalidated bucket(s)",
      ),
    ).toBeInTheDocument();
    expect(
      within(row).getByText("tile_size: not operative, 0 unvalidated bucket(s)"),
    ).toBeInTheDocument();
  });

  it("shows no reconciled-validity section for a record whose reconcilers all ran empty", () => {
    const record: DeliveryEventRecord = {
      ...BASE,
      event_id: "empty-reconciliations",
      document_reconciliations: {},
      dimension_reconciliations: {},
    };

    render(<DeliveryEventsPanel records={[record]} loadError={null} />);
    const row = screen.getByTestId("delivery-empty-reconciliations");

    expect(within(row).queryByText("Reconciled validity")).not.toBeInTheDocument();
  });

  it("shows no reconciled-validity section for a record predating the fields", () => {
    const record: DeliveryEventRecord = { ...BASE, event_id: "predates-reconciliations" };

    render(<DeliveryEventsPanel records={[record]} loadError={null} />);
    const row = screen.getByTestId("delivery-predates-reconciliations");

    expect(within(row).queryByText("Reconciled validity")).not.toBeInTheDocument();
  });
});
