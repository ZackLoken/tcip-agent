import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";

import type { DeliveryEventRecord } from "@/api/inference";
import { DeliveryEventsPanel } from "@/components/DeliveryEventsPanel";

const BASE: DeliveryEventRecord = {
  event_id: "evt-base",
  door: "results.export_csv",
  delivery_kind: "state_crossing_dates",
  trait: "subject_a",
  trait_revision: 1,
  trait_revision_sha256: "b".repeat(64),
  output_path: "C:/proj/results_export/subject_a_phenology.csv",
  output_sha256: "a".repeat(64),
  producer: { checkpoint_sha256: "c".repeat(64), experiment_id: "exp-1" },
  buckets: [
    {
      path: "C:/data/predictions/baseline/2026-01-01",
      date: "2026-01-01",
      assessment_id: "assessment-1",
      validated: true,
      reason: null,
    },
  ],
  scale_assessment_id: null,
  validated: true,
  acknowledgment: null,
  population: ["P-001"],
  require_all_dates_complete: null,
  plant_mapping: null,
  produced_at: "2026-02-03T12:00:00+00:00",
};

describe("DeliveryEventsPanel bucket findings", () => {
  it("names the assessment behind a validated bucket and the reason behind one that is not", () => {
    const record: DeliveryEventRecord = {
      ...BASE,
      event_id: "with-buckets",
      validated: false,
      acknowledgment: {
        acknowledgment_id: "ack-1",
        acknowledged_by: "user:breeder",
        reason: "shipping before assessing",
        result_sha256: "0".repeat(64),
        recorded_at: "2026-02-03T11:59:00+00:00",
      },
      buckets: [
        {
          path: "C:/data/predictions/baseline/2026-01-01",
          date: "2026-01-01",
          assessment_id: "assessment-1",
          validated: true,
          reason: null,
        },
        {
          path: "C:/data/predictions/baseline/2026-01-08",
          date: "2026-01-08",
          assessment_id: null,
          validated: false,
          reason: "no assessment answers for it",
        },
      ],
    };

    render(<DeliveryEventsPanel records={[record]} loadError={null} />);
    const row = screen.getByTestId("delivery-with-buckets");

    expect(
      within(row).getByText(
        "C:/data/predictions/baseline/2026-01-01: validated by assessment assessment-1",
      ),
    ).toBeInTheDocument();
    expect(
      within(row).getByText(
        "C:/data/predictions/baseline/2026-01-08: not validated (no assessment answers for it)",
      ),
    ).toBeInTheDocument();
    expect(within(row).getByText("user:breeder")).toBeInTheDocument();
    expect(within(row).getByText("shipping before assessing")).toBeInTheDocument();
  });

  it("shows no acknowledgment for a validated record", () => {
    render(<DeliveryEventsPanel records={[BASE]} loadError={null} />);
    const row = screen.getByTestId("delivery-evt-base");

    expect(within(row).queryByText("Acknowledged by")).not.toBeInTheDocument();
  });
});
