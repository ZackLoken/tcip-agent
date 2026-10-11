import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { isCanopySegmentDisclosure } from "@/api/inference";
import { DeliveryEventsPanel } from "@/components/DeliveryEventsPanel";
import { DELIVERY_RECORDS } from "@/test/deliveryRecords";

describe("DeliveryEventsPanel bucket findings", () => {
  it("names the assessment behind a validated bucket", () => {
    const record = DELIVERY_RECORDS.validated;
    render(<DeliveryEventsPanel records={[record]} loadError={null} />);
    const row = screen.getByTestId(`delivery-${record.event_id}`);

    for (const bucket of record.buckets) {
      expect(
        within(row).getByText(`${bucket.bucket}: validated by assessment ${bucket.assessment_id}`),
      ).toBeInTheDocument();
    }
    expect(within(row).queryByText("Acknowledged by")).not.toBeInTheDocument();
  });

  it("names the reason behind an unvalidated bucket and the breeder who acknowledged it", () => {
    const record = DELIVERY_RECORDS.acknowledged_mapping;
    if (!record.acknowledgment) throw new Error("the fixture delivery was not acknowledged");
    render(<DeliveryEventsPanel records={[record]} loadError={null} />);
    const row = screen.getByTestId(`delivery-${record.event_id}`);

    for (const bucket of record.buckets) {
      expect(
        within(row).getByText(`${bucket.bucket}: not validated (${bucket.reason})`),
      ).toBeInTheDocument();
    }
    expect(within(row).getByText(record.acknowledgment.acknowledged_by)).toBeInTheDocument();
    expect(within(row).getByText(record.acknowledgment.reason)).toBeInTheDocument();
  });

  it("counts the plants a canopy delivery delivered from its recorded population", () => {
    const record = DELIVERY_RECORDS.canopy;
    const pm = record.plant_mapping;
    if (!pm || !isCanopySegmentDisclosure(pm)) throw new Error("the fixture names no segments");
    render(<DeliveryEventsPanel records={[record]} loadError={null} />);

    expect(
      within(screen.getByTestId("canopy-disclosure")).getByText(
        `Canopy segments (${pm.plant_registry.name}): ${record.population.length} registry ` +
          `plant(s) delivered, ${pm.segments_without_plant} segment(s) with no plant`,
      ),
    ).toBeInTheDocument();
  });
});
