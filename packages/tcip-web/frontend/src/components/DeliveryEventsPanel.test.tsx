import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { ANSWERS, CANOPY_SEGMENTS, UNANSWERED, UNVALIDATED } from "@/api/types.generated";
import { DeliveryEventsPanel } from "@/components/DeliveryEventsPanel";
import { acknowledgmentOf, DELIVERY_RECORDS, disclosureOf } from "@/test/deliveryRecords";

describe("DeliveryEventsPanel bucket findings", () => {
  it("names the assessment behind a validated bucket", () => {
    const served = DELIVERY_RECORDS.validated;
    const record = served.event;
    render(<DeliveryEventsPanel records={[served]} loadError={null} />);
    const row = screen.getByTestId(`delivery-${record.event_id}`);

    for (const bucket of record.buckets) {
      expect(
        within(row).getByText(`${bucket.bucket}: validated by assessment ${bucket.assessment_id}`),
      ).toBeInTheDocument();
    }
    expect(within(row).queryByText("Acknowledged by")).not.toBeInTheDocument();
  });

  it("names the reason behind an unvalidated bucket and the breeder who acknowledged it", () => {
    const served = DELIVERY_RECORDS.acknowledged_mapping;
    const record = served.event;
    render(<DeliveryEventsPanel records={[served]} loadError={null} />);
    const row = screen.getByTestId(`delivery-${record.event_id}`);

    for (const bucket of record.buckets) {
      if (bucket.kind !== UNVALIDATED) throw new Error("the produced bucket was validated");
      expect(
        within(row).getByText(`${bucket.bucket}: not validated (${bucket.reason})`),
      ).toBeInTheDocument();
    }
    const acknowledgment = acknowledgmentOf(served);
    expect(within(row).getByText(acknowledgment.acknowledged_by)).toBeInTheDocument();
    expect(within(row).getByText(acknowledgment.reason)).toBeInTheDocument();
  });

  it("names the scale assessment that answered", () => {
    const served = DELIVERY_RECORDS.scale_answered;
    const { scale } = served.event;
    if (scale?.kind !== ANSWERS) throw new Error("the produced scale read did not answer");
    render(<DeliveryEventsPanel records={[served]} loadError={null} />);
    const row = screen.getByTestId(`delivery-${served.event.event_id}`);

    expect(within(row).getByTestId("scale-finding")).toHaveTextContent(
      `scale assessment ${scale.assessment_id}`,
    );
  });

  it("names the reason a scale assessment answered for nothing", () => {
    const served = DELIVERY_RECORDS.scale_unanswered;
    const { scale } = served.event;
    if (scale?.kind !== UNANSWERED) throw new Error("the produced scale read answered");
    render(<DeliveryEventsPanel records={[served]} loadError={null} />);
    const row = screen.getByTestId(`delivery-${served.event.event_id}`);

    expect(within(row).getByTestId("scale-finding")).toHaveTextContent(
      `scale assessment ${scale.assessment_id} not validated (${scale.reason})`,
    );
  });

  it("renders no scale line for a delivery that named none", () => {
    const served = DELIVERY_RECORDS.validated;
    expect(served.event.scale).toBeNull();
    const { container } = render(<DeliveryEventsPanel records={[served]} loadError={null} />);

    expect(within(container).queryByTestId("scale-finding")).not.toBeInTheDocument();
  });

  it("counts the plants a canopy delivery delivered from its recorded population", () => {
    const served = DELIVERY_RECORDS.canopy;
    const record = served.event;
    const pm = disclosureOf(served, (d) => (d.kind === CANOPY_SEGMENTS ? d : null));
    render(<DeliveryEventsPanel records={[served]} loadError={null} />);

    expect(
      within(screen.getByTestId("canopy-disclosure")).getByText(
        `Canopy segments (${pm.plant_registry.name}): ${record.population.length} registry ` +
          `plant(s) delivered, ${pm.segments_without_plant} segment(s) with no plant`,
      ),
    ).toBeInTheDocument();
  });
});
