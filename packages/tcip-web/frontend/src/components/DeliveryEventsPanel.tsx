/**
 * What has shipped from this project: one row per completed delivery, read-only. A delivery event
 * is a fact recorded after an artifact already shipped under the trait revision the breeder
 * confirmed, which it names, so this panel carries no confirm/withdraw affordance.
 */

import {
  isCanopySegmentDisclosure,
  isPlantMappingDisclosure,
  isPlantRegistryDisclosure,
} from "@/api/inference";
import type { ServedDeliveryEvent } from "@/api/types.generated";

function DeliveryEventRow({ record }: { record: ServedDeliveryEvent }) {
  return (
    <li
      className="rounded border border-tcip-border p-3"
      data-testid={`delivery-${record.event_id}`}
    >
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[13px]">{record.trait}</span>
        <span className="text-[13px] text-tcip-muted">
          {`revision ${record.trait_revision}, ${record.delivery_kind}`}
        </span>
        <span className="ml-auto text-[13px] text-tcip-muted">{record.produced_at}</span>
      </div>
      <dl className="mt-2 grid grid-cols-[170px_1fr] gap-x-3 gap-y-1 text-[13px]">
        <dt className="text-tcip-muted">Door</dt>
        <dd className="font-mono">{record.door}</dd>
        <dt className="text-tcip-muted">Output path</dt>
        <dd>{record.output_path}</dd>
        <dt className="text-tcip-muted">Validated</dt>
        <dd>{record.validated ? "yes" : "no"}</dd>
        {record.acknowledgment && (
          <>
            <dt className="text-tcip-muted">Acknowledged by</dt>
            <dd>{record.acknowledgment.acknowledged_by}</dd>
            <dt className="text-tcip-muted">Reason</dt>
            <dd>{record.acknowledgment.reason}</dd>
          </>
        )}
      </dl>
      <div className="mt-2 flex flex-col gap-0.5">
        <div className="text-[13px] text-tcip-muted">Buckets delivered</div>
        {record.buckets.map((bucket) => (
          <div
            key={`${bucket.dataset_root}\0${bucket.bucket}`}
            className="font-mono text-[13px] text-tcip-muted"
          >
            {`${bucket.bucket}: ` +
              (bucket.validated
                ? `validated by assessment ${bucket.assessment_id}`
                : `not validated (${bucket.reason})`)}
          </div>
        ))}
      </div>
      {record.plant_mapping && isCanopySegmentDisclosure(record.plant_mapping) && (
        <div className="mt-2 text-[13px] text-tcip-muted" data-testid="canopy-disclosure">
          {(() => {
            const pm = record.plant_mapping;
            return (
              <>
                <div>
                  {`Canopy segments (${pm.plant_registry.name}): ${record.population.length} ` +
                    `registry plant(s) delivered, ${pm.segments_without_plant} segment(s) ` +
                    "with no plant"}
                </div>
                {pm.plants_without_segment.length > 0 && (
                  <div>{`No segment: ${pm.plants_without_segment.join(", ")}`}</div>
                )}
                {pm.plants_within_position_error.length > 0 && (
                  <div>{`Within the position error: ${pm.plants_within_position_error.join(", ")}`}</div>
                )}
                {pm.plants_outside_raster.length > 0 && (
                  <div>{`Outside the raster: ${pm.plants_outside_raster.join(", ")}`}</div>
                )}
                {pm.plants_with_ambiguous_detections.length > 0 && (
                  <div>{`Ambiguous detection: ${pm.plants_with_ambiguous_detections.join(", ")}`}</div>
                )}
              </>
            );
          })()}
        </div>
      )}
      {record.plant_mapping && isPlantRegistryDisclosure(record.plant_mapping) && (
        <div className="mt-2 text-[13px] text-tcip-muted">
          <div>
            {`Plant registry ${record.plant_mapping.plant_registry.name}: ` +
              `${record.plant_mapping.detections_unattributed} detection(s) attributed to no ` +
              `plant on the delivered raster (${record.plant_mapping.plant_attribution}-level ` +
              "attribution)"}
          </div>
          {record.plant_mapping.plants_outside_raster.length > 0 && (
            <div>
              {`Outside the raster: ${record.plant_mapping.plants_outside_raster.join(", ")}`}
            </div>
          )}
        </div>
      )}
      {record.plant_mapping && isPlantMappingDisclosure(record.plant_mapping) && (
        <div className="mt-2 text-[13px] text-tcip-muted">
          {`Delivered dates ${record.plant_mapping.dates_delivered.join(", ")}: ` +
            `${record.plant_mapping.images_unattributed} attributed to no plant ` +
            `(${record.plant_mapping.plant_attribution}-level attribution)`}
          {record.plant_mapping.resolved_key === null &&
            " (the cited mapping record no longer resolves)"}
          {record.plant_mapping.resolved_key !== null &&
            record.plant_mapping.resolved_key !== record.plant_mapping.name &&
            ` (cited record archived as ${record.plant_mapping.resolved_key})`}
        </div>
      )}
    </li>
  );
}

export function DeliveryEventsPanel({
  records,
  loadError,
}: {
  records: ServedDeliveryEvent[];
  loadError: string | null;
}) {
  return (
    <div className="tcip-panel p-4">
      <h2 className="tcip-heading mb-3">Deliveries</h2>
      {loadError ? (
        <div className="tcip-problem">{loadError}</div>
      ) : records.length === 0 ? (
        <div className="text-tcip-muted">No deliveries.</div>
      ) : (
        <ul className="flex flex-col gap-2">
          {records.map((record) => (
            <DeliveryEventRow key={record.event_id} record={record} />
          ))}
        </ul>
      )}
    </div>
  );
}
