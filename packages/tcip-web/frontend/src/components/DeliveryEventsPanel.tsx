/**
 * What has shipped from this project: one row per completed delivery, read-only. A delivery event
 * is a fact recorded after an artifact already shipped under a meaning the breeder already
 * confirmed through the operationalization mechanism, not a statement of its own, so this panel
 * carries no confirm/withdraw affordance and no correction disclosure.
 */

import { Fragment } from "react";

import {
  isCanopySegmentDisclosure,
  isPlantMappingDisclosure,
  isPlantRegistryDisclosure,
  type DeliveryEventRecord,
} from "@/api/inference";

/** One bucket's binding evidence, as `record_delivery_binding_event` (resolution.py) writes it. */
interface DocumentBinding {
  ok?: boolean;
  claimed?: boolean;
  experiment_id?: string | null;
  producing_experiment_id?: string | null;
  checkpoint_sha256?: string | null;
  record_digest?: string | null;
  note?: string | null;
}

function bucketStatusText(binding: DocumentBinding): string {
  if (binding.ok && binding.claimed) return "verified";
  if (binding.claimed) return "claimed, not verified";
  return "no claim";
}

function DeliveryEventRow({ record }: { record: DeliveryEventRecord }) {
  const buckets = Object.entries(record.documents ?? {}) as [string, DocumentBinding][];
  return (
    <li
      className="rounded border border-tcip-border p-3"
      data-testid={`delivery-${record.event_id}`}
    >
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[12px]">{record.trait ?? "unresolved trait"}</span>
        <span className="text-[11px] text-tcip-muted">
          {record.delivery_kind ?? "unresolved delivery kind"}
        </span>
        <span className="ml-auto text-[11px] text-tcip-muted">{record.produced_at}</span>
      </div>
      <dl className="mt-2 grid grid-cols-[170px_1fr] gap-x-3 gap-y-1 text-[11px]">
        <dt className="text-tcip-muted">Door</dt>
        <dd className="font-mono">{record.door}</dd>
        <dt className="text-tcip-muted">Output path</dt>
        <dd>{record.output_path ?? "no file"}</dd>
        {record.acknowledged_by && (
          <>
            <dt className="text-tcip-muted">Acknowledged by</dt>
            <dd>{record.acknowledged_by}</dd>
            <dt className="text-tcip-muted">Reason</dt>
            <dd>{record.acknowledgement_reason}</dd>
          </>
        )}
      </dl>
      {buckets.length > 0 && (
        <div className="mt-2 flex flex-col gap-0.5">
          <div className="text-[11px] text-tcip-muted">Per-bucket evidence</div>
          {buckets.map(([bucket, binding]) => (
            <Fragment key={bucket}>
              <div className="font-mono text-[11px] text-tcip-muted">
                {bucket}: {bucketStatusText(binding)}
              </div>
            </Fragment>
          ))}
        </div>
      )}
      {record.plant_mapping && isCanopySegmentDisclosure(record.plant_mapping) && (
        <div className="mt-2 text-[11px] text-tcip-muted" data-testid="canopy-disclosure">
          {(() => {
            const pm = record.plant_mapping;
            const delivered = pm.segment_ties.length - pm.plants_with_ambiguous_detections.length;
            const registered = delivered + pm.plants_without_segment.length
              + pm.plants_outside_raster.length + pm.plants_with_ambiguous_detections.length;
            return (
              <>
                <div>
                  {`Canopy segments (${pm.plant_registry.name}): ${delivered}/${registered} `
                    + `registry plant(s) delivered, ${pm.segments_without_plant} segment(s) `
                    + "with no plant"}
                </div>
                {pm.plants_without_segment.length > 0 && (
                  <div>{`No segment: ${pm.plants_without_segment.join(", ")}`}</div>
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
        <div className="mt-2 text-[11px] text-tcip-muted">
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
        <div className="mt-2 text-[11px] text-tcip-muted">
          {`Delivered dates ${record.plant_mapping.dates_delivered.join(", ")}: ` +
            `${record.plant_mapping.images_unattributed} attributed to no plant ` +
            `(${record.plant_mapping.plant_attribution}-level attribution)`}
          {record.plant_mapping_resolved_key === null &&
            " (the cited mapping record no longer resolves)"}
          {record.plant_mapping_resolved_key != null &&
            record.plant_mapping_resolved_key !== record.plant_mapping.name &&
            ` (cited record archived as ${record.plant_mapping_resolved_key})`}
        </div>
      )}
      {record.superseded && (
        <div
          className="mt-2 text-[11px] text-tcip-fp"
          data-testid={`superseded-${record.event_id}`}
        >
          {`Superseded: ${record.superseded.reason}` +
            (record.superseded.replacement_event_id
              ? ` (replaced by ${record.superseded.replacement_event_id})`
              : "")}
        </div>
      )}
    </li>
  );
}

export function DeliveryEventsPanel({
  records,
  loadError,
}: {
  records: DeliveryEventRecord[];
  loadError: string | null;
}) {
  return (
    <div className="tcip-panel p-4">
      <div className="tcip-heading mb-1">Delivery events</div>
      <p className="mb-3 text-[11px] text-tcip-muted">
        What has shipped from this project, and the real per-bucket verification evidence the
        delivering door reconciled at the time. A delivery event is a fact, not a statement: there
        is nothing here to confirm.
      </p>
      {loadError ? (
        <div className="text-[11px] text-tcip-fp">{loadError}</div>
      ) : records.length === 0 ? (
        <div className="text-[11px] text-tcip-muted">
          Nothing has shipped from this project yet.
        </div>
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
