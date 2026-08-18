/**
 * What has shipped from this project: one row per completed delivery, read-only. A delivery event
 * is a fact recorded after an artifact already shipped under a meaning the breeder already
 * confirmed through the operationalization mechanism, not a statement of its own, so this panel
 * carries no confirm/withdraw affordance and no correction disclosure.
 */

import { Fragment } from "react";

import type { DeliveryEventRecord } from "@/api/inference";

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
        <dd>{record.output_path ?? "none (rendered on screen, no file written)"}</dd>
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
      {loadError && <div className="mb-3 text-[11px] text-tcip-fp">{loadError}</div>}
      {records.length === 0 ? (
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
