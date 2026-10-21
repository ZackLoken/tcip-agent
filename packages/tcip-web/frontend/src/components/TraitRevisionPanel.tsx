/**
 * The breeder's confirmation surface for trait revisions: every trait's revision list, the entry
 * of the revision shown, and confirm or withdraw on that revision. The agent proposes a revision;
 * nothing here edits an entry. A confirmation posts the shown revision's own entry hash, so a
 * click lands on the entry that was displayed.
 */

import { Fragment, useCallback, useEffect, useState } from "react";

import { StructuredRefusalError } from "@/api/http";
import {
  resultsApi,
  type ServedTraitRecord,
  type ServedTraitRevision as TraitRevision,
  type TraitsListing,
} from "@/api/inference";
import { useEditableAgentRequest } from "@/hooks/useEditableAgentRequest";
import { useStore } from "@/store";

function valueText(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

/** Whether an entry field holds a value worth a row: null, an empty string and an empty list do
 *  not. */
function holdsValue(value: unknown): boolean {
  if (value === null || value === undefined || value === "") return false;
  return !(Array.isArray(value) && value.length === 0);
}

const words = (field: string): string => field.replaceAll("_", " ");

const dayOf = (timestamp: string): string => timestamp.slice(0, 10);

function revisionStateText(revision: TraitRevision): string {
  if (revision.withdrawn_at) {
    return `Confirmed by ${revision.confirmed_by}, withdrawn by ${revision.withdrawn_by} on ${dayOf(revision.withdrawn_at)}.`;
  }
  if (revision.confirmed_at) {
    return `Confirmed by ${revision.confirmed_by} on ${dayOf(revision.confirmed_at)}.`;
  }
  return "Not confirmed";
}

// The agent proposes, the breeder confirms; a correction is a message that proposes no meaning.
function correctionRequest(trait: string, revision: TraitRevision): string {
  return (
    `Revision ${revision.number} of the "${trait}" trait is not what I mean. Why the agent ` +
    `proposed it: ${revision.rationale} What should change: `
  );
}

function RevisionEntry({
  revision,
  definitions,
}: {
  revision: TraitRevision;
  definitions: Record<string, string>;
}) {
  const { operationalizations, name: _name, ...fields } = revision.entry;
  const rows = Object.entries(fields).filter(([, value]) => holdsValue(value));
  return (
    <>
      {rows.length > 0 && (
        <dl className="mt-3 grid grid-cols-[180px_1fr] gap-x-4 gap-y-1">
          {rows.map(([field, value]) => (
            <Fragment key={field}>
              <dt className="text-tcip-muted">{words(field)}</dt>
              <dd>{valueText(value)}</dd>
            </Fragment>
          ))}
        </dl>
      )}
      {Object.entries(operationalizations).map(([kind, op]) => (
        <dl
          key={kind}
          className="mt-3 grid grid-cols-[180px_1fr] gap-x-4 gap-y-1"
          data-testid={`operationalization-${kind}`}
        >
          <dt className="text-tcip-muted">{words(kind)}</dt>
          <dd>{op.statement}</dd>
          <dt className="text-tcip-muted">Decided by</dt>
          <dd>{op.mechanism}</dd>
          <dt className="text-tcip-muted">Subject</dt>
          <dd>{op.measured_subject}</dd>
          {op.delivered_phenotypes.map((p) => (
            <Fragment key={p}>
              <dt className="text-tcip-muted">Delivers</dt>
              <dd>{definitions[p] ? `${p}: ${definitions[p]}` : p}</dd>
            </Fragment>
          ))}
          {op.delivered_value_keys.length > 0 && (
            <>
              <dt className="text-tcip-muted">Value keys</dt>
              <dd>{op.delivered_value_keys.join(", ")}</dd>
            </>
          )}
        </dl>
      ))}
    </>
  );
}

function TraitRow({
  record,
  definitions,
  pending,
  note,
  auditWarning,
  onDecide,
}: {
  record: ServedTraitRecord;
  definitions: Record<string, string>;
  pending: boolean;
  note: string | null;
  auditWarning: string | null;
  onDecide: (revision: TraitRevision, confirmed: boolean) => void;
}) {
  const latest = record.revisions[record.revisions.length - 1];
  const [shownNumber, setShownNumber] = useState(latest.number);
  const shown = record.revisions.find((r) => r.number === shownNumber) ?? latest;
  const { request, setRequest } = useEditableAgentRequest(correctionRequest(record.trait, shown));
  const [correcting, setCorrecting] = useState(false);

  return (
    <li className="py-4 first:pt-0 last:pb-0" data-testid={`trait-${record.trait}`}>
      <div className="flex items-baseline gap-3">
        <span className="font-medium">{record.trait}</span>
        <span className="text-[12px] text-tcip-muted">
          {record.latest_confirmed === null
            ? "Not confirmed"
            : `Confirmed, revision ${record.latest_confirmed}`}
        </span>
      </div>
      {record.revisions.length > 1 && (
        <ol className="mt-2 flex flex-wrap gap-1" aria-label={`${record.trait} revisions`}>
          {record.revisions.map((r) => (
            <li key={r.number}>
              <button
                className={r.number === shown.number ? "tcip-btn-primary h-7" : "tcip-btn h-7"}
                aria-pressed={r.number === shown.number}
                onClick={() => setShownNumber(r.number)}
              >
                {`Revision ${r.number}`}
              </button>
            </li>
          ))}
        </ol>
      )}
      <div className="mt-2 text-tcip-muted">
        {`Revision ${shown.number}, proposed ${dayOf(shown.proposed_at)}. `}
        <span className={shown.confirmed ? undefined : "tcip-problem"}>
          {revisionStateText(shown)}
        </span>
      </div>
      <p className="mt-2 max-w-[70ch]">{shown.rationale}</p>
      {shown.relayed_note && (
        <p className="mt-1 max-w-[70ch] text-tcip-muted">{`From you: ${shown.relayed_note}`}</p>
      )}
      <RevisionEntry revision={shown} definitions={definitions} />
      {auditWarning && <div className="mt-2 text-tcip-warn">{auditWarning}</div>}
      {note && <div className="mt-2 tcip-problem">{note}</div>}
      <div className="mt-3 flex items-center gap-2">
        {!shown.confirmed_at && (
          <button
            className="tcip-btn-primary"
            onClick={() => onDecide(shown, true)}
            disabled={pending}
          >
            {pending ? "Confirming…" : "Confirm"}
          </button>
        )}
        {shown.confirmed && (
          <button className="tcip-btn" onClick={() => onDecide(shown, false)} disabled={pending}>
            {pending ? "Withdrawing…" : "Withdraw"}
          </button>
        )}
        <button
          className="tcip-btn"
          aria-expanded={correcting}
          onClick={() => setCorrecting((open) => !open)}
        >
          Correct
        </button>
      </div>
      {correcting && (
        <div className="mt-2 flex max-w-[70ch] flex-col gap-2">
          <textarea
            className="tcip-input h-24 w-full resize-none py-2 leading-5"
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            spellCheck={true}
            aria-label={`Correction for ${record.trait}`}
          />
          <button
            className="tcip-btn self-start"
            onClick={() => useStore.getState().sendToAgentTerminal(request)}
            disabled={!request.trim()}
          >
            Send to the agent
          </button>
        </div>
      )}
    </li>
  );
}

/** The open project's trait revisions; its parent keys it by the open project so a switch
 *  remounts it. */
export function TraitRevisionPanel() {
  const [listing, setListing] = useState<TraitsListing | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [pendingTrait, setPendingTrait] = useState<string | null>(null);
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [auditWarnings, setAuditWarnings] = useState<Record<string, string | null>>({});

  const reload = useCallback(async () => {
    try {
      setListing(await resultsApi.traits());
      setLoadError(null);
    } catch (e) {
      setListing(null);
      setLoadError(`Could not load traits: ${e instanceof Error ? e.message : String(e)}`);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  async function decide(trait: string, revision: TraitRevision, confirmed: boolean) {
    setPendingTrait(trait);
    setNotes((prev) => ({ ...prev, [trait]: "" }));
    setAuditWarnings((prev) => ({ ...prev, [trait]: null }));
    try {
      const res = await resultsApi.confirmTraitRevision({
        trait,
        revision: revision.number,
        entry_sha256: revision.entry_sha256,
        user: useStore.getState().user,
        confirmed,
      });
      setAuditWarnings((prev) => ({ ...prev, [trait]: res.audit_warning }));
    } catch (e) {
      const moved = e instanceof StructuredRefusalError && e.status === 409;
      setNotes((prev) => ({
        ...prev,
        [trait]: moved
          ? "This revision is not the one shown. Read what is on file above, then decide on that."
          : `${confirmed ? "Could not confirm" : "Could not withdraw"}: ${
              e instanceof Error ? e.message : String(e)
            }`,
      }));
    } finally {
      setPendingTrait(null);
      await reload();
    }
  }

  return (
    <div className="tcip-panel p-5">
      <div className="tcip-heading mb-3">Traits</div>
      {loadError && <div className="mb-3 tcip-problem">{loadError}</div>}
      {listing && listing.unreadable.length > 0 && (
        <div className="mb-3 tcip-problem">
          <div>These trait records will not read:</div>
          <ul className="mt-1 list-disc pl-4">
            {listing.unreadable.map((u) => (
              <li key={u.trait}>
                {u.trait}: {u.reason}
              </li>
            ))}
          </ul>
        </div>
      )}
      {listing && listing.traits.length === 0 ? (
        <div className="text-tcip-muted">No trait proposed.</div>
      ) : (
        <ul className="flex flex-col divide-y divide-tcip-border">
          {(listing?.traits ?? []).map((record) => (
            <TraitRow
              key={`${record.trait}:${record.revisions.length}`}
              record={record}
              definitions={listing?.definitions ?? {}}
              pending={pendingTrait === record.trait}
              note={notes[record.trait] || null}
              auditWarning={auditWarnings[record.trait] || null}
              onDecide={(revision, confirmed) => void decide(record.trait, revision, confirmed)}
            />
          ))}
        </ul>
      )}
    </div>
  );
}
