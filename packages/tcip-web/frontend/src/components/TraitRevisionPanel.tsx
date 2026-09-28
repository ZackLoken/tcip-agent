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
import { DisclosureChevron } from "@/components/CollapsibleSection";
import { useEditableAgentRequest } from "@/hooks/useEditableAgentRequest";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { useStore } from "@/store";

function valueText(value: unknown): string {
  if (value === null || value === undefined) return UNSET_GLYPH;
  if (Array.isArray(value)) return value.length > 0 ? value.join(", ") : "none";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function revisionStateText(revision: TraitRevision): string {
  if (revision.withdrawn_at) {
    return `Confirmed by ${revision.confirmed_by} on ${revision.confirmed_at}, withdrawn by ${revision.withdrawn_by} on ${revision.withdrawn_at}`;
  }
  if (revision.confirmed_at)
    return `Confirmed by ${revision.confirmed_by} on ${revision.confirmed_at}`;
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
  const { operationalizations, ...fields } = revision.entry;
  return (
    <>
      <dl className="mt-2 grid grid-cols-[190px_1fr] gap-x-3 gap-y-1 text-[11px]">
        {Object.entries(fields).map(([field, value]) => (
          <Fragment key={field}>
            <dt className="text-tcip-muted">{field}</dt>
            <dd>{valueText(value)}</dd>
          </Fragment>
        ))}
      </dl>
      {Object.entries(operationalizations).map(([kind, op]) => (
        <div
          key={kind}
          className="mt-2 rounded border border-tcip-border p-2 text-[11px]"
          data-testid={`operationalization-${kind}`}
        >
          <div className="font-mono">{kind}</div>
          <p className="mt-1">{op.statement}</p>
          <p className="mt-1 text-tcip-muted">Decided by: {op.mechanism}</p>
          <p className="text-tcip-muted">Measured subject: {op.measured_subject}</p>
          {op.delivered_phenotypes.map((p) => (
            <p key={p} className="text-tcip-muted">
              {`${p}: ${definitions[p] ?? "not defined in this project's crop vocabulary"}`}
            </p>
          ))}
          {op.delivered_value_keys.length > 0 && (
            <p className="text-tcip-muted">Value keys: {op.delivered_value_keys.join(", ")}</p>
          )}
        </div>
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
    <li className="rounded border border-tcip-border p-3" data-testid={`trait-${record.trait}`}>
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[12px]">{record.trait}</span>
        <span
          className={`ml-auto text-[11px] ${record.latest_confirmed === null ? "text-tcip-fp" : "text-tcip-muted"}`}
        >
          {record.latest_confirmed === null
            ? "No revision is confirmed; nothing delivers under this trait"
            : `Deliveries read revision ${record.latest_confirmed}`}
        </span>
      </div>
      <ol
        className="mt-2 flex flex-wrap gap-1 text-[11px]"
        aria-label={`${record.trait} revisions`}
      >
        {record.revisions.map((r) => (
          <li key={r.number}>
            <button
              className={
                r.number === shown.number ? "tcip-btn-primary text-[11px]" : "tcip-btn text-[11px]"
              }
              aria-pressed={r.number === shown.number}
              onClick={() => setShownNumber(r.number)}
            >
              {`Revision ${r.number}`}
            </button>
          </li>
        ))}
      </ol>
      <div className="mt-2 text-[11px]">
        <div className={shown.confirmed ? "text-tcip-muted" : "text-tcip-fp"}>
          {revisionStateText(shown)}
        </div>
        {shown.confirmed_at && (
          <div className="text-tcip-muted">
            {shown.identity_from_request
              ? "That name came with the confirming request."
              : "That name came from the backend's own environment, not from the confirming request."}
          </div>
        )}
        <div className="mt-1 text-tcip-muted">
          {`Proposed ${shown.proposed_at}. Why: ${shown.rationale}`}
          {shown.relayed_note && ` Relayed from you: ${shown.relayed_note}`}
        </div>
      </div>
      <RevisionEntry revision={shown} definitions={definitions} />
      {auditWarning && (
        <div className="mt-2 text-[11px] text-tcip-warn">Warning: {auditWarning}</div>
      )}
      {note && <div className="mt-2 text-[11px] text-tcip-fp">{note}</div>}
      <div className="mt-2 flex items-center gap-2">
        {!shown.confirmed_at && (
          <button
            className="tcip-btn-primary text-[11px]"
            onClick={() => onDecide(shown, true)}
            disabled={pending}
          >
            {pending ? "Confirming…" : `Confirm revision ${shown.number}`}
          </button>
        )}
        {shown.confirmed && (
          <button
            className="tcip-btn text-[11px]"
            onClick={() => onDecide(shown, false)}
            disabled={pending}
          >
            {pending ? "Withdrawing…" : "Withdraw this confirmation"}
          </button>
        )}
        <button
          className="tcip-btn text-[11px]"
          aria-expanded={correcting}
          onClick={() => setCorrecting((open) => !open)}
        >
          <DisclosureChevron open={correcting} />
          Send a correction to the agent
        </button>
      </div>
      {correcting && (
        <div className="mt-2 flex flex-col gap-1">
          <textarea
            className="tcip-input h-20 w-full resize-none text-[11px] leading-4"
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            spellCheck={true}
            aria-label={`Correction for ${record.trait}`}
          />
          <button
            className="tcip-btn self-start text-[11px]"
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

export function TraitRevisionPanel({ projectRoot }: { projectRoot: string }) {
  const [listing, setListing] = useState<TraitsListing | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [pendingTrait, setPendingTrait] = useState<string | null>(null);
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [auditWarnings, setAuditWarnings] = useState<Record<string, string | null>>({});

  const reload = useCallback(async () => {
    try {
      setListing(await resultsApi.traits(projectRoot));
      setLoadError(null);
    } catch (e) {
      setListing(null);
      setLoadError(
        `Could not load this project's traits: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }, [projectRoot]);

  useEffect(() => {
    void reload();
  }, [reload]);

  async function decide(trait: string, revision: TraitRevision, confirmed: boolean) {
    setPendingTrait(trait);
    setNotes((prev) => ({ ...prev, [trait]: "" }));
    setAuditWarnings((prev) => ({ ...prev, [trait]: null }));
    try {
      const res = await resultsApi.confirmTraitRevision({
        project_root: projectRoot,
        trait,
        revision: revision.number,
        entry_sha256: revision.entry_sha256,
        user: useStore.getState().user || undefined,
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
    <div className="tcip-panel p-4">
      <div className="tcip-heading mb-1">What each trait's delivered numbers mean</div>
      <p className="mb-3 text-[11px] text-tcip-muted">
        The agent proposes each trait's entry, from what it measures to what each delivered number
        means, as a numbered revision. A delivery reads the latest revision you confirmed. Read the
        revision shown, then confirm it or send the agent a correction. A confirmation you gave
        stands until you withdraw it.
      </p>
      {loadError && <div className="mb-3 text-[11px] text-tcip-fp">{loadError}</div>}
      {listing && listing.unreadable.length > 0 && (
        <div className="mb-3 text-[11px] text-tcip-fp">
          <div>These traits' records will not read:</div>
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
        <div className="text-[11px] text-tcip-muted">
          No trait is proposed for this project yet. The agent proposes one before anything can
          deliver under it.
        </div>
      ) : (
        <ul className="flex flex-col gap-2">
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
