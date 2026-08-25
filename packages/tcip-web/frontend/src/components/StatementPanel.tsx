/**
 * The generalized confirmation surface for a statement record: the agent states, the breeder
 * confirms or withdraws, and a moved/superseded record re-renders what is on file rather than what
 * was last shown. Extends the pattern that shipped first for operationalization records
 * (`OperationalizationPanel`/`OperationalizationRow` in `ResultsTab.tsx`) to any statement kind
 * sharing the same shape: a trait-spec authoring statement today, and any future kind without a
 * new panel needing to be built for it.
 *
 * Rendering that genuinely differs between kinds (the operationalization row's "delivers" line and
 * its superseded-fields block, which a trait-spec statement has no equivalent of) is passed in as
 * render props rather than switched on internally, so this component carries no vocabulary of any
 * one statement kind.
 */

import { Fragment, type ReactNode, useState } from "react";

import { DisclosureChevron } from "@/components/CollapsibleSection";
import { useEditableAgentRequest } from "@/hooks/useEditableAgentRequest";
import { useStore } from "@/store";
import { fieldValueText, STATEMENT_FIELD_LABELS } from "@/lib/statementFields";

/** The fields every statement record carries, regardless of kind. */
export interface BaseStatementRecord {
  trait: string;
  stated_by: string | null;
  stated_at: string | null;
  relayed_note: string | null;
  confirmed_by: string | null;
  confirmed_at: string | null;
  identity_from_request: boolean | null;
  confirmed_current: boolean;
  record_seen: string;
}

/** One flat `field: value` row, read generically off the served `statement_fields` list rather
 *  than a hand-picked subset, so a field the server drops leaves the row and one it adds appears. */
export function flatStatementFields<R extends Record<string, unknown>>(
  record: R,
  fields: string[],
): ReactNode {
  return (
    <dl className="mt-2 grid grid-cols-[170px_1fr] gap-x-3 gap-y-1 text-[11px]">
      {fields.map((field) => (
        <Fragment key={field}>
          <dt className="text-tcip-muted">{STATEMENT_FIELD_LABELS[field] ?? field}</dt>
          <dd>{fieldValueText(record[field])}</dd>
        </Fragment>
      ))}
    </dl>
  );
}

function StatementRow<R extends BaseStatementRecord>({
  record,
  kindLabel,
  testId,
  refused,
  confirming,
  withdrawing,
  note,
  auditWarning,
  headerExtra,
  fieldsBody,
  supersededBlock,
  registryProblemBlock,
  correctionSeed,
  correctionAriaLabel,
  onConfirm,
  onWithdraw,
}: {
  record: R;
  kindLabel: string;
  testId: string;
  refused: boolean;
  confirming: boolean;
  withdrawing: boolean;
  note: string | null;
  auditWarning: string | null;
  headerExtra: ReactNode;
  fieldsBody: ReactNode;
  supersededBlock: ReactNode;
  registryProblemBlock: ReactNode;
  correctionSeed: string;
  correctionAriaLabel: string;
  onConfirm: () => void;
  onWithdraw: () => void;
}) {
  const { request, setRequest } = useEditableAgentRequest(correctionSeed);
  const [correcting, setCorrecting] = useState(false);

  return (
    <li
      className={`rounded border p-3 ${refused ? "border-tcip-fp" : "border-tcip-border"}`}
      data-testid={testId}
    >
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[12px]">{record.trait}</span>
        {kindLabel && <span className="text-[11px] text-tcip-muted">{kindLabel}</span>}
        <span
          className={`ml-auto text-[11px] ${record.confirmed_current ? "text-tcip-muted" : "text-tcip-fp"}`}
        >
          {record.confirmed_by && record.confirmed_current
            ? `Confirmed by ${record.confirmed_by} on ${record.confirmed_at}`
            : record.confirmed_by
              ? `Confirmed by ${record.confirmed_by} on ${record.confirmed_at}, no longer current`
              : "Not confirmed"}
        </span>
      </div>
      {record.confirmed_by && (
        <div className="text-[11px] text-tcip-muted">
          {record.identity_from_request
            ? "That name came with the confirming request."
            : "That name came from the backend's own environment, not from the confirming request."}
        </div>
      )}
      {headerExtra}
      {fieldsBody}
      {supersededBlock}
      {registryProblemBlock}
      {auditWarning && (
        <div className="mt-2 text-[11px] text-tcip-warn">Warning: {auditWarning}</div>
      )}
      {note && <div className="mt-2 text-[11px] text-tcip-fp">{note}</div>}
      <div className="mt-2 flex items-center gap-2">
        {!record.confirmed_current && (
          <button
            className="tcip-btn-primary text-[11px]"
            onClick={onConfirm}
            disabled={confirming || withdrawing}
          >
            {confirming ? "Confirming…" : "Confirm this record"}
          </button>
        )}
        {record.confirmed_by && (
          <button
            className="tcip-btn text-[11px]"
            onClick={onWithdraw}
            disabled={confirming || withdrawing}
          >
            {withdrawing ? "Withdrawing…" : "Withdraw this confirmation"}
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
            aria-label={correctionAriaLabel}
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

export interface StatementPanelProps<R extends BaseStatementRecord> {
  heading: string;
  description: ReactNode;
  records: R[];
  loadError: string | null;
  emptyText: string;
  refusalBanner?: ReactNode;
  recordKey: (record: R) => string;
  kindLabelOf: (record: R) => string;
  testIdOf: (record: R) => string;
  refusedOf?: (record: R) => boolean;
  headerExtraOf?: (record: R) => ReactNode;
  fieldsBodyOf: (record: R) => ReactNode;
  supersededBlockOf?: (record: R) => ReactNode;
  registryProblemBlockOf?: (record: R) => ReactNode;
  correctionSeedOf: (record: R) => string;
  correctionAriaLabelOf: (record: R) => string;
  confirmingKey: string | null;
  withdrawingKey: string | null;
  notes: Record<string, string>;
  auditWarnings: Record<string, string | null>;
  onConfirm: (record: R) => void;
  onWithdraw: (record: R) => void;
}

export function StatementPanel<R extends BaseStatementRecord>({
  heading,
  description,
  records,
  loadError,
  emptyText,
  refusalBanner,
  recordKey,
  kindLabelOf,
  testIdOf,
  refusedOf,
  headerExtraOf,
  fieldsBodyOf,
  supersededBlockOf,
  registryProblemBlockOf,
  correctionSeedOf,
  correctionAriaLabelOf,
  confirmingKey,
  withdrawingKey,
  notes,
  auditWarnings,
  onConfirm,
  onWithdraw,
}: StatementPanelProps<R>) {
  return (
    <div className="tcip-panel p-4">
      <div className="tcip-heading mb-1">{heading}</div>
      <p className="mb-3 text-[11px] text-tcip-muted">{description}</p>
      {refusalBanner}
      {loadError && <div className="mb-3 text-[11px] text-tcip-fp">{loadError}</div>}
      {records.length === 0 ? (
        <div className="text-[11px] text-tcip-muted">{emptyText}</div>
      ) : (
        <ul className="flex flex-col gap-2">
          {records.map((record) => {
            const key = recordKey(record);
            return (
              <StatementRow
                key={key}
                record={record}
                kindLabel={kindLabelOf(record)}
                testId={testIdOf(record)}
                refused={refusedOf ? refusedOf(record) : false}
                confirming={confirmingKey === key}
                withdrawing={withdrawingKey === key}
                note={notes[key] || null}
                auditWarning={auditWarnings[key] || null}
                headerExtra={headerExtraOf ? headerExtraOf(record) : null}
                fieldsBody={fieldsBodyOf(record)}
                supersededBlock={supersededBlockOf ? supersededBlockOf(record) : null}
                registryProblemBlock={
                  registryProblemBlockOf ? registryProblemBlockOf(record) : null
                }
                correctionSeed={correctionSeedOf(record)}
                correctionAriaLabel={correctionAriaLabelOf(record)}
                onConfirm={() => onConfirm(record)}
                onWithdraw={() => onWithdraw(record)}
              />
            );
          })}
        </ul>
      )}
    </div>
  );
}
