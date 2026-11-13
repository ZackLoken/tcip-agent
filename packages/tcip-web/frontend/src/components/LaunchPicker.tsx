/**
 * The launch surface: a list of rows read from records (never typed by the breeder), each a
 * recorded run or sweep to start again. A run's row also carries its own nested "Data" choice
 * ("As recorded" or a listed partition), submitted with Start as the selection directory the
 * server itself offered, never a path this component resolves.
 */

import { useState, type ReactNode } from "react";

import { messageOf, readRefusalText, StructuredRefusalError } from "@/api/http";
import {
  ADMITTED,
  BLOCKED,
  type ChoiceCheck,
  type SelectionChoice,
  type SplitChoices,
  UNREAD,
  type UnreadSelectionChoice,
} from "@/api/types.generated";
import { ErrorRetry } from "@/components/ErrorRetry";

/** Shown in place of the partition choices when the listing found none. */
export const NO_OTHER_PARTITION =
  "this listing found no other recorded partition the config can bind to; the agent can draw one.";

/** A partition choice's label: its selection directory and, for one that read, its draw seed,
 * grouping and per-side sample counts, each as its record states it. */
function choiceLabel(s: SelectionChoice | UnreadSelectionChoice): ReactNode {
  return (
    <>
      <span className="block font-mono">{s.selection_dir}</span>
      {s.kind === "read" && (
        <span className="block text-tcip-muted">
          seed {s.seed} · {s.group_by} · train {s.train} · val {s.val} · calibration {s.calibration}
        </span>
      )}
    </>
  );
}

/** Whether the offered selection `s` can be bound: one that will not read is its own refusal,
 * one that read answers by its check. */
function checkOf(s: SelectionChoice | UnreadSelectionChoice): ChoiceCheck | UnreadSelectionChoice {
  return s.kind === UNREAD ? s : s.check;
}

export interface LaunchPickerRow {
  key: string;
  /** The row's own display content (id, builder/task, subject, date, state, parent...). */
  content: ReactNode;
  /** The branch line shown once this row is selected, describing what Start will do. */
  branchLine: string;
  /** Replaces branchLine once a partition (rather than "As recorded") is the current choice. */
  branchLineForData?: string;
  /** Present only for a run's row: its data choices, the nested "Data" choice under the branch
   * line ("As recorded" and each listed partition, each with its own compatibility). */
  data?: SplitChoices;
  /** Set only while this row's own Data choices are still being fetched, after selection. */
  dataLoading?: boolean;
  /** Set only when the per-row Data-choices fetch itself failed: shown as text on the row
   * rather than swallowed into a Data section with no control. */
  dataError?: string;
  /** Starts the run/sweep this row names, with the chosen selection directory or null for "As
   * recorded". Rejects with the backend's refusal on failure. */
  onStart: (selectionDir: string | null) => Promise<void>;
}

interface Refusal {
  key: string;
  issues: string[];
}

export interface LaunchPickerProps {
  list: {
    title: string;
    emptyMessage: string;
    rows: LaunchPickerRow[];
    /** Set when the rows themselves failed to load: rendered in place of emptyMessage, with
     * onRetry as its Retry control, the way the runs list reports its own load failure. */
    error?: string;
    onRetry: () => void;
  };
  /** Called once a row transitions into the selected state (never on deselection), so a caller
   * can fetch that row's own Data choices lazily instead of prefetching every row's on open. */
  onSelect?: (key: string) => void;
}

function refusalIssues(e: unknown): string[] {
  if (e instanceof StructuredRefusalError && Array.isArray(e.detail.issues)) {
    const issues = e.detail.issues.filter((i): i is string => typeof i === "string");
    if (issues.length > 0) return issues;
  }
  return [messageOf(e)];
}

export function LaunchPicker({ list, onSelect }: LaunchPickerProps) {
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [selectedDataDir, setSelectedDataDir] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [refusal, setRefusal] = useState<Refusal | null>(null);

  async function start(row: LaunchPickerRow) {
    setStarting(true);
    setRefusal(null);
    try {
      await row.onStart(row.data ? selectedDataDir : null);
      setSelectedKey(null);
    } catch (e) {
      setRefusal({ key: row.key, issues: refusalIssues(e) });
    } finally {
      setStarting(false);
    }
  }

  function select(key: string) {
    setRefusal(null);
    setSelectedDataDir(null);
    setSelectedKey((current) => {
      const next = current === key ? null : key;
      if (next !== null) onSelect?.(next);
      return next;
    });
  }

  return (
    <div className="flex flex-col gap-3">
      <div>
        <div className="tcip-heading mb-1">{list.title}</div>
        {list.error ? (
          <ErrorRetry message={list.error} onRetry={list.onRetry} className="text-[13px]" />
        ) : list.rows.length === 0 ? (
          <div className="text-[13px] text-tcip-muted">{list.emptyMessage}</div>
        ) : (
          <ul className="space-y-1">
            {list.rows.map((row) => {
              const selected = selectedKey === row.key;
              const chosen = row.data?.selections.find((c) => c.selection_dir === selectedDataDir);
              const current =
                selectedDataDir === null ? row.data?.as_recorded.check : chosen && checkOf(chosen);
              const dataStillLoading = Boolean(row.dataLoading);
              return (
                <li key={row.key}>
                  <button
                    type="button"
                    aria-expanded={selected}
                    className={`w-full p-2 rounded border text-left transition-colors ${
                      selected
                        ? "border-tcip-accent bg-tcip-accent/10"
                        : "border-tcip-border hover:border-tcip-border-hover hover:bg-tcip-hover"
                    }`}
                    onClick={() => select(row.key)}
                  >
                    {row.content}
                  </button>
                  {selected && (
                    <div className="mt-1 pl-2">
                      <div className="text-[12px] text-tcip-muted mb-1">
                        {selectedDataDir && row.branchLineForData
                          ? row.branchLineForData
                          : row.branchLine}
                      </div>
                      {row.dataLoading && (
                        <div className="mb-2 text-[12px] text-tcip-muted">
                          Loading data choices…
                        </div>
                      )}
                      {row.dataError && (
                        <div className="mb-2 text-[12px] tcip-problem">{row.dataError}</div>
                      )}
                      {row.data && (
                        <div className="mb-2">
                          <div className="tcip-heading mb-1">Data</div>
                          <ul className="space-y-1">
                            <li>
                              <label className="flex items-start gap-1 text-[12px]">
                                <input
                                  type="radio"
                                  name={`data-${row.key}`}
                                  checked={selectedDataDir === null}
                                  disabled={row.data.as_recorded.check.kind === BLOCKED}
                                  onChange={() => setSelectedDataDir(null)}
                                />
                                <span>
                                  As recorded: {row.data.as_recorded.line}
                                  {row.data.as_recorded.check.kind === BLOCKED &&
                                    selectedDataDir !== null && (
                                      <span className="block tcip-problem">
                                        {readRefusalText(row.data.as_recorded.check.reason)}
                                      </span>
                                    )}
                                </span>
                              </label>
                            </li>
                            {row.data.selections.length === 0 ? (
                              <li className="text-[12px] text-tcip-muted pl-4">
                                {NO_OTHER_PARTITION}
                              </li>
                            ) : (
                              <>
                                {row.data.selections.map((choice) => {
                                  const check = checkOf(choice);
                                  return (
                                    <li key={choice.selection_dir}>
                                      <label className="flex items-start gap-1 text-[12px]">
                                        <input
                                          type="radio"
                                          name={`data-${row.key}`}
                                          checked={selectedDataDir === choice.selection_dir}
                                          disabled={check.kind !== ADMITTED}
                                          onChange={() => setSelectedDataDir(choice.selection_dir)}
                                        />
                                        <span>
                                          {choiceLabel(choice)}
                                          {choice.replaced_split_keys.length > 0 && (
                                            <span className="block text-tcip-muted">
                                              replaces the recorded{" "}
                                              {choice.replaced_split_keys.join(", ")}
                                            </span>
                                          )}
                                          {check.kind !== ADMITTED &&
                                            choice.selection_dir !== selectedDataDir && (
                                              <span className="block tcip-problem">
                                                {readRefusalText(check.reason)}
                                              </span>
                                            )}
                                        </span>
                                      </label>
                                    </li>
                                  );
                                })}
                              </>
                            )}
                          </ul>
                        </div>
                      )}
                      <button
                        type="button"
                        className="tcip-btn-primary text-[13px]"
                        disabled={
                          starting ||
                          dataStillLoading ||
                          (current !== undefined && current.kind !== ADMITTED)
                        }
                        onClick={() => void start(row)}
                      >
                        Start
                      </button>
                      {dataStillLoading ? (
                        <div className="mt-1 text-[12px] text-tcip-muted" role="status">
                          checking the data choice
                        </div>
                      ) : current !== undefined && current.kind !== ADMITTED ? (
                        <div className="mt-1 text-[12px] tcip-problem" role="status">
                          {readRefusalText(current.reason)}
                        </div>
                      ) : refusal?.key === row.key ? (
                        <ul role="status" className="mt-1 text-[13px] tcip-problem list-disc pl-4">
                          {refusal.issues.map((issue) => (
                            <li key={issue}>{issue}</li>
                          ))}
                        </ul>
                      ) : null}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
