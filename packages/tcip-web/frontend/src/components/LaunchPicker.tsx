/**
 * The config-picker launch surface, shared by the Training and Tuning tabs' headers: a list of
 * rows read from records (never typed by the breeder) plus the agent request composer that
 * remains reachable from both. Training passes real rows (its configs) and Tuning passes none
 * (a sweep from a config that has never been swept is the agent's own path); the composer row
 * is always present and always last. A training-config row also carries its own nested "Data"
 * choice ("As recorded" or a listed partition), submitted with Start as the split manifest
 * directory the server itself offered, never a path this component resolves.
 */

import { useId, useState, type ReactNode } from "react";

import { StructuredRefusalError } from "@/api/http";

export interface DataChoice {
  /** The split manifest directory this choice binds to; the string Start submits unchanged. */
  manifestDir: string;
  /** Draw seed, grouping and per-side member counts under the config's own date. */
  label: ReactNode;
  disabled?: boolean;
  /** Shown under a disabled choice: the compatibility reason, or an unreadable record's text. */
  reason?: string;
  /** The recorded data.split keys choosing this partition drops (seed, group_by, ...), shown
   * beside the two standing disclosures; empty or absent when the config recorded none. */
  replacedSplitKeys?: string[];
}

export interface DataPicker {
  /** "As recorded"'s own case line, read from the snapshot: bound to a manifest, or drawn. */
  asRecordedLine: string;
  asRecordedDisabled?: boolean;
  asRecordedReason?: string;
  /** One entry per offered or refused candidate manifest; empty when none was found. */
  choices: DataChoice[];
  /** Shown in place of the choices when none were found. */
  absenceMessage: string;
}

export interface LaunchPickerRow {
  key: string;
  /** The row's own display content (id, builder/task, subject, date, state, parent...). */
  content: ReactNode;
  /** The branch line shown once this row is selected, describing what Start will do. */
  branchLine: string;
  /** Replaces branchLine once a partition (rather than "As recorded") is the current choice. */
  branchLineForData?: string;
  /** Present only for a training-config row: the nested "Data" choice under the branch line. */
  data?: DataPicker;
  /** Set only while this row's own Data choices are still being fetched, after selection. */
  dataLoading?: boolean;
  /** Set only when the per-row Data-choices fetch itself failed: shown as text on the row
   * rather than swallowed into a Data section with no control. */
  dataError?: string;
  /** Starts the run/sweep this row names, with the chosen manifest directory or null for "As
   * recorded". Rejects with the backend's refusal on failure. */
  onStart: (splitManifestDir: string | null) => Promise<void>;
}

interface Refusal {
  key: string;
  issues: string[];
}

export interface LaunchPickerProps {
  /** Omitted entirely for a picker with no records surface: the panel then offers only the
   * agent row (Tuning's "Start a sweep" header, where a sweep from a never-swept config is
   * always the agent's own path). */
  list?: {
    title: string;
    emptyMessage: string;
    rows: LaunchPickerRow[];
    /** Set when the rows themselves failed to load: rendered in place of emptyMessage, with
     * onRetry as its Retry control, the way the runs list reports its own load failure. */
    error?: string;
    onRetry?: () => void;
  };
  composerLabel: string;
  request: string;
  onRequestChange: (text: string) => void;
  onSend: () => void;
  /** Called once a row transitions into the selected state (never on deselection), so a caller
   * can fetch that row's own Data choices lazily instead of prefetching every row's on open. */
  onSelect?: (key: string) => void;
}

function refusalIssues(e: unknown): string[] {
  if (e instanceof StructuredRefusalError && Array.isArray(e.detail.issues)) {
    const issues = e.detail.issues.filter((i): i is string => typeof i === "string");
    if (issues.length > 0) return issues;
  }
  return [e instanceof Error ? e.message : String(e)];
}

export function LaunchPicker({
  list,
  composerLabel,
  request,
  onRequestChange,
  onSend,
  onSelect,
}: LaunchPickerProps) {
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [selectedDataDir, setSelectedDataDir] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [refusal, setRefusal] = useState<Refusal | null>(null);
  const composerLabelId = useId();

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
      {list && (
        <div>
          <div className="tcip-heading mb-1">{list.title}</div>
          {list.error ? (
            <div className="text-[11px] text-tcip-fp">
              {list.error}{" "}
              <button
                type="button"
                className="tcip-btn text-[11px] ml-1"
                onClick={() => list.onRetry?.()}
              >
                Retry
              </button>
            </div>
          ) : list.rows.length === 0 ? (
            <div className="text-[11px] text-tcip-muted">{list.emptyMessage}</div>
          ) : (
            <ul className="space-y-1">
              {list.rows.map((row) => {
                const selected = selectedKey === row.key;
                const selectedChoice = row.data
                  ? selectedDataDir === null
                    ? { disabled: row.data.asRecordedDisabled, reason: row.data.asRecordedReason }
                    : row.data.choices.find((c) => c.manifestDir === selectedDataDir)
                  : undefined;
                const startBlocked = Boolean(selectedChoice?.disabled);
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
                        <div className="text-[10px] text-tcip-muted mb-1">
                          {selectedDataDir && row.branchLineForData
                            ? row.branchLineForData
                            : row.branchLine}
                        </div>
                        {row.dataLoading && (
                          <div className="mb-2 text-[10px] text-tcip-muted">
                            Loading data choices…
                          </div>
                        )}
                        {row.dataError && (
                          <div className="mb-2 text-[10px] text-tcip-fp">{row.dataError}</div>
                        )}
                        {row.data && (
                          <div className="mb-2">
                            <div className="tcip-heading mb-1">Data</div>
                            <div className="text-[10px] text-tcip-muted mb-1">
                              Members are checked against the labels at launch; an unadmitted member
                              refuses the run.
                            </div>
                            <ul className="space-y-1">
                              <li>
                                <label className="flex items-start gap-1 text-[10px]">
                                  <input
                                    type="radio"
                                    name={`data-${row.key}`}
                                    checked={selectedDataDir === null}
                                    disabled={row.data.asRecordedDisabled}
                                    onChange={() => setSelectedDataDir(null)}
                                  />
                                  <span>
                                    As recorded: {row.data.asRecordedLine}
                                    {row.data.asRecordedDisabled &&
                                      row.data.asRecordedReason &&
                                      selectedDataDir !== null && (
                                        <span className="block text-tcip-fp">
                                          {row.data.asRecordedReason}
                                        </span>
                                      )}
                                  </span>
                                </label>
                              </li>
                              {row.data.choices.length === 0 ? (
                                <li className="text-[10px] text-tcip-muted pl-4">
                                  {row.data.absenceMessage}
                                </li>
                              ) : (
                                <>
                                  <li className="text-[10px] text-tcip-muted pl-4">
                                    Choosing a partition replaces any recorded explicit validation
                                    source with its own val side; admission then reads confirmed
                                    negatives under the labels&apos; date.
                                  </li>
                                  {row.data.choices.map((choice) => (
                                    <li key={choice.manifestDir}>
                                      <label className="flex items-start gap-1 text-[10px]">
                                        <input
                                          type="radio"
                                          name={`data-${row.key}`}
                                          checked={selectedDataDir === choice.manifestDir}
                                          disabled={choice.disabled}
                                          onChange={() => setSelectedDataDir(choice.manifestDir)}
                                        />
                                        <span>
                                          {choice.label}
                                          {choice.replacedSplitKeys &&
                                            choice.replacedSplitKeys.length > 0 && (
                                              <span className="block text-tcip-muted">
                                                replaces the recorded split policy:{" "}
                                                {choice.replacedSplitKeys.join(", ")}
                                              </span>
                                            )}
                                          {choice.disabled &&
                                            choice.reason &&
                                            choice.manifestDir !== selectedDataDir && (
                                              <span className="block text-tcip-fp">
                                                {choice.reason}
                                              </span>
                                            )}
                                        </span>
                                      </label>
                                    </li>
                                  ))}
                                </>
                              )}
                            </ul>
                          </div>
                        )}
                        <button
                          type="button"
                          className="tcip-btn-primary text-[11px]"
                          disabled={starting || startBlocked || dataStillLoading}
                          onClick={() => void start(row)}
                        >
                          Start
                        </button>
                        {dataStillLoading ? (
                          <div className="mt-1 text-[10px] text-tcip-muted" role="status">
                            checking the data choice
                          </div>
                        ) : startBlocked && selectedChoice?.reason ? (
                          <div className="mt-1 text-[10px] text-tcip-fp" role="status">
                            {selectedChoice.reason}
                          </div>
                        ) : refusal?.key === row.key ? (
                          <ul
                            role="status"
                            className="mt-1 text-[11px] text-tcip-fp list-disc pl-4"
                          >
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
      )}

      <div>
        <label htmlFor={composerLabelId} className="tcip-heading mb-1 block">
          {composerLabel}
        </label>
        <textarea
          id={composerLabelId}
          className="tcip-input w-full h-28 text-[11px] leading-4 resize-none mb-2"
          value={request}
          onChange={(e) => onRequestChange(e.target.value)}
          spellCheck={true}
        />
        <button
          type="button"
          className="tcip-btn-primary text-[11px]"
          onClick={onSend}
          disabled={!request.trim()}
        >
          Send to agent
        </button>
      </div>
    </div>
  );
}
