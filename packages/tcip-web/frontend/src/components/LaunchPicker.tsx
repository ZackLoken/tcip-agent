/**
 * The config-picker launch surface, shared by the Training and Tuning tabs' headers: a list of
 * rows read from records (never typed by the breeder) plus the agent request composer that
 * remains reachable from both. Training passes real rows (its configs) and Tuning passes none
 * (a sweep from a config that has never been swept is the agent's own path); the composer row
 * is always present and always last.
 */

import { useState, type ReactNode } from "react";

import { StructuredRefusalError } from "@/api/http";

export interface LaunchPickerRow {
  key: string;
  /** The row's own display content (id, builder/task, subject, date, state, parent...). */
  content: ReactNode;
  /** The branch line shown once this row is selected, describing what Start will do. */
  branchLine: string;
  /** Starts the run/sweep this row names. Rejects with the backend's refusal on failure. */
  onStart: () => Promise<void>;
}

interface Refusal {
  key: string;
  issues: string[];
}

export interface LaunchPickerProps {
  /** Omitted entirely for a picker with no records surface: the panel then offers only the
   * agent row (Tuning's "Start a sweep" header, where a sweep from a never-swept config is
   * always the agent's own path). */
  list?: { title: string; emptyMessage: string; rows: LaunchPickerRow[] };
  composerLabel: string;
  request: string;
  onRequestChange: (text: string) => void;
  onSend: () => void;
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
}: LaunchPickerProps) {
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [refusal, setRefusal] = useState<Refusal | null>(null);

  async function start(row: LaunchPickerRow) {
    setStarting(true);
    setRefusal(null);
    try {
      await row.onStart();
      setSelectedKey(null);
    } catch (e) {
      setRefusal({ key: row.key, issues: refusalIssues(e) });
    } finally {
      setStarting(false);
    }
  }

  function select(key: string) {
    setRefusal(null);
    setSelectedKey((current) => (current === key ? null : key));
  }

  return (
    <div className="flex flex-col gap-3">
      {list && (
        <div>
          <div className="tcip-heading mb-1">{list.title}</div>
          {list.rows.length === 0 ? (
            <div className="text-[11px] text-tcip-muted">{list.emptyMessage}</div>
          ) : (
            <ul className="space-y-1">
              {list.rows.map((row) => {
                const selected = selectedKey === row.key;
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
                        <div className="text-[10px] text-tcip-muted mb-1">{row.branchLine}</div>
                        <button
                          type="button"
                          className="tcip-btn-primary text-[11px]"
                          disabled={starting}
                          onClick={() => void start(row)}
                        >
                          Start
                        </button>
                        {refusal?.key === row.key && (
                          <ul className="mt-1 text-[11px] text-tcip-fp list-disc pl-4">
                            {refusal.issues.map((issue) => (
                              <li key={issue}>{issue}</li>
                            ))}
                          </ul>
                        )}
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
        <div className="tcip-heading mb-1">{composerLabel}</div>
        <textarea
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
