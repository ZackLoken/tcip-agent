import { useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";

import { metaApi, type FrictionReport, type Retrospective } from "@/api/meta";
import { sessionsApi, type SessionEntry } from "@/api/sessions";
import { CollapsibleSection } from "@/components/CollapsibleSection";
import { useStore } from "@/store";

// Sized to this panel's own typography rather than the markdown renderer's defaults, and links
// open in a new tab so a retrospective can never navigate the app away from unsaved GUI state.
const MARKDOWN_COMPONENTS: Components = {
  h1: ({ children }) => <div className="tcip-heading mt-2 mb-1">{children}</div>,
  h2: ({ children }) => <div className="tcip-heading mt-2 mb-1">{children}</div>,
  h3: ({ children }) => (
    <div className="text-[11px] font-semibold text-tcip-fg mt-2 mb-1">{children}</div>
  ),
  p: ({ children }) => <p className="text-[11px] leading-4 my-1">{children}</p>,
  ul: ({ children }) => <ul className="list-disc pl-4 text-[11px] leading-4 my-1">{children}</ul>,
  ol: ({ children }) => (
    <ol className="list-decimal pl-4 text-[11px] leading-4 my-1">{children}</ol>
  ),
  li: ({ children }) => <li className="my-0.5">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-tcip-fg">{children}</strong>,
  em: ({ children }) => <em className="italic text-tcip-muted">{children}</em>,
  code: ({ children }) => <code className="font-mono text-[10px] text-tcip-fg">{children}</code>,
  pre: ({ children }) => (
    <pre className="text-[10px] leading-4 overflow-auto my-1 whitespace-pre-wrap">{children}</pre>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-tcip-border pl-2 text-tcip-muted">
      {children}
    </blockquote>
  ),
  hr: () => <hr className="my-2 border-tcip-border" />,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-tcip-accent underline">
      {children}
    </a>
  ),
};

function fmtDuration(seconds: number): string {
  const s = Math.round(seconds);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return m < 60 ? `${m}m ${rem}s` : `${Math.floor(m / 60)}h ${m % 60}m`;
}

/**
 * Surfaces the agent's meta-loop output so a human can read it: friction
 * reports (from the `claude_reports` MCP tool) and end-of-session
 * retrospectives (from `project_retrospective`). Read-only.
 */
export function MetaTab() {
  const projectRoot = useStore((s) => s.gui.dataset.project_root);

  const [reports, setReports] = useState<FrictionReport[]>([]);
  const [retros, setRetros] = useState<Retrospective[]>([]);
  const [sessions, setSessions] = useState<SessionEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!projectRoot) return;
    setLoading(true);
    setError(null);
    try {
      const [r, rt, sess] = await Promise.all([
        metaApi.reports(projectRoot),
        metaApi.retrospectives(projectRoot),
        sessionsApi.load(projectRoot).catch(() => ({ sessions: [] as SessionEntry[] })),
      ]);
      setReports(r.reports);
      setRetros(rt.retrospectives);
      setSessions(sess.sessions ?? []);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [projectRoot]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // A session with no image entries touched nothing: the backend drops an image entry that ends up
  // with no time, no adds and no final count, so an empty `images` map is the honest "nothing
  // happened" signal. Zero new annotation records is not, since confirming a negative and
  // reviewing an existing label are both real effort that adds no record.
  const shownSessions = useMemo(
    () => sessions.filter((s) => Object.keys(s.images ?? {}).length > 0),
    [sessions],
  );

  return (
    <div className="flex-1 overflow-auto p-4 flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <div className="tcip-heading">Agent meta-loop</div>
        <button className="tcip-btn text-[11px]" onClick={() => void refresh()} disabled={loading}>
          {loading ? "Loading…" : <>↻&nbsp;&nbsp;Refresh</>}
        </button>
        {error && <div className="text-[11px] text-tcip-fp">{error}</div>}
      </div>

      <CollapsibleSection
        className="tcip-panel p-4"
        title="Friction reports"
        right={`${reports.length} shown`}
        storageKey="tcip.meta.frictionOpen"
        defaultOpen
      >
        {reports.length > 0 ? (
          <div className="flex flex-col gap-2">
            {reports.map((r) => (
              <div
                key={r.file}
                className="border-t border-tcip-border pt-2 first:border-t-0 first:pt-0"
              >
                <div className="flex items-center gap-2 text-[11px] text-tcip-muted">
                  <span className="tcip-badge bg-tcip-border/60 text-tcip-fg">
                    {r.category || "—"}
                  </span>
                  <span className="font-mono">{r.timestamp ?? r.file}</span>
                </div>
                <div className="text-[12px] mt-1 whitespace-pre-wrap">{r.detail}</div>
                {r.context && Object.keys(r.context).length > 0 && (
                  <pre className="text-[10px] text-tcip-muted mt-1 overflow-auto">
                    {JSON.stringify(r.context, null, 2)}
                  </pre>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[11px] text-tcip-muted">No friction reports yet.</div>
        )}
      </CollapsibleSection>

      <CollapsibleSection
        className="tcip-panel p-4"
        title="Retrospectives"
        right={`${retros.length} shown`}
        storageKey="tcip.meta.retrospectivesOpen"
        defaultOpen
      >
        {retros.length > 0 ? (
          <div className="flex flex-col gap-3">
            {retros.map((rt) => (
              <div
                key={rt.project_id}
                className="border-t border-tcip-border pt-2 first:border-t-0 first:pt-0"
              >
                <div className="flex items-center gap-2 text-[11px] text-tcip-muted">
                  <span className="font-semibold text-tcip-fg">{rt.project_id}</span>
                  <span className="font-mono">{rt.timestamp}</span>
                </div>
                <div className="mt-1">
                  <ReactMarkdown components={MARKDOWN_COMPONENTS}>{rt.content}</ReactMarkdown>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[11px] text-tcip-muted">No retrospectives yet.</div>
        )}
      </CollapsibleSection>

      <CollapsibleSection
        className="tcip-panel p-4"
        title="Annotation sessions"
        right={`${shownSessions.length} recorded`}
        storageKey="tcip.meta.sessionsOpen"
        defaultOpen
      >
        {shownSessions.length > 0 ? (
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-tcip-border">
                <th className="tcip-th">Started</th>
                <th className="tcip-th">User</th>
                <th className="tcip-th">Images</th>
                <th className="tcip-th">Annotations</th>
                <th className="tcip-th">Time</th>
                <th className="tcip-th">Avg / annotation</th>
              </tr>
            </thead>
            <tbody>
              {shownSessions.map((s, i) => (
                <tr
                  key={`${s.started}-${i}`}
                  className="border-t border-tcip-border first:border-t-0"
                >
                  <td className="py-1.5 pr-3 font-mono">{s.started}</td>
                  <td className="pr-3">{s.user || "—"}</td>
                  <td className="pr-3 tabular-nums">{s.images_annotated}</td>
                  <td className="pr-3 tabular-nums">{s.total_annotations}</td>
                  <td
                    className="pr-3 tabular-nums"
                    title={`New annotations: ${fmtDuration(s.new_annotation_seconds)} · Review: ${fmtDuration(s.review_seconds)} · Negative confirmation: ${fmtDuration(s.negative_confirmation_seconds)}`}
                  >
                    {fmtDuration(s.total_time_seconds)}
                  </td>
                  <td className="pr-3 tabular-nums">
                    {s.avg_seconds_per_annotation ? `${s.avg_seconds_per_annotation}s` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="text-[11px] text-tcip-muted">
            No annotation sessions yet; they're recorded automatically while you annotate.
          </div>
        )}
      </CollapsibleSection>
    </div>
  );
}
