import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";

import { metaApi, type FrictionReport, type Retrospective } from "@/api/meta";
import { sessionsApi } from "@/api/sessions";
import type { SessionSummary } from "@/api/types.generated";
import { CollapsibleSection } from "@/components/CollapsibleSection";
import { TabHeading } from "@/components/TabHeading";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";

// Sized to this panel's own typography rather than the markdown renderer's defaults, and links
// open in a new tab so a retrospective can never navigate the app away from unsaved GUI state.
const MARKDOWN_COMPONENTS: Components = {
  h1: ({ children }) => <div className="tcip-heading mt-2 mb-1">{children}</div>,
  h2: ({ children }) => <div className="tcip-heading mt-2 mb-1">{children}</div>,
  h3: ({ children }) => (
    <div className="text-[13px] font-semibold text-tcip-fg mt-2 mb-1">{children}</div>
  ),
  p: ({ children }) => <p className="text-[13px] leading-4 my-1">{children}</p>,
  ul: ({ children }) => <ul className="list-disc pl-4 text-[13px] leading-4 my-1">{children}</ul>,
  ol: ({ children }) => (
    <ol className="list-decimal pl-4 text-[13px] leading-4 my-1">{children}</ol>
  ),
  li: ({ children }) => <li className="my-0.5">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold text-tcip-fg">{children}</strong>,
  em: ({ children }) => <em className="italic text-tcip-muted">{children}</em>,
  code: ({ children }) => <code className="font-mono text-[12px] text-tcip-fg">{children}</code>,
  pre: ({ children }) => (
    <pre className="text-[12px] leading-4 overflow-auto my-1 whitespace-pre-wrap">{children}</pre>
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
 * reports (from the `report_friction` MCP tool) and end-of-session
 * retrospectives (from `write_retrospective`). Read-only.
 */
export function MetaTab() {
  const projectRoot = useStore(selectProjectRoot);

  // null until a load answers, so no section claims to be empty before it is known to be.
  const [reports, setReports] = useState<FrictionReport[] | null>(null);
  const [retros, setRetros] = useState<Retrospective[] | null>(null);
  // undefined until a load answers, null once one has failed: the panel then says so instead of
  // claiming no sessions exist.
  const [sessions, setSessions] = useState<SessionSummary[] | null | undefined>(undefined);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!projectRoot) return;
    setLoading(true);
    setError(null);
    try {
      const [r, rt, sess] = await Promise.all([
        metaApi.reports(),
        metaApi.retrospectives(),
        sessionsApi.load(),
      ]);
      setReports(r.reports);
      setRetros(rt.retrospectives);
      setSessions(sess.sessions);
    } catch (e) {
      setSessions(null);
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [projectRoot]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // No entries means nothing happened; zero added annotations does not, since confirming a
  // negative and reviewing a label are real effort that adds no record.
  const shownSessions = useMemo(
    () => (sessions ?? []).filter((s) => s.entries.length > 0),
    [sessions],
  );

  return (
    <div className="flex-1 overflow-auto p-4 flex flex-col gap-4">
      <TabHeading tab="meta" />
      <div className="flex items-center gap-3">
        <button className="tcip-btn" onClick={() => void refresh()} disabled={loading}>
          Refresh
        </button>
        {error && <div className="tcip-problem">{error}</div>}
      </div>

      <CollapsibleSection
        className="tcip-panel p-4"
        title="Friction"
        right={reports === null ? "" : `${reports.length} shown`}
        storageKey="tcip.meta.frictionOpen"
        defaultOpen
      >
        {reports === null ? null : reports.length > 0 ? (
          <div className="flex flex-col gap-2">
            {reports.map((r) => (
              <div
                key={r.report_id}
                className="border-t border-tcip-border pt-2 first:border-t-0 first:pt-0"
              >
                <div className="flex items-center gap-2 text-tcip-muted">
                  <span className="tcip-badge bg-tcip-border/60 text-tcip-fg">
                    {r.category || UNSET_GLYPH}
                  </span>
                  <span className="font-mono">{r.timestamp ?? r.report_id}</span>
                </div>
                <div className="mt-1 whitespace-pre-wrap">{r.detail ?? r.malformed}</div>
                {r.context && Object.keys(r.context).length > 0 && (
                  <dl className="mt-1 grid grid-cols-[max-content_1fr] gap-x-3 gap-y-0.5 text-tcip-muted">
                    {Object.entries(r.context).map(([key, value]) => (
                      <Fragment key={key}>
                        <dt className="font-mono">{key}</dt>
                        <dd className="font-mono whitespace-pre-wrap break-all">
                          {typeof value === "string" ? value : JSON.stringify(value)}
                        </dd>
                      </Fragment>
                    ))}
                  </dl>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="text-tcip-muted">No reports.</div>
        )}
      </CollapsibleSection>

      <CollapsibleSection
        className="tcip-panel p-4"
        title="Retrospectives"
        right={retros === null ? "" : `${retros.length} shown`}
        storageKey="tcip.meta.retrospectivesOpen"
        defaultOpen
      >
        {retros === null ? null : retros.length > 0 ? (
          <div className="flex flex-col gap-3">
            {retros.map((rt) => (
              <div
                key={rt.project_id}
                className="border-t border-tcip-border pt-2 first:border-t-0 first:pt-0"
              >
                <div className="flex items-center gap-2 text-tcip-muted">
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
          <div className="text-tcip-muted">No retrospectives.</div>
        )}
      </CollapsibleSection>

      <CollapsibleSection
        className="tcip-panel p-4"
        title="Sessions"
        right={
          sessions === undefined
            ? ""
            : sessions === null
              ? "not loaded"
              : `${shownSessions.length} recorded`
        }
        storageKey="tcip.meta.sessionsOpen"
        defaultOpen
      >
        {sessions === undefined ? null : sessions === null ? (
          <div className="tcip-problem">Sessions could not be loaded: {error}</div>
        ) : shownSessions.length > 0 ? (
          <table className="w-full">
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
                  <td className="pr-3">{s.user}</td>
                  <td className="pr-3 tabular-nums">{s.images_annotated}</td>
                  <td className="pr-3 tabular-nums">{s.total_annotations}</td>
                  <td
                    className="pr-3 tabular-nums"
                    title={`New annotations: ${fmtDuration(s.seconds_by_activity.new_annotation)} · Review: ${fmtDuration(s.seconds_by_activity.review)} · Negative confirmation: ${fmtDuration(s.seconds_by_activity.negative_confirmation)}`}
                  >
                    {fmtDuration(s.total_time_seconds)}
                  </td>
                  <td className="pr-3 tabular-nums">
                    {s.avg_seconds_per_annotation === null
                      ? UNSET_GLYPH
                      : `${s.avg_seconds_per_annotation}s`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="text-tcip-muted">No sessions.</div>
        )}
      </CollapsibleSection>
    </div>
  );
}
