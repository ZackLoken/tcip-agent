import { useCallback, useEffect, useState } from "react";

import { metaApi, type FrictionReport, type Retrospective } from "@/api/meta";
import { sessionsApi, type SessionEntry } from "@/api/sessions";
import { useStore } from "@/store";

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

  return (
    <div className="flex-1 overflow-auto p-4 flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <div className="tcip-heading">Agent meta-loop</div>
        <button className="tcip-btn text-[11px]" onClick={() => void refresh()} disabled={loading}>
          {loading ? "Loading…" : <>↻&nbsp;&nbsp;Refresh</>}
        </button>
        {error && <div className="text-[11px] text-tcip-fp">{error}</div>}
      </div>

      <div className="tcip-panel p-4">
        <div className="tcip-heading mb-3">Friction reports — {reports.length} shown</div>
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
      </div>

      <div className="tcip-panel p-4">
        <div className="tcip-heading mb-3">Retrospectives — {retros.length} shown</div>
        {retros.length > 0 ? (
          <div className="flex flex-col gap-3">
            {retros.map((rt) => (
              <div
                key={rt.project_id}
                className="border-t border-tcip-border pt-2 first:border-t-0 first:pt-0"
              >
                <div className="flex items-center gap-2 text-[11px] text-tcip-muted">
                  <span className="font-semibold text-tcip-fg">{rt.project_id}</span>
                  <span className="font-mono">{rt.modified}</span>
                </div>
                <pre className="text-[11px] mt-1 whitespace-pre-wrap leading-4">{rt.content}</pre>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[11px] text-tcip-muted">No retrospectives yet.</div>
        )}
      </div>

      <div className="tcip-panel p-4">
        <div className="tcip-heading mb-3">Annotation sessions — {sessions.length} recorded</div>
        {sessions.length > 0 ? (
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
              {sessions.map((s, i) => (
                <tr
                  key={`${s.started}-${i}`}
                  className="border-t border-tcip-border first:border-t-0"
                >
                  <td className="py-1.5 pr-3 font-mono">{s.started}</td>
                  <td className="pr-3">{s.user || "—"}</td>
                  <td className="pr-3 tabular-nums">{s.images_annotated}</td>
                  <td className="pr-3 tabular-nums">{s.total_annotations}</td>
                  <td className="pr-3 tabular-nums">{fmtDuration(s.total_time_seconds)}</td>
                  <td className="pr-3 tabular-nums">
                    {s.avg_seconds_per_annotation ? `${s.avg_seconds_per_annotation}s` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div className="text-[11px] text-tcip-muted">
            No annotation sessions yet — they're recorded automatically while you annotate.
          </div>
        )}
      </div>
    </div>
  );
}
