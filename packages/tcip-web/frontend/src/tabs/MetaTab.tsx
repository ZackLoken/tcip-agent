import { useCallback, useEffect, useState } from "react";

import { metaApi, type FrictionReport, type Retrospective } from "@/api/meta";
import { useStore } from "@/store";

/**
 * Surfaces the agent's meta-loop output so a human can read it: friction
 * reports (from the `claude_reports` MCP tool) and end-of-session
 * retrospectives (from `project_retrospective`). Read-only.
 */
export function MetaTab() {
  const projectRoot = useStore((s) => s.gui.dataset.project_root);

  const [reports, setReports] = useState<FrictionReport[]>([]);
  const [retros, setRetros] = useState<Retrospective[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = useCallback(async () => {
    if (!projectRoot) return;
    setLoading(true);
    setError(null);
    try {
      const [r, rt] = await Promise.all([
        metaApi.reports(projectRoot),
        metaApi.retrospectives(projectRoot),
      ]);
      setReports(r.reports);
      setRetros(rt.retrospectives);
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
        <div className="font-semibold text-[13px]">Agent meta-loop</div>
        <button className="tcip-btn text-[11px]" onClick={() => void refresh()} disabled={loading}>
          {loading ? "Loading…" : "Refresh"}
        </button>
        {error && <div className="text-[11px] text-tcip-fp">{error}</div>}
      </div>

      <div className="tcip-panel p-3">
        <div className="text-[12px] text-tcip-muted mb-2">
          Friction reports — {reports.length} shown
        </div>
        {reports.length > 0 ? (
          <div className="flex flex-col gap-2">
            {reports.map((r) => (
              <div key={r.file} className="border-t border-tcip-border pt-2 first:border-t-0 first:pt-0">
                <div className="flex items-center gap-2 text-[11px] text-tcip-muted">
                  <span className="px-1.5 rounded bg-tcip-border text-tcip-fg">{r.category || "—"}</span>
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

      <div className="tcip-panel p-3">
        <div className="text-[12px] text-tcip-muted mb-2">
          Retrospectives — {retros.length} shown
        </div>
        {retros.length > 0 ? (
          <div className="flex flex-col gap-3">
            {retros.map((rt) => (
              <div key={rt.project_id} className="border-t border-tcip-border pt-2 first:border-t-0 first:pt-0">
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
    </div>
  );
}
