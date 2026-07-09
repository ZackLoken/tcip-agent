import { useEffect, useState } from "react";

import { useStore } from "@/store";

function fmtSeconds(totalSeconds: number): string {
  const clamped = Math.max(0, Math.floor(totalSeconds));
  const mins = Math.floor(clamped / 60);
  const secs = clamped % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

export function StatusBar() {
  const view = useStore((s) => s.gui.view);
  const canvasDims = useStore((s) => ({ w: s.canvas.imgWidth, h: s.canvas.imgHeight }));
  const dirty = useStore((s) => s.canvas.dirty);
  const matches = useStore((s) => s.review.matches);
  const activeTab = useStore((s) => s.gui.active_tab);
  const sessionTracking = useStore((s) => s.sessionTracking);
  const agentActivity = useStore((s) => s.agentActivity);

  // Per-image timer. Compute the elapsed seconds inside the interval (not in the
  // render body) so there's no Date.now()-during-render hack.
  const [imageSeconds, setImageSeconds] = useState(0);
  useEffect(() => {
    if (activeTab !== "annotate" || !sessionTracking.imageEnterTimeMs) {
      setImageSeconds(0);
      return;
    }
    const enter = sessionTracking.imageEnterTimeMs;
    const compute = () => setImageSeconds(Math.max(0, Math.floor((Date.now() - enter) / 1000)));
    compute();
    const id = window.setInterval(compute, 1000);
    return () => window.clearInterval(id);
  }, [activeTab, sessionTracking.imageEnterTimeMs]);

  // Briefly surface the latest agent panel push (e.g. the agent writing labels).
  const [showAgent, setShowAgent] = useState(false);
  useEffect(() => {
    if (!agentActivity) return;
    // agentActivity is a fresh object per push, so depending on it fires once per event.
    setShowAgent(true);
    const id = window.setTimeout(() => setShowAgent(false), 8000);
    return () => window.clearTimeout(id);
  }, [agentActivity]);

  const agentStem =
    agentActivity && typeof agentActivity.data.stem === "string" ? agentActivity.data.stem : null;

  return (
    <div className="h-statusbar flex items-center gap-4 px-3 border-t border-tcip-border bg-tcip-panel text-[11px] text-tcip-muted shrink-0">
      <span className="tabular-nums">Zoom: {(view.scale * 100).toFixed(0)}%</span>
      {activeTab === "annotate" && sessionTracking.currentImageName ? (
        <span className="tabular-nums">Image time: {fmtSeconds(imageSeconds)}</span>
      ) : null}
      {canvasDims.w ? (
        <span className="tabular-nums">
          Image: {canvasDims.w}×{canvasDims.h}
        </span>
      ) : null}
      {activeTab === "annotate" && dirty && <span className="text-tcip-warn">Unsaved changes</span>}
      {activeTab === "review" && matches && (
        <span className="tabular-nums">
          <span className="text-tcip-tp">TP {matches.n_tp}</span>
          {" · "}
          <span className="text-tcip-fp">FP {matches.n_fp}</span>
          {" · "}
          <span className="text-tcip-fn">FN {matches.n_fn}</span>
        </span>
      )}
      <div className="flex-1" />
      {showAgent && agentActivity && (
        <span className="text-tcip-accent" title={JSON.stringify(agentActivity.data)}>
          ⚡ Agent: {agentActivity.eventType}
          {agentStem ? ` (${agentStem})` : ""}
        </span>
      )}
    </div>
  );
}
