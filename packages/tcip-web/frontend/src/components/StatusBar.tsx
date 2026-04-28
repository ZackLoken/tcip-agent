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
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (activeTab !== "annotate" || !sessionTracking.imageEnterTimeMs) return;
    const id = window.setInterval(() => setTick((n) => n + 1), 1000);
    return () => window.clearInterval(id);
  }, [activeTab, sessionTracking.imageEnterTimeMs]);

  const nowMs = Date.now() + tick * 0;
  const imageSeconds = sessionTracking.imageEnterTimeMs
    ? Math.max(0, Math.floor((nowMs - sessionTracking.imageEnterTimeMs) / 1000))
    : 0;

  return (
    <div className="h-statusbar flex items-center gap-4 px-3 border-t border-tcip-border bg-tcip-panel text-[11px] text-tcip-muted shrink-0">
      <span>Zoom: {(view.scale * 100).toFixed(0)}%</span>
      {activeTab === "annotate" && sessionTracking.currentImageName ? (
        <span>Image time: {fmtSeconds(imageSeconds)}</span>
      ) : null}
      {canvasDims.w ? (
        <span>
          Image: {canvasDims.w}×{canvasDims.h}
        </span>
      ) : null}
      {activeTab === "annotate" && dirty && (
        <span className="text-tcip-warn">Unsaved changes</span>
      )}
      {activeTab === "review" && matches && (
        <span>
          <span className="text-tcip-tp">TP {matches.n_tp}</span>
          {" · "}
          <span className="text-tcip-fp">FP {matches.n_fp}</span>
          {" · "}
          <span className="text-tcip-fn">FN {matches.n_fn}</span>
        </span>
      )}
      <div className="flex-1" />
    </div>
  );
}
