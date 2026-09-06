import { useEffect, useState } from "react";

import { ProjectBreadcrumb } from "@/components/ProjectBreadcrumb";
import { useReviewColors } from "@/lib/reviewColors";
import { useStore } from "@/store";
import { selectCanvasMatchesDataset } from "@/store/slices/canvas";

function fmtSeconds(totalSeconds: number): string {
  const clamped = Math.max(0, Math.floor(totalSeconds));
  const mins = Math.floor(clamped / 60);
  const secs = clamped % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

// Pluralized on its own value: a count of one never borrows the other count's plural form.
function countLabel(count: number, singular: string, plural: string): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

export function StatusBar() {
  const view = useStore((s) => s.gui.view);
  const canvasMatchesDataset = useStore(selectCanvasMatchesDataset);
  const canvasDims = useStore((s) => ({ w: s.canvas.imgWidth, h: s.canvas.imgHeight }));
  const dirty = useStore((s) => s.canvas.dirty);
  const boxCount = useStore((s) => s.canvas.boxes.length);
  const polyCount = useStore((s) => s.canvas.polygons.length);
  const pointCount = useStore((s) => s.canvas.points.length);
  const matches = useStore((s) => s.review.matches);
  const activeTab = useStore((s) => s.gui.active_tab);
  const sessionTracking = useStore((s) => s.sessionTracking);
  const agentActivity = useStore((s) => s.agentActivity);
  const [reviewColors] = useReviewColors();

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
      {canvasDims.w && canvasMatchesDataset ? (
        <span className="tabular-nums">
          Image: {canvasDims.w}×{canvasDims.h}
        </span>
      ) : null}
      {activeTab === "annotate" && canvasMatchesDataset && (polyCount > 0 || boxCount > 0) && (
        <span className="tabular-nums">
          {polyCount > 0 && countLabel(polyCount, "polygon", "polygons")}
          {polyCount > 0 && boxCount > 0 && <span className="mx-1.5 text-tcip-border">|</span>}
          {boxCount > 0 && countLabel(boxCount, "box", "boxes")}
        </span>
      )}
      {/* Points are counted on their own: they are not detection targets, so they never fold into
          the box count. */}
      {activeTab === "annotate" && canvasMatchesDataset && pointCount > 0 && (
        <span className="tabular-nums">{pointCount} points</span>
      )}
      {activeTab === "annotate" && dirty && <span className="text-tcip-warn">Unsaved changes</span>}
      {activeTab === "review" && matches && (
        <span className="tabular-nums">
          <span style={{ color: reviewColors.tp }}>TP {matches.n_tp}</span>
          <span className="mx-1.5 text-tcip-border">|</span>
          <span style={{ color: reviewColors.fp }}>FP {matches.n_fp}</span>
          <span className="mx-1.5 text-tcip-border">|</span>
          <span style={{ color: reviewColors.fn }}>FN {matches.n_fn}</span>
        </span>
      )}
      {activeTab === "review" && matches && matches.n_total > 0 && (
        <ReviewProgress reviewed={matches.n_reviewed} total={matches.n_total} />
      )}
      <div className="flex-1" />
      {showAgent && agentActivity && (
        <span className="text-tcip-accent" title={JSON.stringify(agentActivity.data)}>
          ⚡ Agent: {agentActivity.eventType}
          {agentStem ? ` (${agentStem})` : ""}
        </span>
      )}
      {/* Project breadcrumb, lower-right: project (recent) · date (switch) · Switch Project (all). */}
      <ProjectBreadcrumb />
    </div>
  );
}

/** Persimmon progress wheel + count of detections reviewed on the current image. */
function ReviewProgress({ reviewed, total }: { reviewed: number; total: number }) {
  const C = 44; // circumference of an r=7 circle (2·π·7)
  const frac = total ? reviewed / total : 0;
  return (
    <span
      className="flex items-center gap-1.5 tabular-nums"
      title="Detections reviewed on this image, updates as you accept, edit, or reject"
    >
      <svg width="13" height="13" viewBox="0 0 20 20" aria-hidden="true">
        <circle cx="10" cy="10" r="7" fill="none" stroke="#33352C" strokeWidth="3" />
        <circle
          cx="10"
          cy="10"
          r="7"
          fill="none"
          stroke="#E6976B"
          strokeWidth="3"
          strokeDasharray={`${frac * C} ${C}`}
          strokeLinecap="round"
          transform="rotate(-90 10 10)"
        />
      </svg>
      {reviewed} / {total} reviewed
    </span>
  );
}
