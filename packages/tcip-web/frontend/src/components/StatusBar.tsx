import { useStore } from "@/store";

export function StatusBar() {
  const view = useStore((s) => s.gui.view);
  const canvasDims = useStore((s) => ({ w: s.canvas.imgWidth, h: s.canvas.imgHeight }));
  const dirty = useStore((s) => s.canvas.dirty);
  const matches = useStore((s) => s.review.matches);
  const activeTab = useStore((s) => s.gui.active_tab);

  return (
    <div className="h-statusbar flex items-center gap-4 px-3 border-t border-tcip-border bg-tcip-panel text-[11px] text-tcip-muted shrink-0">
      <span>Zoom: {(view.scale * 100).toFixed(0)}%</span>
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
