import { useStore } from "@/store";
import type { TabName } from "@/store/types";

const TABS: { id: TabName; label: string }[] = [
  { id: "annotate", label: "Annotate" },
  { id: "review", label: "Review" },
  { id: "training", label: "Training" },
  { id: "tuning", label: "Tuning" },
  { id: "inference", label: "Inference" },
  { id: "results", label: "Results" },
];

export function TopBar() {
  const activeTab = useStore((s) => s.gui.active_tab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const dataset = useStore((s) => s.gui.dataset);
  const mode = useStore((s) => s.gui.mode);
  const setMode = useStore((s) => s.setMode);
  const active_class = useStore((s) => s.gui.active_class);
  const setActiveClass = useStore((s) => s.setActiveClass);
  const class_names = useStore((s) => s.gui.class_names);
  const wsStatus = useStore((s) => s.wsStatus);

  const currentImage = dataset.image_list[dataset.current_image_index] ?? "—";
  const total = dataset.image_list.length;

  return (
    <div className="h-topbar flex items-center gap-2 px-3 border-b border-tcip-border bg-tcip-panel shrink-0">
      <div className="font-semibold tracking-wide mr-2 text-tcip-fg">TCIP</div>

      <div className="flex items-center gap-1 mr-4">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`px-3 h-7 rounded text-[12px] ${
              activeTab === t.id
                ? "bg-tcip-accent text-white"
                : "bg-transparent text-tcip-fg hover:bg-tcip-border"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {(activeTab === "annotate" || activeTab === "review") && (
        <>
          <div className="flex items-center gap-1">
            <label className="text-[11px] text-tcip-muted">Class</label>
            <select
              className="tcip-select"
              value={active_class}
              onChange={(e) => setActiveClass(Number(e.target.value))}
            >
              {Object.entries(class_names).map(([cid, name]) => (
                <option key={cid} value={cid}>
                  {cid}: {name}
                </option>
              ))}
            </select>
          </div>

          {activeTab === "annotate" && (
            <div className="flex items-center gap-1 ml-2">
              <label className="text-[11px] text-tcip-muted">Mode</label>
              <div className="flex rounded overflow-hidden border border-tcip-border">
                <button
                  onClick={() => setMode("box")}
                  className={`px-3 h-8 text-[12px] ${mode === "box" ? "bg-tcip-accent text-white" : "bg-tcip-panel text-tcip-fg"}`}
                >
                  Box
                </button>
                <button
                  onClick={() => setMode("polygon")}
                  className={`px-3 h-8 text-[12px] ${mode === "polygon" ? "bg-tcip-accent text-white" : "bg-tcip-panel text-tcip-fg"}`}
                >
                  Polygon
                </button>
              </div>
            </div>
          )}
        </>
      )}

      <div className="flex-1" />

      <div className="text-[11px] text-tcip-muted truncate max-w-md">
        {dataset.dataset_root && dataset.date ? (
          <>
            {dataset.dataset_root.split(/[/\\]/).slice(-2).join("/")} · {dataset.date}
          </>
        ) : (
          "no dataset selected"
        )}
      </div>

      <div className="mx-3 text-[12px] text-tcip-fg font-mono">
        {currentImage} {total > 0 && <span className="text-tcip-muted">({dataset.current_image_index + 1} / {total})</span>}
      </div>

      <div className="flex items-center gap-1 text-[11px]">
        <span
          className={`w-2 h-2 rounded-full ${
            wsStatus === "connected"
              ? "bg-tcip-tp"
              : wsStatus === "connecting"
              ? "bg-tcip-fn"
              : "bg-tcip-fp"
          }`}
        />
        <span className="text-tcip-muted">{wsStatus}</span>
      </div>
    </div>
  );
}
