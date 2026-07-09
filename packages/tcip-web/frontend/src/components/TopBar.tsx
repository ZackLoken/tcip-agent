import { useStore } from "@/store";
import type { TabName } from "@/store/types";

const TABS: { id: TabName; label: string }[] = [
  { id: "annotate", label: "Annotate" },
  { id: "review", label: "Review" },
  { id: "training", label: "Training" },
  { id: "tuning", label: "Tuning" },
  { id: "inference", label: "Inference" },
  { id: "results", label: "Results" },
  { id: "meta", label: "Meta" },
];

/**
 * Global app bar: logo, tab navigation, the loaded-dataset breadcrumb, and the WS
 * connection pill. Tab-specific tools live in each tab's own toolbar (e.g.
 * AnnotateToolbar), not here.
 */
export function TopBar() {
  const activeTab = useStore((s) => s.gui.active_tab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const dataset = useStore((s) => s.gui.dataset);
  const wsStatus = useStore((s) => s.wsStatus);

  return (
    <div className="h-topbar flex items-center gap-2 px-3 border-b border-tcip-border bg-tcip-panel shrink-0">
      <img
        src="/assets/si_logo.png"
        alt="Savanna Institute"
        className="h-9 w-auto mr-2 select-none"
        draggable={false}
      />

      {/* Tabs */}
      <div className="flex items-center gap-1 mr-4">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`px-3 h-7 rounded text-[12px] font-medium transition-colors ${
              activeTab === t.id
                ? "bg-tcip-accent text-white"
                : "bg-transparent text-tcip-muted hover:text-tcip-fg hover:bg-tcip-hover"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="flex-1" />

      {/* Dataset breadcrumb */}
      <div className="text-[11px] text-tcip-muted truncate max-w-md">
        {dataset.dataset_root && dataset.date ? (
          <>
            <span className="font-mono">
              {dataset.dataset_root.split(/[/\\]/).slice(-2).join("/")}
            </span>
            <span className="mx-1.5 text-tcip-border">·</span>
            <span className="font-mono">{dataset.date}</span>
          </>
        ) : (
          "no dataset selected"
        )}
      </div>

      {/* WS pill */}
      <div className="flex items-center gap-1.5 h-6 px-2 ml-2 rounded-full border border-tcip-border bg-tcip-bg text-[11px]">
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
