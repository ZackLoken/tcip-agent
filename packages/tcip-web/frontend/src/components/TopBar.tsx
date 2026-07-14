import { useEffect, useState } from "react";

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
 * Global app bar: logo, centered tab navigation, agent-rail toggle, and a connection
 * indicator that appears only when the connection is degraded. The project breadcrumb
 * lives in the StatusBar's lower-right corner; tab-specific tools live in each tab's
 * own toolbar (e.g. AnnotateToolbar), not here.
 */
export function TopBar() {
  const activeTab = useStore((s) => s.gui.active_tab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const wsStatus = useStore((s) => s.wsStatus);
  const terminalOpen = useStore((s) => s.terminalOpen);
  const setTerminalOpen = useStore((s) => s.setTerminalOpen);

  // Grace window so the initial handshake doesn't flash "reconnecting…" on every load;
  // hard failures (error/disconnected) surface immediately.
  const [graceOver, setGraceOver] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setGraceOver(true), 2000);
    return () => clearTimeout(t);
  }, []);
  const degraded =
    wsStatus === "error" || wsStatus === "disconnected" || (wsStatus === "connecting" && graceOver);

  return (
    <div className="h-topbar grid grid-cols-[1fr_auto_1fr] items-center px-3 border-b border-tcip-border bg-tcip-panel shrink-0">
      <img
        src="/assets/si_logo.png"
        alt="Savanna Institute"
        className="h-9 w-auto select-none justify-self-start"
        draggable={false}
      />

      {/* Tabs — the grid's auto track keeps them truly centered */}
      <div className="flex items-center gap-1 justify-self-center">
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

      <div className="flex items-center justify-self-end">
        {/* Connection state — visible only when something is wrong; it self-dismisses
            when the auto-reconnect succeeds. */}
        {degraded && (
          <div className="flex items-center gap-1.5 h-6 px-2 mr-2 rounded-full border border-tcip-border bg-tcip-bg text-[11px]">
            <span
              className={`w-2 h-2 rounded-full ${
                wsStatus === "connecting" ? "bg-tcip-fn" : "bg-tcip-fp"
              }`}
            />
            <span className="text-tcip-muted">
              {wsStatus === "connecting" ? "reconnecting…" : "disconnected — retrying"}
            </span>
          </div>
        )}

        {/* Agent rail toggle */}
        <button
          onClick={() => setTerminalOpen(!terminalOpen)}
          aria-pressed={terminalOpen}
          aria-label="Toggle agent terminal"
          className={`px-2.5 h-7 rounded text-[12px] font-medium transition-colors ${
            terminalOpen
              ? "bg-tcip-accent text-white"
              : "bg-transparent text-tcip-muted hover:text-tcip-fg hover:bg-tcip-hover"
          }`}
        >
          TCIP Agent
        </button>
      </div>
    </div>
  );
}
