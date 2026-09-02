import { useEffect, useRef, useState } from "react";

import { TAB_NAMES } from "@/api/types.generated";
import { TAB_LABELS } from "@/lib/tabLabels";
import { useStore } from "@/store";
import type { TabName } from "@/store/types";

/** The id both this strip's own tab buttons and App's tab panel wrapper derive from a tab
 * name, so `aria-controls`/`id` and `aria-labelledby` name the same element by construction. */
export function tabButtonId(tab: TabName): string {
  return `tab-${tab}`;
}

export function tabPanelId(tab: TabName): string {
  return `tabpanel-${tab}`;
}

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
  const canvasBindingMissing = useStore((s) => s.canvasBindingMissing);
  const terminalOpen = useStore((s) => s.terminalOpen);
  const setTerminalOpen = useStore((s) => s.setTerminalOpen);

  // Grace window so the initial handshake doesn't flash "reconnecting…" on every load;
  // hard failures (error/disconnected) surface immediately.
  const [graceOver, setGraceOver] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setGraceOver(true), 2000);
    return () => clearTimeout(t);
  }, []);
  const wsDegraded =
    wsStatus === "error" || wsStatus === "disconnected" || (wsStatus === "connecting" && graceOver);
  const degraded = wsDegraded || canvasBindingMissing;
  const label = wsDegraded
    ? wsStatus === "connecting"
      ? "reconnecting…"
      : "disconnected, retrying"
    : "canvas not synced, reopen the project";

  const tabRefs = useRef<Partial<Record<TabName, HTMLButtonElement>>>({});

  // Roving tabindex: arrow keys move focus and selection together between tabs, wrapping at
  // either end; only the active tab stays a Tab stop, matching the tablist pattern.
  function onTabKeyDown(e: React.KeyboardEvent<HTMLButtonElement>, index: number) {
    let nextIndex: number | null = null;
    if (e.key === "ArrowRight") nextIndex = (index + 1) % TAB_NAMES.length;
    else if (e.key === "ArrowLeft") nextIndex = (index - 1 + TAB_NAMES.length) % TAB_NAMES.length;
    else if (e.key === "Home") nextIndex = 0;
    else if (e.key === "End") nextIndex = TAB_NAMES.length - 1;
    if (nextIndex === null) return;
    e.preventDefault();
    const nextTab = TAB_NAMES[nextIndex];
    setActiveTab(nextTab);
    tabRefs.current[nextTab]?.focus();
  }

  return (
    <div className="h-topbar grid grid-cols-[1fr_auto_1fr] items-center px-3 border-b border-tcip-border bg-tcip-panel shrink-0">
      <img
        src="/assets/si_logo.png"
        alt="Savanna Institute"
        className="h-9 w-auto select-none justify-self-start"
        draggable={false}
      />

      {/* Tabs: the grid's auto track keeps them truly centered */}
      <div role="tablist" aria-label="Tabs" className="flex items-center gap-1 justify-self-center">
        {TAB_NAMES.map((id, index) => (
          <button
            key={id}
            id={tabButtonId(id)}
            ref={(el) => {
              if (el) tabRefs.current[id] = el;
            }}
            role="tab"
            aria-selected={activeTab === id}
            aria-controls={tabPanelId(id)}
            tabIndex={activeTab === id ? 0 : -1}
            onClick={() => setActiveTab(id)}
            onKeyDown={(e) => onTabKeyDown(e, index)}
            className={`px-3 h-7 rounded text-[12px] font-medium transition-colors ${
              activeTab === id
                ? "bg-tcip-accent text-white"
                : "bg-transparent text-tcip-muted hover:text-tcip-fg hover:bg-tcip-hover"
            }`}
          >
            {TAB_LABELS[id]}
          </button>
        ))}
      </div>

      <div className="flex items-center justify-self-end">
        {/* Connection/canvas-binding state: visible only when something is wrong; it
            self-dismisses once the underlying condition resolves. */}
        {degraded && (
          <div className="flex items-center gap-1.5 h-6 px-2 mr-2 rounded-full border border-tcip-border bg-tcip-bg text-[11px]">
            <span
              className={`w-2 h-2 rounded-full ${
                wsStatus === "connecting" && wsDegraded ? "bg-tcip-fn" : "bg-tcip-fp"
              }`}
            />
            <span className="text-tcip-muted">{label}</span>
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
