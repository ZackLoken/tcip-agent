import { Terminal } from "lucide-react";
import { type ReactNode, useRef } from "react";

import { TAB_NAMES } from "@/api/types.generated";
import { AGENT_NAME } from "@/lib/agentName";
import { TAB_LABELS } from "@/lib/tabLabels";
import { useStore } from "@/store";
import { selectAnnotatorNamed } from "@/store/slices/user";
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
 * Global app bar, on every screen: the logo and the `notifications` button; once the person is
 * named, also the centered tab navigation and the agent-rail toggle. The project breadcrumb
 * lives in the StatusBar's lower-right corner; tab-specific tools live in each tab's own toolbar
 * (e.g. AnnotateToolbar), not here.
 */
export function TopBar({ notifications }: { notifications: ReactNode }) {
  const annotatorNamed = useStore(selectAnnotatorNamed);
  const activeTab = useStore((s) => s.gui.active_tab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const terminalOpen = useStore((s) => s.terminalOpen);
  const setTerminalOpen = useStore((s) => s.setTerminalOpen);
  // A minimized rail's switch of harness waits for the agent's hand-off; the rail starts the
  // chosen harness once it is open again.
  const handingOff = useStore((s) => s.terminalPhase === "handing-off") && !terminalOpen;

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
    <div className="h-topbar grid grid-cols-[minmax(max-content,1fr)_minmax(0,auto)_minmax(max-content,1fr)] gap-2 items-center px-3 border-b border-tcip-border bg-tcip-panel shrink-0">
      <img
        src="/assets/WHITE_SI_logo_SIDESTACKED.png"
        alt="Savanna Institute"
        className="h-9 w-auto select-none justify-self-start"
        draggable={false}
      />

      {/* Tabs: the grid's equal outer tracks keep them centered; on a narrow window the outer
          tracks keep their content and the tabs scroll, so the logo and the buttons stay reachable */}
      {annotatorNamed ? (
        <div
          role="tablist"
          aria-label="Tabs"
          className="flex items-center gap-1 justify-self-center min-w-0 max-w-full overflow-x-auto"
        >
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
              className={`px-3 h-7 rounded text-[13px] font-medium transition-colors ${
                activeTab === id
                  ? "bg-tcip-accent text-white"
                  : "bg-transparent text-tcip-muted hover:text-tcip-fg hover:bg-tcip-hover"
              }`}
            >
              {TAB_LABELS[id]}
            </button>
          ))}
        </div>
      ) : (
        <div />
      )}

      <div className="flex items-center justify-self-end">
        {notifications}

        {/* Agent rail toggle */}
        {annotatorNamed && (
          <button
            onClick={() => setTerminalOpen(!terminalOpen)}
            aria-pressed={terminalOpen}
            aria-label="Toggle agent terminal"
            title={
              handingOff
                ? "The agent is writing its hand-off; open the rail to start the chosen harness once it is recorded"
                : AGENT_NAME
            }
            className={`flex items-center gap-1.5 px-2.5 h-7 rounded text-[13px] font-medium transition-colors ${
              terminalOpen
                ? "bg-tcip-accent text-white"
                : "bg-transparent text-tcip-muted hover:text-tcip-fg hover:bg-tcip-hover"
            }`}
          >
            <Terminal aria-hidden="true" size={16} />
            {handingOff && (
              <span
                aria-hidden="true"
                data-testid="handoff-mark"
                className="h-1.5 w-1.5 rounded-full bg-tcip-warn animate-pulse"
              />
            )}
          </button>
        )}
      </div>
    </div>
  );
}
