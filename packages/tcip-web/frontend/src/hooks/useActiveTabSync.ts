/**
 * Mirror the active tab into the backend GUI state so view_gui_state reports the tab the human
 * actually sees. Every writer (TopBar clicks, breadcrumb resets, focus events, restore/boot
 * hydration) mutates gui.active_tab, so one effect covers them all; mergeSnapshot treats
 * active_tab as client-owned, so the WS snapshot echo never loops it back.
 */

import { useEffect } from "react";

import { api } from "@/api/client";
import { useStore } from "@/store";
import type { TabName } from "@/store/types";

// Debounced at the nav-index sync cadence; fire-and-forget (a dropped sync leaves gui.json one
// tab stale until the next change).
let tabSyncTimer: ReturnType<typeof setTimeout> | null = null;
export function syncActiveTab(tab: TabName): void {
  if (tabSyncTimer !== null) clearTimeout(tabSyncTimer);
  tabSyncTimer = setTimeout(() => {
    tabSyncTimer = null;
    api.state.tab(tab).catch(() => {});
  }, 400);
}

export function useActiveTabSync(): void {
  const activeTab = useStore((s) => s.gui.active_tab);
  useEffect(() => {
    syncActiveTab(activeTab);
  }, [activeTab]);
}
