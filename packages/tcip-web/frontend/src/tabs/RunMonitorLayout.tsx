/**
 * The shell the Training and Tuning tabs share: a fixed-width scrolling sidebar of runs beside a
 * detail region. Each tab's list is its own (Training's is one row per run, Tuning's is a sweep
 * row that expands into its trials), so it arrives as `children` rather than a row template here.
 */

import type { ReactNode } from "react";

export function RunMonitorEmpty({ children }: { children: ReactNode }) {
  return <div className="text-[11px] text-tcip-muted">{children}</div>;
}

export function RunMonitorLayout({
  title,
  onRefresh,
  headerRight,
  children,
  detailHeader,
  detail,
}: {
  title: string;
  onRefresh: () => void;
  /** The action that hands this tab's request to the agent. */
  headerRight?: ReactNode;
  /** Sidebar body: the request composer and the tab's own list. */
  children: ReactNode;
  detailHeader: ReactNode;
  detail: ReactNode;
}) {
  return (
    <div className="flex-1 grid grid-cols-[400px_1fr] overflow-hidden">
      <div className="border-r border-tcip-border flex flex-col overflow-hidden">
        <div className="px-4 pt-4 pb-2 flex items-center gap-2">
          <span className="tcip-heading">{title}</span>
          <span className="flex-1" />
          <button className="tcip-btn text-[11px]" onClick={onRefresh}>
            <span aria-hidden="true">↻</span>&nbsp;&nbsp;Refresh
          </button>
          {headerRight}
        </div>
        <div className="flex-1 overflow-auto px-4 pb-4">{children}</div>
      </div>

      <div className="flex flex-col overflow-hidden">
        <div className="p-4 border-b border-tcip-border flex items-center gap-2">
          {detailHeader}
        </div>
        <div className="flex-1 p-4 overflow-auto">{detail}</div>
      </div>
    </div>
  );
}
