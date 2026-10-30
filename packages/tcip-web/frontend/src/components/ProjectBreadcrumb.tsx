/**
 * Status-bar project breadcrumb: three fast-tracks in the lower-right corner:
 *   project name → a dropdown of recent projects (jump straight in),
 *   date        → a dropdown of this project's dates (switch without the workspace),
 *   Switch Project → the full workspace (all projects).
 * The date switch re-opens the same project on another date, keeping the current subject/model
 * when that date has them. (Per-date view/position restore is layered on separately.)
 */

import { useEffect, useRef, useState, type ReactNode } from "react";

import { api } from "@/api/client";
import { messageOf } from "@/api/http";
import type { ProjectsListing } from "@/api/types.generated";
import {
  choiceForDate,
  openable,
  type OpenableProject,
  openWithDefaults,
  startOpen,
} from "@/lib/openProject";
import { loadRecentProjectIds } from "@/lib/recentProjects";
import { useStore } from "@/store";
import type { DatasetSelection, TabName } from "@/store/types";

const DATASET_TABS: ReadonlySet<TabName> = new Set(["annotate", "results"]);

export function ProjectBreadcrumb() {
  const dataset = useStore((s) => s.gui.dataset);
  const openProject = useStore((s) => s.openProject);
  const user = useStore((s) => s.user);
  const clearDataset = useStore((s) => s.clearDataset);
  const activeTab = useStore((s) => s.gui.active_tab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const pushToast = useStore((s) => s.pushToast);

  const [menu, setMenu] = useState<null | "project" | "date">(null);
  const [listing, setListing] = useState<ProjectsListing | null>(null);
  const busy = useStore((s) => s.opening !== null);
  const rootRef = useRef<HTMLDivElement>(null);

  // Load the project list once per open project: names it, and powers the date switcher. Never
  // polled; a rename lands here at the next open.
  useEffect(() => {
    if (!openProject) return;
    let canceled = false;
    api.projects
      .list()
      .then((r) => {
        if (!canceled) setListing(r);
      })
      .catch(() => {
        if (!canceled) pushToast("Could not load the project list.");
      });
    return () => {
      canceled = true;
    };
  }, [openProject, pushToast]);

  // Close the menu on an outside click.
  useEffect(() => {
    if (!menu) return;
    const onDoc = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setMenu(null);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [menu]);

  function switchProject() {
    setMenu(null);
    clearDataset();
    // Land on a dataset-dependent tab so the project front door is shown.
    if (!DATASET_TABS.has(activeTab)) setActiveTab("annotate");
  }

  async function openRecent(project: OpenableProject) {
    setMenu(null);
    // The one opening transition saves the outgoing UI state and restores this project's saved
    // position/filters, and reports its own failure.
    await openWithDefaults(project);
  }

  async function switchDate(newDate: string, current: OpenableProject, selected: DatasetSelection) {
    setMenu(null);
    if (newDate === selected.date) return;
    const { subject, bucket } = choiceForDate(current.tree, newDate, selected.subject);
    await startOpen(current, newDate, subject, bucket);
  }

  if (!openProject) {
    return <span>no project open</span>;
  }

  const readable = listing?.projects.filter(openable) ?? [];
  const current = readable.find((p) => p.record.id === openProject.id) ?? null;
  // Every recent project still in the listing, named by it, the open one marked as current and
  // inert (hiding it would read as broken in a one-project workspace).
  const isCurrent = (r: OpenableProject) => r.record.id === openProject.id;
  let recent: OpenableProject[] = [];
  let recentProblem: string | null = null;
  try {
    recent = loadRecentProjectIds().flatMap((id) => {
      const project = readable.find((p) => p.record.id === id);
      return project ? [project] : [];
    });
  } catch (e) {
    recentProblem = messageOf(e);
  }

  return (
    <div ref={rootRef} className="relative flex items-center">
      <span title="Your name" className="font-mono">
        {user}
      </span>
      <span className="mx-1.5 text-tcip-border">|</span>
      <button
        onClick={() => setMenu((m) => (m === "project" ? null : "project"))}
        disabled={busy}
        title="Recent projects"
        className="max-w-[16rem] truncate font-mono hover:text-tcip-fg transition-colors"
      >
        {current?.record.display_name ?? "Loading…"}
      </button>
      <span className="mx-1.5 text-tcip-border">|</span>
      {dataset ? (
        <button
          onClick={() => setMenu((m) => (m === "date" ? null : "date"))}
          disabled={busy}
          title="Switch date"
          className="font-mono hover:text-tcip-fg transition-colors"
        >
          {dataset.date}
        </button>
      ) : (
        <span className="font-mono text-tcip-muted">No dated images</span>
      )}
      <span className="mx-1.5 text-tcip-border">|</span>
      <button
        onClick={switchProject}
        disabled={busy}
        title="All projects"
        className="text-tcip-season-1 hover:text-tcip-fg transition-colors"
      >
        Switch project
      </button>

      {menu === "project" && (
        <Dropdown title="Recent projects">
          {recentProblem ? (
            <EmptyRow text={recentProblem} />
          ) : recent.length === 0 ? (
            <EmptyRow text="No recent projects" />
          ) : (
            recent.map((r) => (
              <MenuButton
                key={r.record.id}
                onClick={() => (isCurrent(r) ? setMenu(null) : void openRecent(r))}
                label={r.record.display_name}
                active={isCurrent(r)}
              />
            ))
          )}
          <div className="mt-1 border-t border-tcip-border pt-1">
            <MenuButton onClick={switchProject} label="All projects…" />
          </div>
        </Dropdown>
      )}

      {menu === "date" && dataset && (
        <Dropdown title="Switch date">
          {!current ? (
            <EmptyRow text="Loading…" />
          ) : (
            current.tree.captures.map(({ name }) => (
              <MenuButton
                key={name}
                onClick={() => void switchDate(name, current, dataset)}
                label={name}
                active={name === dataset.date}
              />
            ))
          )}
        </Dropdown>
      )}
    </div>
  );
}

function Dropdown({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="absolute bottom-full right-0 z-30 mb-2 max-h-80 w-64 overflow-auto rounded-md border border-tcip-border bg-tcip-panel p-1.5 shadow-lg">
      <div className="px-1.5 py-1 tcip-label">{title}</div>
      {children}
    </div>
  );
}

function MenuButton({
  onClick,
  label,
  sub,
  active,
}: {
  onClick: () => void;
  label: string;
  sub?: string;
  active?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      className={`block w-full rounded px-1.5 py-1 text-left hover:bg-tcip-bg hover:text-tcip-fg ${
        active ? "text-tcip-fg" : "text-tcip-muted"
      }`}
    >
      <span className="block truncate font-mono">
        {active ? "● " : ""}
        {label}
      </span>
      {sub && <span className="block truncate text-[12px] text-tcip-muted">{sub}</span>}
    </button>
  );
}

function EmptyRow({ text }: { text: string }) {
  return <div className="px-1.5 py-1 text-tcip-muted">{text}</div>;
}
