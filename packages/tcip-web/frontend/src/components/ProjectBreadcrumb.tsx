/**
 * Status-bar project breadcrumb: three fast-tracks in the lower-right corner —
 *   project name → a dropdown of recent projects (jump straight in),
 *   date        → a dropdown of this project's dates (switch without the workspace),
 *   Switch Project → the full workspace (all projects).
 * The date switch re-opens the same project on another date, keeping the current trait/model
 * when that date has them. (Per-date view/position restore is layered on separately.)
 */

import { useEffect, useRef, useState, type ReactNode } from "react";

import { api, type ProjectSummary } from "@/api/client";
import { openProjectByName, openWorkspaceProject } from "@/lib/openProject";
import { loadRecentProjects } from "@/lib/recentProjects";
import { useStore } from "@/store";

const traitsForDate = (p: ProjectSummary, d: string): string[] => p.traits_by_date[d] ?? [];
const modelsForDate = (p: ProjectSummary, d: string): string[] => p.models_by_date[d] ?? [];

/** The model bucket the current predictions dir points at (predictions/<model>/<date>/…), or null. */
function currentModel(predDetectDir: string | null, predSegmentDir: string | null): string | null {
  const dir = predDetectDir ?? predSegmentDir;
  if (!dir) return null;
  const after = dir.split(/[/\\]predictions[/\\]/)[1];
  return after ? (after.split(/[/\\]/)[0] ?? null) : null;
}

export function ProjectBreadcrumb() {
  const dataset = useStore((s) => s.gui.dataset);
  const patchGui = useStore((s) => s.patchGui);
  const clearDataset = useStore((s) => s.clearDataset);
  const activeTab = useStore((s) => s.gui.active_tab);
  const setActiveTab = useStore((s) => s.setActiveTab);
  const pushToast = useStore((s) => s.pushToast);

  const [menu, setMenu] = useState<null | "project" | "date">(null);
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [busy, setBusy] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  // Load the project list when a menu first opens — powers the date switcher and resolves recent
  // project names. On-demand so the status bar never polls.
  useEffect(() => {
    if (!menu || projects) return;
    let cancelled = false;
    api.projects
      .list()
      .then((r) => {
        if (!cancelled) setProjects(r.projects);
      })
      .catch(() => {
        if (!cancelled) pushToast("Could not load the project list.");
      });
    return () => {
      cancelled = true;
    };
  }, [menu, projects, pushToast]);

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
    if (!["annotate", "review", "results"].includes(activeTab)) setActiveTab("annotate");
  }

  async function openRecent(name: string) {
    setMenu(null);
    setBusy(true);
    try {
      const sel = await openProjectByName(name);
      if (sel) patchGui({ dataset: sel });
      else pushToast("That project is no longer in the workspace.");
    } catch (e) {
      pushToast(`Could not open project: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  async function switchDate(newDate: string, current: ProjectSummary) {
    setMenu(null);
    if (newDate === dataset.date) return;
    setBusy(true);
    try {
      // Keep the current trait/model when the new date has them, else fall to that date's first.
      const traits = traitsForDate(current, newDate);
      const models = modelsForDate(current, newDate);
      const trait =
        dataset.annotation_type && traits.includes(dataset.annotation_type)
          ? dataset.annotation_type
          : (traits[0] ?? null);
      const curModel = currentModel(
        dataset.predictions_detect_dir,
        dataset.predictions_segment_dir,
      );
      const model = curModel && models.includes(curModel) ? curModel : (models[0] ?? null);
      const sel = await openWorkspaceProject(current, newDate, trait, model);
      patchGui({ dataset: sel });
    } catch (e) {
      pushToast(`Could not switch date: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setBusy(false);
    }
  }

  if (!dataset.dataset_root || !dataset.date) {
    return <span>no project open</span>;
  }

  const projectName = dataset.dataset_root.split(/[/\\]/).slice(-1)[0];
  const current = projects?.find((p) => p.path === dataset.project_root) ?? null;
  const recent = loadRecentProjects().filter((r) => r.path !== dataset.project_root);

  return (
    <div ref={rootRef} className="relative flex items-center">
      <button
        onClick={() => setMenu((m) => (m === "project" ? null : "project"))}
        disabled={busy}
        title="Recent projects"
        className="max-w-[16rem] truncate font-mono hover:text-tcip-fg transition-colors"
      >
        {projectName}
      </button>
      <span className="mx-1.5 text-tcip-border">|</span>
      <button
        onClick={() => setMenu((m) => (m === "date" ? null : "date"))}
        disabled={busy}
        title="Switch date"
        className="font-mono hover:text-tcip-fg transition-colors"
      >
        {dataset.date}
      </button>
      <span className="mx-1.5 text-tcip-border">|</span>
      <button
        onClick={switchProject}
        disabled={busy}
        title="Open the full project list"
        className="text-tcip-season-1 hover:text-tcip-fg transition-colors"
      >
        Switch Project
      </button>

      {menu === "project" && (
        <Dropdown title="Recent projects">
          {recent.length === 0 ? (
            <EmptyRow text="No other recent projects" />
          ) : (
            recent.map((r) => (
              <MenuButton
                key={r.path}
                onClick={() => void openRecent(r.name)}
                label={r.name}
                sub={r.path}
              />
            ))
          )}
          <div className="mt-1 border-t border-tcip-border pt-1">
            <MenuButton onClick={switchProject} label="All projects…" />
          </div>
        </Dropdown>
      )}

      {menu === "date" && (
        <Dropdown title="Switch date">
          {!current ? (
            <EmptyRow text="Loading…" />
          ) : (
            current.dates.map((d) => (
              <MenuButton
                key={d}
                onClick={() => void switchDate(d, current)}
                label={d}
                active={d === dataset.date}
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
      <div className="px-1.5 py-1 text-[10px] font-semibold uppercase tracking-wide text-tcip-muted">
        {title}
      </div>
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
      {sub && <span className="block truncate text-[10px] text-tcip-muted">{sub}</span>}
    </button>
  );
}

function EmptyRow({ text }: { text: string }) {
  return <div className="px-1.5 py-1 text-tcip-muted">{text}</div>;
}
