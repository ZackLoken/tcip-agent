/**
 * The front door. Lists the workspace's projects by their own records (display name, site) and
 * opens one: the human never browses the filesystem for two roots. Opening a project makes it the
 * backend's open project; a date/subject/model can be picked per project. The project the backend
 * opened at start (the workspace's last-opened one) auto-opens on first load. Project creation is
 * agent-driven; the user hands the agent data paths rather than hand-structuring a folder here.
 */

import { useEffect, useId, useRef, useState } from "react";

import { api, type ProjectSummary } from "@/api/client";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { SeasonRail } from "@/components/SeasonRail";
import { UNSET_GLYPH } from "@/lib/glyphs";
import {
  defaultDate,
  modelsForDate,
  openProjectById,
  openWorkspaceProject,
  subjectsForDate,
} from "@/lib/openProject";
import { forgetRecentProject } from "@/lib/recentProjects";
import { useStore } from "@/store";

/** A listed project whose record reads, so it carries an id and a display name. */
type OpenableProject = ProjectSummary & { id: string; display_name: string };

const openable = (p: ProjectSummary): p is OpenableProject =>
  p.id !== null && p.display_name !== null;

// Session-scoped: auto-open the backend's open project only on the app's first load, so a later
// "Switch project" (which returns here) doesn't immediately re-open the same project.
let autoOpenAttempted = false;

function RemovalDialog({
  project,
  onClose,
  onRemoved,
}: {
  project: OpenableProject;
  onClose: () => void;
  onRemoved: () => void;
}) {
  const user = useStore((s) => s.user);
  const nameFieldId = useId();
  const [confirmText, setConfirmText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const name = project.display_name;

  async function confirmRemoval() {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await api.projects.remove({ id: project.id, confirm_name: confirmText, user });
      try {
        forgetRecentProject(project.id);
      } catch (e) {
        useStore
          .getState()
          .pushToast(
            `Removed, but the recent projects list was not updated: ${e instanceof Error ? e.message : String(e)}`,
          );
      }
      if (useStore.getState().openProject?.id === project.id) useStore.getState().clearDataset();
      const archiveName = res.archive_path.split(/[/\\]/).filter(Boolean).pop() ?? res.archive_path;
      useStore
        .getState()
        .pushToast(
          `Removed ${name}: archived as ${archiveName} and moved into the workspace's holding ` +
            "directory. Nothing is deleted.",
          "success",
        );
      onRemoved();
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <ConfirmDialog heading={`Remove ${name}`} onClose={onClose} busy={submitting}>
      <div className="flex flex-col gap-3 text-[12px]">
        {submitError && (
          <p className="text-tcip-fp" aria-live="polite">
            {submitError}
          </p>
        )}
        <p className="text-tcip-muted">
          {name} is archived to the workspace&apos;s holding directory and moved there. Nothing is
          deleted; the archive imports back through{" "}
          <span className="font-mono">tcip import-project</span>.
        </p>
        <label className="flex flex-col gap-1" htmlFor={nameFieldId}>
          <span className="tcip-label">Type the project name to confirm</span>
          <input
            id={nameFieldId}
            className="tcip-input"
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            autoComplete="off"
            spellCheck={false}
          />
        </label>
        <div className="flex gap-2">
          <button
            className="tcip-btn-primary flex-1"
            disabled={confirmText !== name || submitting}
            onClick={confirmRemoval}
          >
            {submitting ? "Removing…" : "Remove"}
          </button>
          <button className="tcip-btn flex-1" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </ConfirmDialog>
  );
}

function RenameDialog({
  project,
  onClose,
  onRenamed,
}: {
  project: OpenableProject;
  onClose: () => void;
  onRenamed: () => void;
}) {
  const user = useStore((s) => s.user);
  const newNameFieldId = useId();
  const [newName, setNewName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const name = project.display_name;
  const canConfirm = newName.trim().length > 0 && newName.trim() !== name && !submitting;

  async function confirmRename() {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await api.projects.rename({ id: project.id, display_name: newName, user });
      useStore.getState().pushToast(`Renamed ${name} to ${res.display_name}.`, "success");
      onRenamed();
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <ConfirmDialog heading={`Rename ${name}`} onClose={onClose} busy={submitting}>
      <div className="flex flex-col gap-3 text-[12px]">
        {submitError && (
          <p className="text-tcip-fp" aria-live="polite">
            {submitError}
          </p>
        )}
        <p className="text-tcip-muted">
          The name the project list shows changes; its folder and everything in it stay as they are.
        </p>
        <label className="flex flex-col gap-1" htmlFor={newNameFieldId}>
          <span className="tcip-label">New name</span>
          <input
            id={newNameFieldId}
            className="tcip-input"
            value={newName}
            onChange={(e) => {
              setNewName(e.target.value);
              setSubmitError(null);
            }}
            autoComplete="off"
            spellCheck={false}
          />
        </label>
        <div className="flex gap-2">
          <button
            className="tcip-btn-primary flex-1"
            disabled={!canConfirm}
            onClick={confirmRename}
          >
            {submitting ? "Renaming…" : "Rename"}
          </button>
          <button className="tcip-btn flex-1" onClick={onClose}>
            Cancel
          </button>
        </div>
      </div>
    </ConfirmDialog>
  );
}

function relativeTime(epochSeconds: number): string {
  const deltaMs = Date.now() - epochSeconds * 1000;
  const mins = Math.round(deltaMs / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  const hrs = Math.round(mins / 60);
  if (hrs < 24) return `${hrs} hr ago`;
  const days = Math.round(hrs / 24);
  return `${days} day${days === 1 ? "" : "s"} ago`;
}

export function ProjectPicker() {
  const user = useStore((s) => s.user);
  const setUser = useStore((s) => s.setUser);
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [lastOpenedProblem, setLastOpenedProblem] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [date, setDate] = useState("");
  const [subject, setSubject] = useState("");
  const [model, setModel] = useState("");
  const [opening, setOpening] = useState(false);
  const [openError, setOpenError] = useState<string | null>(null);
  const [removalTarget, setRemovalTarget] = useState<OpenableProject | null>(null);
  const [renameTarget, setRenameTarget] = useState<OpenableProject | null>(null);
  const openedRef = useRef(false);
  const annotatorFieldRef = useRef<HTMLInputElement | null>(null);

  function selectCard(p: OpenableProject) {
    setSelected(p.id);
    const d = defaultDate(p.dates);
    setDate(d);
    setSubject(subjectsForDate(p, d)[0] ?? "");
    setModel(modelsForDate(p, d)[0] ?? "");
    setOpenError(null);
  }

  // Changing date re-scopes the subject/model choices to that date's data: keep the current
  // pick if it's still valid there, else fall to the first available (or none).
  function chooseDate(p: ProjectSummary, newDate: string) {
    setDate(newDate);
    const subjects = subjectsForDate(p, newDate);
    const models = modelsForDate(p, newDate);
    setSubject((prev) => (subjects.includes(prev) ? prev : (subjects[0] ?? "")));
    setModel((prev) => (models.includes(prev) ? prev : (models[0] ?? "")));
  }

  async function openProject(
    p: OpenableProject,
    chosenDate: string,
    chosenSubject: string,
    chosenModel: string,
  ) {
    if (openedRef.current) return;
    if (!chosenDate) {
      // Opening with no date can't satisfy datasetReady, so it would leave the picker on
      // screen with a dead button. Tell the human instead of silently latching.
      setOpenError("This project has no dated images yet, ingest images first.");
      return;
    }
    openedRef.current = true;
    setOpening(true);
    setOpenError(null);
    try {
      await openWorkspaceProject(p, chosenDate, chosenSubject, chosenModel);
    } catch (e) {
      openedRef.current = false;
      setOpenError(String(e));
    } finally {
      setOpening(false);
    }
  }

  function refetch(): Promise<void> {
    return api.projects
      .list()
      .then((res) => {
        setProjects(res.projects);
        setLastOpenedProblem(res.last_opened_problem);
      })
      .catch((e) => {
        setLoadError(e instanceof Error ? e.message : String(e));
      });
  }

  useEffect(() => {
    let canceled = false;
    // Claim the attempt now, before the fetch: a picker that unmounts mid-fetch (every load
    // where the app opens the project itself) must still count as having tried.
    const alreadyAttempted = autoOpenAttempted;
    autoOpenAttempted = true;
    api.projects
      .list()
      .then((res) => {
        if (canceled) return;
        setProjects(res.projects);
        setLastOpenedProblem(res.last_opened_problem);
        if (alreadyAttempted) return;
        // Auto-open the backend's open project on first app load, only when its default date
        // has labeled subjects; else preselect its card.
        const open = res.projects.filter(openable).find((p) => p.id === res.open_id);
        if (!open) return;
        selectCard(open);
        const d = defaultDate(open.dates);
        if (d && subjectsForDate(open, d).length > 0) {
          void openProjectById(open.id).catch((e) => setOpenError(String(e)));
        }
      })
      .catch((e) => {
        if (!canceled) setLoadError(String(e));
      });
    return () => {
      canceled = true;
    };
    // Run once on mount.
  }, []);

  return (
    <div className="h-full w-full overflow-auto bg-gradient-to-b from-tcip-bg to-[#181a12] p-6 flex justify-center">
      <div className="w-full max-w-3xl flex flex-col gap-5">
        <div className="animate-tcip-rise">
          <span className="tcip-eyebrow">Field station</span>
          <h1 className="text-xl font-semibold text-tcip-fg mt-2">Open a project</h1>
        </div>

        <label className="flex flex-col gap-1 animate-tcip-rise">
          <span className="tcip-label">Annotator</span>
          <input
            ref={annotatorFieldRef}
            type="text"
            className="tcip-input max-w-xs"
            placeholder="your name (e.g. jordan)"
            value={user}
            onChange={(e) => setUser(e.target.value)}
            spellCheck={false}
            autoComplete="off"
          />
        </label>

        {loadError && (
          <div className="tcip-panel p-4 text-[12px] text-tcip-fp">
            Could not load projects. {loadError}
          </div>
        )}

        {lastOpenedProblem && (
          <div className="tcip-panel p-4 text-[12px] text-tcip-warn">
            The project opened last could not be opened again: {lastOpenedProblem}
          </div>
        )}

        {projects && projects.length === 0 && (
          <div className="tcip-panel p-6 text-[12px] text-tcip-muted flex flex-col gap-2">
            <span className="text-tcip-fg font-medium">No projects yet</span>
            <span>
              Ask the agent to structure your images into a project; it creates one with{" "}
              <span className="font-mono">initialize_project</span>, given a name and a site.
            </span>
          </div>
        )}

        {projects && projects.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {projects.map((p, index) => {
              if (!openable(p)) {
                return (
                  <div key={p.path} className="tcip-panel p-4 flex flex-col gap-1 text-[11px]">
                    <span className="font-mono text-tcip-fg truncate" title={p.path}>
                      {p.path}
                    </span>
                    <span className="text-tcip-fp">{p.record_problem}</span>
                  </div>
                );
              }
              const isSelected = p.id === selected;
              return (
                <div
                  key={p.id}
                  className={`tcip-panel p-4 flex flex-col gap-2 animate-tcip-rise transition-colors ${
                    isSelected ? "border-tcip-accent" : "hover:border-tcip-border-hover"
                  }`}
                >
                  <button
                    type="button"
                    aria-pressed={isSelected}
                    aria-labelledby={`project-name-${index}`}
                    aria-describedby={`project-desc-${index}`}
                    className="flex flex-col gap-2 w-full text-left cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-tcip-accent/70 focus-visible:ring-offset-1 focus-visible:ring-offset-tcip-bg"
                    onClick={() => selectCard(p)}
                  >
                    <span
                      id={`project-name-${index}`}
                      className="font-medium text-tcip-fg truncate"
                      title={p.display_name}
                    >
                      {p.display_name}
                    </span>
                    <div id={`project-desc-${index}`} className="contents">
                      {p.is_open && (
                        <span className="tcip-badge bg-tcip-accent/20 text-tcip-accent self-start">
                          open
                        </span>
                      )}
                      {p.site && (
                        <span className="text-[11px] text-tcip-muted truncate" title={p.site}>
                          {p.site}
                        </span>
                      )}
                      {p.label_problem && (
                        <span className="text-[11px] text-tcip-fp truncate" title={p.label_problem}>
                          {p.label_problem}
                        </span>
                      )}
                      {/* Signature: the project's captures across the season, each date labeled. */}
                      <SeasonRail
                        dates={p.dates}
                        showLabels
                        active={date || null}
                        className="my-0.5"
                      />
                      <div className="text-[11px] text-tcip-muted flex flex-wrap gap-x-3 gap-y-0.5">
                        <span>{p.image_count} image(s)</span>
                        <span>
                          {p.dates.length} date{p.dates.length === 1 ? "" : "s"}
                        </span>
                        <span>
                          {p.subjects.length} subject{p.subjects.length === 1 ? "" : "s"}
                        </span>
                        <span>
                          {p.models.length} model{p.models.length === 1 ? "" : "s"}
                        </span>
                      </div>
                      <span className="text-[10px] text-tcip-muted">
                        Updated {relativeTime(p.modified)}
                      </span>
                    </div>
                  </button>

                  {isSelected && (
                    <div className="flex flex-col gap-2 pt-2 mt-1 border-t border-tcip-border">
                      <div className="grid grid-cols-3 gap-2">
                        <label className="flex flex-col gap-1">
                          <span className="tcip-label">Date</span>
                          <select
                            className="tcip-select"
                            value={date}
                            onChange={(e) => chooseDate(p, e.target.value)}
                          >
                            {p.dates.length === 0 && <option value="">no dates</option>}
                            {p.dates.map((d) => (
                              <option key={d} value={d}>
                                {d}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="flex flex-col gap-1">
                          <span className="tcip-label">Subject</span>
                          <select
                            className="tcip-select"
                            value={subject}
                            onChange={(e) => setSubject(e.target.value)}
                          >
                            <option
                              value=""
                              aria-label={
                                subjectsForDate(p, date).length ? "no subject chosen" : undefined
                              }
                            >
                              {subjectsForDate(p, date).length ? UNSET_GLYPH : "no labels"}
                            </option>
                            {subjectsForDate(p, date).map((t) => (
                              <option key={t} value={t}>
                                {t}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="flex flex-col gap-1">
                          <span className="tcip-label">Model</span>
                          <select
                            className="tcip-select"
                            value={model}
                            onChange={(e) => setModel(e.target.value)}
                          >
                            <option
                              value=""
                              aria-label={
                                modelsForDate(p, date).length ? "no model chosen" : undefined
                              }
                            >
                              {modelsForDate(p, date).length ? UNSET_GLYPH : "no preds"}
                            </option>
                            {modelsForDate(p, date).map((m) => (
                              <option key={m} value={m}>
                                {m}
                              </option>
                            ))}
                          </select>
                        </label>
                      </div>
                      {openError && <span className="text-[11px] text-tcip-fp">{openError}</span>}
                      <div className="flex items-center gap-2 flex-wrap">
                        <button
                          className="tcip-btn-primary flex-1"
                          disabled={opening || !date}
                          onClick={() => openProject(p, date, subject, model)}
                        >
                          {opening
                            ? "Opening…"
                            : !date
                              ? "This project has no dated images"
                              : "Open project"}
                        </button>
                        <button
                          type="button"
                          className="tcip-btn"
                          onClick={() => setRenameTarget(p)}
                        >
                          Rename…
                        </button>
                        <button
                          type="button"
                          className="tcip-btn"
                          onClick={() => setRemovalTarget(p)}
                        >
                          Remove…
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {!projects && !loadError && (
          <div className="text-[12px] text-tcip-muted">Loading projects…</div>
        )}
      </div>

      {renameTarget && (
        <RenameDialog
          project={renameTarget}
          onClose={() => setRenameTarget(null)}
          onRenamed={() => {
            setRenameTarget(null);
            void refetch();
          }}
        />
      )}

      {removalTarget && (
        <RemovalDialog
          project={removalTarget}
          onClose={() => setRemovalTarget(null)}
          onRemoved={() => {
            setRemovalTarget(null);
            // The dialog's unmount hands focus to the removed card's still-mounted button; the
            // listing update drops it to the body, and only then does it move to a survivor.
            void refetch().then(() => {
              if (document.activeElement === document.body) annotatorFieldRef.current?.focus();
            });
          }}
        />
      )}
    </div>
  );
}
