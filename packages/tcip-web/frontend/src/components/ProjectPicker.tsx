/**
 * The front door. Lists the workspace's projects by their own records (display name, site) and
 * opens one: the human never browses the filesystem for two roots. Opening a project makes it the
 * backend's open project; a date/subject/model can be picked per project. The chosen project and
 * an open in flight live in the store, so they survive the picker being replaced when the
 * name field's draft is committed (Enter or Open). The project the backend already holds open is
 * preselected unless a project is already chosen. Project
 * creation is agent-driven.
 */

import { useEffect, useId, useRef, useState } from "react";

import { api, type ProjectSummary } from "@/api/client";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { SeasonRail } from "@/components/SeasonRail";
import { UNSET_GLYPH } from "@/lib/glyphs";
import {
  bucketsForDate,
  defaultDate,
  openable,
  type OpenableProject,
  startOpen,
  subjectsForDate,
} from "@/lib/openProject";
import { forgetRecentProject } from "@/lib/recentProjects";
import { useStore } from "@/store";
import { isAnnotatorName, selectAnnotatorNamed } from "@/store/slices/user";

function selectCard(p: OpenableProject) {
  const { patchOpenChoice, patchOpenStatus } = useStore.getState();
  const d = defaultDate(p.dates);
  patchOpenChoice({
    projectId: p.id,
    date: d,
    subject: subjectsForDate(p, d)[0] ?? "",
    bucket: bucketsForDate(p, d)[0] ?? "",
  });
  patchOpenStatus({ openError: null });
}

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
      useStore.getState().pushToast(`Removed ${name}. Archived as ${archiveName}.`, "success");
      onRemoved();
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <ConfirmDialog heading={`Remove ${name}`} onClose={onClose} busy={submitting}>
      <div className="flex flex-col gap-3 text-[13px]">
        {submitError && (
          <p className="tcip-problem" aria-live="polite">
            {submitError}
          </p>
        )}
        <p className="text-tcip-muted">{name} is archived, not deleted.</p>
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
            onClick={() => void confirmRemoval()}
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
      <div className="flex flex-col gap-3 text-[13px]">
        {submitError && (
          <p className="tcip-problem" aria-live="polite">
            {submitError}
          </p>
        )}
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
            onClick={() => void confirmRename()}
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
  const annotatorNamed = useStore(selectAnnotatorNamed);
  const annotatorHintId = useId();
  // The field is a draft: only a committed name (Enter, or Open) reaches the store.
  const [draft, setDraft] = useState(user);
  const draftNamed = isAnnotatorName(draft);
  const [openId, setOpenId] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const { projectId: selected, date, subject, bucket } = useStore((s) => s.openChoice);
  const opening = useStore((s) => s.opening !== null);
  const openError = useStore((s) => s.openError);
  const patchOpenChoice = useStore((s) => s.patchOpenChoice);
  const [removalTarget, setRemovalTarget] = useState<OpenableProject | null>(null);
  const [renameTarget, setRenameTarget] = useState<OpenableProject | null>(null);
  const annotatorFieldRef = useRef<HTMLInputElement | null>(null);

  function commitName() {
    setUser(draft);
  }

  // Changing date re-scopes the subject/bucket choices to that date's data: keep the current
  // pick if it's still valid there, else fall to the first available (or none).
  function chooseDate(p: ProjectSummary, newDate: string) {
    const subjects = subjectsForDate(p, newDate);
    const buckets = bucketsForDate(p, newDate);
    patchOpenChoice({
      date: newDate,
      subject: subjects.includes(subject) ? subject : (subjects[0] ?? ""),
      bucket: buckets.includes(bucket) ? bucket : (buckets[0] ?? ""),
    });
  }

  function showListing(res: Awaited<ReturnType<typeof api.projects.list>>) {
    setProjects(res.projects);
    setOpenId(res.open_id);
  }

  function refetch(): Promise<void> {
    return api.projects
      .list()
      .then(showListing)
      .catch((e) => {
        setLoadError(e instanceof Error ? e.message : String(e));
      });
  }

  useEffect(() => {
    let canceled = false;
    api.projects
      .list()
      .then((res) => {
        if (canceled) return;
        showListing(res);
        // The backend's open project is preselected; nothing opens until the person does.
        const open = res.projects.filter(openable).find((p) => p.id === res.open_id);
        if (open && !useStore.getState().openChoice.projectId) selectCard(open);
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
    <div className="h-full w-full overflow-auto bg-linear-to-b from-tcip-bg to-[#181a12] p-6 flex justify-center">
      <div className="w-full max-w-3xl flex flex-col gap-5">
        <h1 className="text-xl font-semibold text-tcip-fg">Open a project</h1>

        <div className="flex flex-col gap-1">
          <label className="flex flex-col gap-1">
            <span className="tcip-label">Your name</span>
            <input
              ref={annotatorFieldRef}
              type="text"
              className="tcip-input max-w-xs"
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && draftNamed) commitName();
              }}
              spellCheck={false}
              autoComplete="off"
              aria-invalid={!annotatorNamed}
              aria-describedby={annotatorNamed ? undefined : annotatorHintId}
            />
          </label>
          {!annotatorNamed && (
            <span id={annotatorHintId} className="text-tcip-muted">
              Enter your name.
            </span>
          )}
        </div>

        {loadError && (
          <div className="tcip-panel p-4 text-[13px] tcip-problem">
            Could not load projects. {loadError}
          </div>
        )}

        {projects && projects.length === 0 && (
          <div className="tcip-panel p-6 text-tcip-muted">No projects.</div>
        )}

        {projects && projects.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {projects.map((p, index) => {
              if (!openable(p)) {
                return (
                  <div key={p.path} className="tcip-panel p-4 flex flex-col gap-1 text-[13px]">
                    <span className="font-mono text-tcip-fg truncate" title={p.path}>
                      {p.path}
                    </span>
                    <span className="tcip-problem">{p.record_problem}</span>
                  </div>
                );
              }
              const isSelected = p.id === selected;
              return (
                <div
                  key={p.id}
                  className={`tcip-panel p-4 flex flex-col gap-2 transition-colors ${
                    isSelected ? "border-tcip-accent" : "hover:border-tcip-border-hover"
                  }`}
                >
                  <button
                    type="button"
                    aria-pressed={isSelected}
                    aria-labelledby={`project-name-${index}`}
                    aria-describedby={`project-desc-${index}`}
                    className="flex flex-col gap-2 w-full text-left cursor-pointer focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-tcip-accent/70 focus-visible:ring-offset-1 focus-visible:ring-offset-tcip-bg"
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
                      {p.id === openId && (
                        <span className="tcip-badge bg-tcip-accent/20 text-tcip-accent self-start">
                          open
                        </span>
                      )}
                      {p.site && (
                        <span className="text-[13px] text-tcip-muted truncate" title={p.site}>
                          {p.site}
                        </span>
                      )}
                      {p.label_problem && (
                        <span className="text-[13px] tcip-problem truncate" title={p.label_problem}>
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
                      <div className="text-[13px] text-tcip-muted flex flex-wrap gap-x-3 gap-y-0.5">
                        <span>{p.image_count} image(s)</span>
                        <span>
                          {p.dates.length} date{p.dates.length === 1 ? "" : "s"}
                        </span>
                        <span>
                          {p.subjects.length} subject{p.subjects.length === 1 ? "" : "s"}
                        </span>
                        <span>{Object.values(p.buckets_by_date).flat().length} bucket(s)</span>
                      </div>
                      <span className="text-[12px] text-tcip-muted">
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
                            onChange={(e) => patchOpenChoice({ subject: e.target.value })}
                          >
                            <option
                              value=""
                              aria-label={
                                subjectsForDate(p, date).length ? "no subject chosen" : undefined
                              }
                            >
                              {subjectsForDate(p, date).length ? UNSET_GLYPH : "none"}
                            </option>
                            {subjectsForDate(p, date).map((t) => (
                              <option key={t} value={t}>
                                {t}
                              </option>
                            ))}
                          </select>
                        </label>
                        <label className="flex flex-col gap-1">
                          <span className="tcip-label">Predictions</span>
                          <select
                            className="tcip-select"
                            value={bucket}
                            onChange={(e) => patchOpenChoice({ bucket: e.target.value })}
                          >
                            <option
                              value=""
                              aria-label={
                                bucketsForDate(p, date).length ? "no predictions chosen" : undefined
                              }
                            >
                              {bucketsForDate(p, date).length ? UNSET_GLYPH : "none"}
                            </option>
                            {bucketsForDate(p, date).map((name) => (
                              <option key={name} value={name}>
                                {name}
                              </option>
                            ))}
                          </select>
                        </label>
                      </div>
                      {openError?.projectId === p.id && (
                        <span className="text-[13px] tcip-problem">{openError.message}</span>
                      )}
                      <div className="flex items-center gap-2 flex-wrap">
                        <button
                          className="tcip-btn-primary flex-1"
                          disabled={opening || !date || !draftNamed}
                          onClick={() => {
                            commitName();
                            void startOpen(p, date, subject, bucket);
                          }}
                        >
                          {opening ? "Opening…" : !date ? "No dated images" : "Open"}
                        </button>
                        <fieldset disabled={!annotatorNamed} className="contents">
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
                        </fieldset>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {!projects && !loadError && <div className="text-tcip-muted">Loading projects…</div>}

        {projects && (
          <p className="text-tcip-muted">The agent creates projects. Ask it in the terminal.</p>
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
