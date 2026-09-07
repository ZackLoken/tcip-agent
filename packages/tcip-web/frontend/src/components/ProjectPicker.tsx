/**
 * The front door. Lists the projects the agent built under the workspace and opens one:
 * the human never browses the filesystem for two roots. Opening a project points the GUI
 * at it (project root = dataset root); a date/subject/model can be picked per project. The
 * active project (set by the agent after ingesting, or by the human here) auto-opens on
 * first load. Project creation is agent-driven (ingest_images); the user hands the agent
 * data paths rather than hand-structuring a folder here.
 */

import { useEffect, useId, useRef, useState } from "react";

import {
  api,
  type PendingRemovalEntry,
  type ProjectSummary,
  type RemovalOutcome,
} from "@/api/client";
import type {
  DependencyWarning,
  DependentProject,
  ExternalRoot,
  RemovalPreview,
} from "@/api/types.generated";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { SeasonRail } from "@/components/SeasonRail";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { adoptWorkspaceProject, defaultDate, openProjectByName } from "@/lib/openProject";
import { forgetRecentProject } from "@/lib/recentProjects";
import { useStore } from "@/store";

// Session-scoped: auto-open the active project only on the app's first load, so a later
// "Switch project" (which returns here) doesn't immediately re-open the same project.
let autoOpenAttempted = false;

// The subjects/models with data on a given date; empty when nothing is labelled/predicted
// there, so the selectors never offer a choice that would open a blank canvas.
const subjectsForDate = (p: ProjectSummary, d: string): string[] => p.subjects_by_date[d] ?? [];
const modelsForDate = (p: ProjectSummary, d: string): string[] => p.models_by_date[d] ?? [];

// The C library's errno values, the ones Python's errno module reports and the backend emits.
const EPERM = 1;
const EACCES = 13;
const EXDEV = 18;

function RemovalDialog({
  name,
  onClose,
  onRemoved,
  onRefetchListing,
}: {
  name: string;
  onClose: () => void;
  onRemoved: () => void;
  onRefetchListing: () => void;
}) {
  const user = useStore((s) => s.user);
  const nameFieldId = useId();
  const [preview, setPreview] = useState<RemovalPreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [confirmText, setConfirmText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [releasing, setReleasing] = useState(false);
  const [releaseError, setReleaseError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.projects
      .removalPreview(name)
      .then((p) => {
        if (!cancelled) setPreview(p);
      })
      .catch((e) => {
        if (!cancelled) setPreviewError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [name]);

  function loadPreview() {
    return api.projects
      .removalPreview(name)
      .then((p) => {
        setPreview(p);
        setPreviewError(null);
      })
      .catch((e) => {
        setPreviewError(e instanceof Error ? e.message : String(e));
      });
  }

  // A failed preview never settles (confirm stays disabled) but does end the check, so the
  // dialog stops being busy; the name field is never disabled, a refusal only blocks confirm.
  const previewSettled = preview !== null;
  const checking = !previewSettled && previewError === null;
  const refusal = submitError ?? preview?.refusal ?? null;
  const canConfirm = previewSettled && !refusal && confirmText === name && !submitting;

  async function confirmRemoval() {
    setSubmitting(true);
    setSubmitError(null);
    try {
      const res = await api.projects.remove({ name, confirm_name: confirmText, user });
      forgetRecentProject(name);
      const archiveName = res.archive_path.split(/[/\\]/).filter(Boolean).pop() ?? res.archive_path;
      const suffix = res.recorded_in_open_project
        ? ""
        : ` This backend has no project open, so the request is recorded in ${name}'s own log.`;
      useStore
        .getState()
        .pushToast(
          `Removal requested: ${name} is archived at ${archiveName} under the workspace's ` +
            "holding directory and moves beside it at the next backend start. Nothing is " +
            `deleted.${suffix}`,
          "success",
        );
      onRemoved();
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : String(e));
      onRefetchListing();
    } finally {
      setSubmitting(false);
    }
  }

  async function releaseBinding() {
    setReleasing(true);
    setReleaseError(null);
    try {
      await api.projects.releaseBinding(name, user);
      useStore
        .getState()
        .pushToast(
          `Released ${name}: it no longer opens by default and the GUI no longer counts it open.`,
          "success",
        );
      await loadPreview();
      onRefetchListing();
    } catch (e) {
      setReleaseError(e instanceof Error ? e.message : String(e));
    } finally {
      setReleasing(false);
    }
  }

  function dependentLine(d: DependentProject): string {
    if (d.unreadable) {
      return `${d.project}: its dataset registry could not be read (${d.unreadable})`;
    }
    return (
      `${d.project} registers images from this project as dataset ${d.dataset_id}; its ` +
      "training fails on a missing image once the move completes"
    );
  }

  function externalLine(r: ExternalRoot): string {
    return `${r.path} (${r.present === false ? "no longer present" : "stays in place"})`;
  }

  return (
    <ConfirmDialog heading={`Remove ${name}`} onClose={onClose} busy={checking}>
      <div className="flex flex-col gap-3 text-[12px]">
        <div className="flex flex-col gap-3" aria-live="polite">
          {checking && (
            <p className="text-tcip-muted">Checking this project&apos;s dependents and refusals…</p>
          )}
          {previewError && (
            <p className="text-tcip-warn">
              This project&apos;s dependents and refusals could not be checked: {previewError}.
              Close and try again.
            </p>
          )}
          {refusal && <p className="text-tcip-fp">{refusal}</p>}
          {preview?.releasable && (
            <div className="flex flex-col gap-1">
              <button
                type="button"
                className="tcip-btn self-start"
                disabled={releasing}
                onClick={releaseBinding}
              >
                {releasing ? "Releasing…" : `Release ${name}`}
              </button>
              <p className="text-tcip-muted">
                Stops it opening by default and forgets it as the GUI&apos;s open project; nothing
                else changes.
              </p>
              {releaseError && <p className="text-tcip-fp">{releaseError}</p>}
            </div>
          )}
          {preview?.external_roots_unreadable && (
            <p className="text-tcip-fp">
              External roots could not be read: {preview.external_roots_unreadable}
            </p>
          )}
          {preview && preview.dependent_projects.length > 0 && (
            <div className="text-tcip-fp">
              <p className="font-medium">
                Other projects depend on this one; their training fails on a missing image once the
                move completes:
              </p>
              <ul className="list-disc pl-4">
                {preview.dependent_projects.map((d, i) => (
                  <li key={i}>{dependentLine(d)}</li>
                ))}
              </ul>
            </div>
          )}
          {preview && preview.external_roots.length > 0 && (
            <div className="text-tcip-muted">
              <p className="font-medium">External roots:</p>
              <ul className="list-disc pl-4">
                {preview.external_roots.map((r, i) => (
                  <li key={i}>{externalLine(r)}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
        <p className="text-tcip-muted">
          {name} is archived now to the workspace&apos;s holding directory and moved there at the
          next backend start. Nothing is deleted; the archive imports back through{" "}
          <span className="font-mono">tcip import-project</span>, or by hand by moving{" "}
          <span className="font-mono">{name}-&lt;stamp&gt;/</span> back under the workspace as{" "}
          <span className="font-mono">{name}</span>.
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
            disabled={!canConfirm}
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

// requested_at is the compact UTC form (YYYYMMDDTHHMMSSZ); render it in the viewer's own
// timezone, falling back to the raw stamp when it doesn't parse.
export function localTime(compactUtc: string): string {
  const m = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/.exec(compactUtc);
  if (!m) return compactUtc;
  const [, y, mo, d, h, mi, s] = m.map(Number);
  const parsed = new Date(Date.UTC(y, mo - 1, d, h, mi, s));
  return parsed.toLocaleString();
}

function holdingDirName(path: string): string {
  const parts = path.split(/[/\\]/).filter(Boolean);
  return parts[parts.length - 1] ?? path;
}

function dependencyWarningLine(w: DependencyWarning): string {
  if (w.present) {
    return (
      `Depends on ${w.target} (dataset ${w.dataset_id}), which is pending removal; its images ` +
      "move to the workspace's holding directory at the next backend start, and this warning " +
      "clears once the dataset is registered again from where they are then"
    );
  }
  return (
    `Depends on ${w.target} (dataset ${w.dataset_id}), which is no longer in the workspace; ` +
    "this warning clears once the dataset is registered again from where its images now are"
  );
}

function removalOutcomeLine(o: RemovalOutcome): string {
  if (o.moved_to) {
    return `${o.name} moved to the workspace's holding directory (${holdingDirName(o.moved_to)}).`;
  }
  if (o.blocked_by) {
    if (o.blocked_errno === EACCES || o.blocked_errno === EPERM) {
      return (
        `${o.name}: the move at this start was refused, another process still holds its files ` +
        `(${o.blocked_by}); it stays pending and moves at the next start after that process exits.`
      );
    }
    if (o.blocked_errno === EXDEV) {
      return (
        `${o.name}: the move at this start was refused, it cannot move across filesystems ` +
        `(${o.blocked_by}); it stays pending until moved by hand.`
      );
    }
    return `${o.name}: blocked (${o.blocked_by})`;
  }
  return `${o.name}: skipped (${o.skipped})`;
}

export function ProjectPicker() {
  const user = useStore((s) => s.user);
  const setUser = useStore((s) => s.setUser);
  const [projects, setProjects] = useState<ProjectSummary[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [date, setDate] = useState("");
  const [subject, setSubject] = useState("");
  const [model, setModel] = useState("");
  const [opening, setOpening] = useState(false);
  const [openError, setOpenError] = useState<string | null>(null);
  const [pendingRemoval, setPendingRemoval] = useState<PendingRemovalEntry[]>([]);
  const [removalOutcomes, setRemovalOutcomes] = useState<RemovalOutcome[]>([]);
  const [removalTarget, setRemovalTarget] = useState<string | null>(null);
  const openedRef = useRef(false);

  function selectCard(p: ProjectSummary) {
    setSelected(p.name);
    const d = defaultDate(p.dates);
    setDate(d);
    setSubject(subjectsForDate(p, d)[0] ?? "");
    setModel(modelsForDate(p, d)[0] ?? "");
    setOpenError(null);
  }

  // Changing date re-scopes the subject/model choices to that date's available data:
  // keep the current pick if it's still valid there, else fall to the first available
  // (or none, which the "Open project" flow handles as no-annotations).
  function chooseDate(p: ProjectSummary, newDate: string) {
    setDate(newDate);
    const subjects = subjectsForDate(p, newDate);
    const models = modelsForDate(p, newDate);
    setSubject((prev) => (subjects.includes(prev) ? prev : (subjects[0] ?? "")));
    setModel((prev) => (models.includes(prev) ? prev : (models[0] ?? "")));
  }

  async function openProject(
    p: ProjectSummary,
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
      // Marks it active so it auto-opens next time and other clients agree; a rejected
      // write surfaces as a toast rather than failing this open.
      await adoptWorkspaceProject(p, chosenDate, chosenSubject, chosenModel);
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
        setPendingRemoval(res.pending_removal);
        setRemovalOutcomes(res.removal_startup_outcomes);
      })
      .catch((e) => {
        setLoadError(e instanceof Error ? e.message : String(e));
      });
  }

  useEffect(() => {
    let cancelled = false;
    // Claim the attempt now, before the fetch: a picker that unmounts mid-fetch (every load
    // where the app opens the project itself) must still count as having tried.
    const alreadyAttempted = autoOpenAttempted;
    autoOpenAttempted = true;
    api.projects
      .list()
      .then((res) => {
        if (cancelled) return;
        setProjects(res.projects);
        setPendingRemoval(res.pending_removal);
        setRemovalOutcomes(res.removal_startup_outcomes);
        // Auto-open the active project on first app load.
        if (!alreadyAttempted) {
          const active = res.projects.find((p) => p.name === res.active);
          const d = active ? defaultDate(active.dates) : "";
          // Auto-open only when the default date has labelled subjects, else preselect the
          // card; no marker write here, since the app opening what it already names isn't a human adoption.
          if (active && d && subjectsForDate(active, d).length > 0) {
            selectCard(active);
            void openProjectByName(active.name).catch((e) => setOpenError(String(e)));
          } else if (active) {
            selectCard(active);
          }
        }
      })
      .catch((e) => {
        if (!cancelled) setLoadError(String(e));
      });
    return () => {
      cancelled = true;
    };
    // Run once on mount.
  }, []);

  // A project with a blocked outcome from this start already appears under the outcomes list
  // below; the plain "Pending removal" line would only repeat it.
  const blockedThisStart = new Set(removalOutcomes.filter((o) => o.blocked_by).map((o) => o.name));

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

        {projects && projects.length === 0 && (
          <div className="tcip-panel p-6 text-[12px] text-tcip-muted flex flex-col gap-2">
            <span className="text-tcip-fg font-medium">No projects yet</span>
            <span>
              Ask the agent to structure your images into a project; it creates one with{" "}
              <span className="font-mono">ingest_images</span>, given a site.
            </span>
          </div>
        )}

        {projects && projects.length > 0 && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {projects.map((p, index) => {
              const isSelected = p.name === selected;
              return (
                <div
                  key={p.name}
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
                    <div className="flex items-center justify-between gap-2">
                      <span
                        id={`project-name-${index}`}
                        className="font-medium text-tcip-fg truncate"
                        title={p.name}
                      >
                        {p.name}
                      </span>
                      {p.is_active && (
                        <span className="tcip-badge bg-tcip-accent/20 text-tcip-accent">
                          active
                        </span>
                      )}
                    </div>
                    <div id={`project-desc-${index}`} className="contents">
                      {p.site ? (
                        <span className="text-[11px] text-tcip-muted truncate" title={p.site}>
                          {p.site}
                        </span>
                      ) : (
                        p.site_problem && (
                          <span
                            className="text-[11px] text-tcip-fp truncate"
                            title={p.site_problem}
                          >
                            {p.site_problem}
                          </span>
                        )
                      )}
                      {p.label_problem && (
                        <span className="text-[11px] text-tcip-fp truncate" title={p.label_problem}>
                          {p.label_problem}
                        </span>
                      )}
                      {p.dependency_warnings.map((w, i) => (
                        <span key={i} className="text-[11px] text-tcip-fp">
                          {dependencyWarningLine(w)}
                        </span>
                      ))}
                      {p.dependency_problem && (
                        <span className="text-[11px] text-tcip-fp">
                          its dataset registry could not be read ({p.dependency_problem})
                        </span>
                      )}
                      {/* Signature: the project's captures across the season, each date labelled. */}
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
                        {(() => {
                          const reason = p.removal_refusal;
                          const reasonId = `remove-reason-${p.name}`;
                          return (
                            <>
                              <button
                                type="button"
                                className="tcip-btn"
                                disabled={!!reason && !p.removal_releasable}
                                aria-describedby={reason ? reasonId : undefined}
                                onClick={() => setRemovalTarget(p.name)}
                              >
                                Remove…
                              </button>
                              {reason && (
                                <span id={reasonId} className="text-[11px] text-tcip-muted">
                                  {reason}
                                </span>
                              )}
                            </>
                          );
                        })()}
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

        {pendingRemoval.filter((p) => !blockedThisStart.has(p.name)).length > 0 && (
          <div className="text-[11px] text-tcip-muted flex flex-col gap-0.5">
            {pendingRemoval
              .filter((p) => !blockedThisStart.has(p.name))
              .map((p) => (
                <span key={p.name}>
                  Pending removal: {p.name}, requested {localTime(p.requested_at)}; moves at the
                  next backend start.
                </span>
              ))}
          </div>
        )}

        {removalOutcomes.length > 0 && (
          <div className="text-[11px] text-tcip-muted flex flex-col gap-0.5">
            {removalOutcomes.map((o, i) => (
              <span key={`${o.name}-${i}`}>{removalOutcomeLine(o)}</span>
            ))}
          </div>
        )}
      </div>

      {removalTarget && (
        <RemovalDialog
          name={removalTarget}
          onClose={() => setRemovalTarget(null)}
          onRemoved={() => {
            setRemovalTarget(null);
            void refetch();
          }}
          onRefetchListing={() => void refetch()}
        />
      )}
    </div>
  );
}
