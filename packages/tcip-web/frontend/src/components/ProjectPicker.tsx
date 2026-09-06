/**
 * The front door. Lists the projects the agent built under the workspace and opens one:
 * the human never browses the filesystem for two roots. Opening a project points the GUI
 * at it (project root = dataset root); a date/subject/model can be picked per project. The
 * active project (set by the agent after ingesting, or by the human here) auto-opens on
 * first load. Project creation is agent-driven (ingest_images); the user hands the agent
 * data paths rather than hand-structuring a folder here.
 */

import { useEffect, useRef, useState } from "react";

import { api, type ProjectSummary } from "@/api/client";
import { SeasonRail } from "@/components/SeasonRail";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { adoptWorkspaceProject, defaultDate, openProjectByName } from "@/lib/openProject";
import { useStore } from "@/store";

// Session-scoped: auto-open the active project only on the app's first load, so a later
// "Switch project" (which returns here) doesn't immediately re-open the same project.
let autoOpenAttempted = false;

// The subjects/models that actually have data on a given date. Empty when nothing is
// labelled/predicted there: the selectors show only these, so a date with no bush
// labels won't offer "bush" (which would open a blank canvas).
const subjectsForDate = (p: ProjectSummary, d: string): string[] => p.subjects_by_date[d] ?? [];
const modelsForDate = (p: ProjectSummary, d: string): string[] => p.models_by_date[d] ?? [];

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
  const [selected, setSelected] = useState<string | null>(null);
  const [date, setDate] = useState("");
  const [subject, setSubject] = useState("");
  const [model, setModel] = useState("");
  const [opening, setOpening] = useState(false);
  const [openError, setOpenError] = useState<string | null>(null);
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
            {projects.map((p) => {
              const isSelected = p.name === selected;
              return (
                <div
                  key={p.name}
                  role="button"
                  tabIndex={0}
                  aria-pressed={isSelected}
                  className={`tcip-panel p-4 flex flex-col gap-2 cursor-pointer transition-colors animate-tcip-rise focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-tcip-accent/70 focus-visible:ring-offset-1 focus-visible:ring-offset-tcip-bg ${
                    isSelected ? "border-tcip-accent" : "hover:border-tcip-border-hover"
                  }`}
                  onClick={() => selectCard(p)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      selectCard(p);
                    }
                  }}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium text-tcip-fg truncate" title={p.name}>
                      {p.name}
                    </span>
                    {p.is_active && (
                      <span className="tcip-badge bg-tcip-accent/20 text-tcip-accent">active</span>
                    )}
                  </div>
                  {p.site ? (
                    <span className="text-[11px] text-tcip-muted truncate" title={p.site}>
                      {p.site}
                    </span>
                  ) : (
                    p.site_problem && (
                      <span className="text-[11px] text-tcip-fp truncate" title={p.site_problem}>
                        {p.site_problem}
                      </span>
                    )
                  )}
                  {p.label_problem && (
                    <span className="text-[11px] text-tcip-fp truncate" title={p.label_problem}>
                      {p.label_problem}
                    </span>
                  )}
                  {/* Signature: the project's captures across the season, each date labelled. */}
                  <SeasonRail dates={p.dates} showLabels active={date || null} className="my-0.5" />
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

                  {isSelected && (
                    <div
                      className="flex flex-col gap-2 pt-2 mt-1 border-t border-tcip-border"
                      onClick={(e) => e.stopPropagation()}
                    >
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
                      <button
                        className="tcip-btn-primary"
                        disabled={opening || !date}
                        onClick={() => openProject(p, date, subject, model)}
                      >
                        {opening
                          ? "Opening…"
                          : !date
                            ? "This project has no dated images"
                            : "Open project"}
                      </button>
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
    </div>
  );
}
