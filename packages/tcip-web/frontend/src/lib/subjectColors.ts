import { useEffect, useState } from "react";

/** Per-browser subject colour overrides: a subject name -> hex, never written to the registry
 *  (`classes.json` stores no colour; see `api/classes.ts`'s `subjectColor`). Shaped after
 *  `lib/reviewColors.ts`'s persisted palette: a `localStorage` map plus a same-tab event, so a
 *  recolour reaches every consumer of `subjectColor` without a page reload. */
export type SubjectColorOverrides = Record<string, string>;

const SUBJECT_COLORS_KEY = "tcip.annotate.subjectColors";
const SUBJECT_COLORS_EVENT = "tcip:subject-colors";

export function loadSubjectColorOverrides(): SubjectColorOverrides {
  try {
    const raw = localStorage.getItem(SUBJECT_COLORS_KEY);
    if (raw) return JSON.parse(raw);
  } catch {
    /* storage disabled, fall back to no overrides */
  }
  return {};
}

function saveSubjectColorOverrides(overrides: SubjectColorOverrides): void {
  try {
    localStorage.setItem(SUBJECT_COLORS_KEY, JSON.stringify(overrides));
  } catch {
    /* storage disabled, the override just won't persist */
  }
  window.dispatchEvent(
    new CustomEvent<SubjectColorOverrides>(SUBJECT_COLORS_EVENT, { detail: overrides }),
  );
}

/** This browser's colour override for one subject, or null when none is set (the caller falls
 *  back to the derived hash). */
export function subjectColorOverride(name: string): string | null {
  return loadSubjectColorOverrides()[name] ?? null;
}

export function setSubjectColorOverride(name: string, hex: string): void {
  saveSubjectColorOverrides({ ...loadSubjectColorOverrides(), [name]: hex });
}

/** Reverts one subject to its derived colour (removes the override, never writes a colour). */
export function resetSubjectColorOverride(name: string): void {
  const next = { ...loadSubjectColorOverrides() };
  delete next[name];
  saveSubjectColorOverrides(next);
}

/** Re-renders the calling component whenever any subject's colour override changes in this
 *  browser (a same-tab custom event, since a `storage` event never fires in the tab that wrote
 *  it). Returns a tick a memo can depend on, so a derived value that calls `subjectColor` recomputes
 *  rather than reading a stale render's colour. */
export function useSubjectColors(): number {
  const [tick, setTick] = useState(0);
  useEffect(() => {
    const onChange = () => setTick((t) => t + 1);
    window.addEventListener(SUBJECT_COLORS_EVENT, onChange);
    return () => window.removeEventListener(SUBJECT_COLORS_EVENT, onChange);
  }, []);
  return tick;
}
