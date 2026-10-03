/** Dataset subject-registry API helpers.
 *
 * The registry is one nested mapping per dataset: subject -> {description?, attributes?}. It
 * carries no integer ids and no colors: a label references names, an id is a per-training-run
 * artifact, and a color is GUI-local (see subjectColor). It travels with the image set: a
 * name-based label is undecodable without it.
 */

import { getJson, postJson } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type { ATTR_TYPES } from "@/api/types.generated";
import { subjectColorOverride } from "@/lib/subjectColors";

/** An attribute's kind, as the backend's registry declares them. */
export type AttrType = (typeof ATTR_TYPES)[number];

/** One attribute of a subject: categorical (unordered) or ordinal (ranked). ``values`` are the
 *  declared value names, in order (the rank order for an ordinal). */
export interface AttributeDef {
  type: AttrType;
  values: string[];
}

/** A subject entry: a human description plus zero or more attributes. A subject with no attributes
 *  is simply detected (e.g. bush). */
export interface SubjectDef {
  description?: string;
  defined_by?: string;
  defined_at?: string;
  attributes?: Record<string, AttributeDef>;
}

/** The nested registry: subject name -> its definition. Top-level keys are the subjects. */
export type Registry = Record<string, SubjectDef>;

export const subjectsApi = {
  // The registry lives in the dataset (not the project); pass dataset_root so a shared image
  // set carries its own subject names.

  // `subjects` is null when no registry is stored; `discovered` names the subjects the labels
  // under the annotations dir hold either way.
  load: (dataset_root: string, annotations_dir: string | null) => {
    const params = new URLSearchParams({ dataset_root });
    if (annotations_dir) params.set("annotations_dir", annotations_dir);
    return getJson<{
      subjects: Registry | null;
      discovered: string[];
      version: string | null;
      unreadable: string[];
    }>(`${ROUTES.getSubjectsLoad}?${params.toString()}`);
  },

  // `version`: the token `load` returned beside this registry, required on every call.
  // `null` asserts the registry was absent at load, never an unconditional write.
  save: (subjects: Registry, dataset_root: string, version: string | null, user: string) =>
    postJson<{
      status: string;
      n_subjects: number;
      subjects_path: string;
      version: string;
    }>(ROUTES.postSubjectsSave, { subjects, dataset_root, version, user }),
};

// High-contrast palette the GUI derives subject/value colors from. Color is GUI-local (the
// registry stores none), so it is a pure function of the name: the same subject renders the same
// color every session with nothing persisted.
export const SUBJECT_COLORS = [
  "#FF0000",
  "#00FFFF",
  "#FFFF00",
  "#FF00FF",
  "#FF8C00",
  "#00FF00",
  "#FFFFFF",
  "#4169E1",
  "#FF69B4",
  "#00CED1",
];

/** A subject's color: this browser's override (lib/subjectColors), else its collision-free slot
 *  in the loaded registry, else the bare name -> hex derivation below. */
export function subjectColor(name: string): string {
  const override = subjectColorOverride(name);
  if (override) return override;
  const slot = registrySlots.get(name);
  if (slot !== undefined) return SUBJECT_COLORS[slot];
  return derivedSubjectColor(name);
}

function fnv1aSlot(name: string, size: number): number {
  let h = 2166136261;
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % size;
}

/** Deterministic name -> hex: a subject (or attribute-value) name always maps to the same swatch,
 *  without storing color anywhere. FNV-1a over the name, indexed into the palette; a name inside
 *  the loaded registry gets its collision-free slot from `subjectColor` instead. */
export function derivedSubjectColor(name: string): string {
  return SUBJECT_COLORS[fnv1aSlot(name, SUBJECT_COLORS.length)];
}

// The loaded registry's collision-free palette slots, name -> index, recomputed whole on every
// registry load; a name outside it falls back to the bare hash above.
let registrySlots: Map<string, number> = new Map();

/** Assigns each of `subjectNames` its own palette slot where the palette has room: sorted names
 *  take their FNV-1a slot when free, else the next free slot going forward (wrapping past the
 *  last). Past `SUBJECT_COLORS.length` names, free slots run out and later names share one again,
 *  exactly as the bare hash always could; a subject's derived color therefore depends on the
 *  registry it sits in, not on its name alone. */
export function setSubjectColorRegistry(subjectNames: Iterable<string>): void {
  const size = SUBJECT_COLORS.length;
  const sorted = Array.from(new Set(subjectNames)).sort();
  const taken = new Set<number>();
  const slots = new Map<string, number>();
  for (const name of sorted) {
    const start = fnv1aSlot(name, size);
    if (taken.size >= size) {
      slots.set(name, start); // every slot spoken for: sharing is unavoidable past the palette
      continue;
    }
    let slot = start;
    while (taken.has(slot)) slot = (slot + 1) % size;
    taken.add(slot);
    slots.set(name, slot);
  }
  registrySlots = slots;
}
