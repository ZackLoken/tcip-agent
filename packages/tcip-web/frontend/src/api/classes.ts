/** Dataset class-registry + per-image-status API helpers.
 *
 * The registry is one nested mapping per dataset: subject -> {description?, attributes?}. It
 * carries no integer ids and no colors: a label references names, an id is a per-training-run
 * artifact, and a color is GUI-local (see subjectColor). It travels with the image set: a
 * name-based label is undecodable without it.
 */

import { getJson, postJson } from "@/api/http";
import { ROUTES } from "@/api/routes";
import { subjectColorOverride } from "@/lib/subjectColors";

/** One attribute of a subject: categorical (unordered) or ordinal (ranked). ``values`` are the
 *  declared value names, in order (the rank order for an ordinal). */
export interface AttributeDef {
  type: "categorical" | "ordinal";
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

// "negative" = the breeder marked the image Complete with no objects: a confirmed negative,
// recorded in image_status.json. An empty label file alone is not this: it reads as
// "unannotated" until that Complete, which is the whole negative-sample rail.
export type ImageStatus = "complete" | "partial" | "negative" | "unannotated";

/** What a registry save's attribute-vocabulary change did to existing confirmations: for each
 *  affected subject, how many of its confirmations were stamped with the outgoing schema (so a
 *  later read tells them apart from ones made under the new vocabulary), and a warning naming any
 *  the sweep itself could not complete. */
export interface SchemaChangeSweep {
  newly_stamped: Record<string, number>;
  warning: string | null;
}

export const classesApi = {
  // The registry lives in the dataset (not the project), so a shared image set carries its own
  // subject names; pass the dataset_root.
  // The annotations dir lets the server derive a draft registry (detection-only, no attributes)
  // from the labels when no classes.json is saved yet.
  load: (project_root: string, dataset_root?: string | null, annotations_dir?: string | null) => {
    const params = new URLSearchParams({ project_root });
    if (dataset_root) params.set("dataset_root", dataset_root);
    if (annotations_dir) params.set("annotations_dir", annotations_dir);
    return getJson<{ subjects: Registry; version: string | null; unreadable: string[] }>(
      `${ROUTES.getClassesLoad}?${params.toString()}`,
    );
  },

  // `version`: the token `load` returned beside this registry, required on every call.
  // `null` asserts the registry was absent at load, never an unconditional write.
  save: (
    project_root: string,
    subjects: Registry,
    dataset_root: string | null | undefined,
    annotations_dir: string | null | undefined,
    version: string | null,
  ) =>
    postJson<{
      status: string;
      n_subjects: number;
      classes_path: string;
      version: string;
      schema_change_sweep: SchemaChangeSweep;
    }>(ROUTES.postClassesSave, { project_root, subjects, dataset_root, annotations_dir, version }),

  // A Complete is a statement about one subject on one date. Every read and write is scoped to it,
  // so confirming an image while annotating leaf cannot mark it negative for a disease subject
  // nobody has looked at yet. Confirmations are dataset-native (like the registry) rather than
  // project-private, so dataset_root/annotations_dir resolve where the store actually lives,
  // same as classesApi.load/save above.
  loadImageStatus: (
    project_root: string,
    subject: string | null,
    date: string | null,
    dataset_root?: string | null,
    annotations_dir?: string | null,
  ) => {
    const params = new URLSearchParams({ project_root });
    if (subject) params.set("subject", subject);
    if (date) params.set("date", date);
    if (dataset_root) params.set("dataset_root", dataset_root);
    if (annotations_dir) params.set("annotations_dir", annotations_dir);
    return getJson<{ statuses: Record<string, ImageStatus> }>(
      `${ROUTES.getClassesImageStatus}?${params.toString()}`,
    );
  },

  /** `user` is the GUI-set identity; omitting it stamps the backend's process identity instead. */
  setImageStatus: (
    project_root: string,
    image_name: string,
    status: ImageStatus,
    subject: string | null,
    date: string | null,
    dataset_root?: string | null,
    annotations_dir?: string | null,
    user?: string,
  ) =>
    postJson<unknown>(ROUTES.postClassesImageStatus, {
      project_root,
      image_name,
      status,
      subject,
      date,
      dataset_root,
      annotations_dir,
      user,
    }),

  setImageStatusBulk: (
    project_root: string,
    statuses: Record<string, ImageStatus>,
    subject: string | null,
    date: string | null,
    dataset_root?: string | null,
    annotations_dir?: string | null,
    user?: string,
  ) =>
    postJson<unknown>(ROUTES.postClassesImageStatusBulk, {
      project_root,
      statuses,
      subject,
      date,
      dataset_root,
      annotations_dir,
      user,
    }),

  deriveImageStatus: (body: {
    project_root: string;
    annotations_dir: string | null;
    subject: string;
    image_list: string[];
    complete_override?: string[];
  }) =>
    postJson<{ statuses: Record<string, ImageStatus>; unreadable: string[] }>(
      ROUTES.postClassesImageStatusDerive,
      body,
    ),
};

// High-contrast palette the GUI derives subject/value colours from. Colour is GUI-local (the
// registry stores none), so it is a pure function of the name: the same subject renders the same
// colour every session with nothing persisted.
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

/** A subject's colour: this browser's override when it set one (see lib/subjectColors), else the
 *  deterministic name -> hex derivation below. Every consumer calls this one function, so a
 *  recolour reaches the canvas, the toolbar, the legend and the agent's mirror alike without any
 *  of them re-deriving colour on their own. */
export function subjectColor(name: string): string {
  const override = subjectColorOverride(name);
  if (override) return override;
  return derivedSubjectColor(name);
}

/** Deterministic name -> hex: a subject (or attribute-value) name always maps to the same swatch,
 *  without storing colour anywhere. FNV-1a over the name, indexed into the palette. */
export function derivedSubjectColor(name: string): string {
  let h = 2166136261;
  for (let i = 0; i < name.length; i++) {
    h ^= name.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return SUBJECT_COLORS[Math.abs(h) % SUBJECT_COLORS.length];
}
