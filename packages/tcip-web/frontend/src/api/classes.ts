/** Class registry + per-image-status API helpers. */

import { getJson, postJson } from "@/api/http";

export interface ClassEntry {
  id: number;
  name: string;
  color: string;
}

// "negative" = the annotator reviewed the image and recorded no objects (an empty
// label file exists on disk = a valid negative), distinct from "unannotated" (no
// label file — never looked at).
export type ImageStatus = "complete" | "partial" | "negative" | "unannotated";

export const classesApi = {
  // Class ids are trait-scoped (catkin=0, bush=0 coexist). Pass the active trait; the label dirs
  // let the server derive a provisional map from labels when a trait has no saved one yet.
  load: (
    project_root: string,
    trait?: string | null,
    annotations_detect_dir?: string | null,
    annotations_segment_dir?: string | null,
  ) => {
    const params = new URLSearchParams({ project_root });
    if (trait) params.set("trait", trait);
    if (annotations_detect_dir) params.set("annotations_detect_dir", annotations_detect_dir);
    if (annotations_segment_dir) params.set("annotations_segment_dir", annotations_segment_dir);
    return getJson<{ classes: ClassEntry[] }>(`/api/classes/load?${params.toString()}`);
  },

  save: (project_root: string, trait: string | null, classes: ClassEntry[]) =>
    postJson<{ status: string; n_classes: number }>("/api/classes/save", {
      project_root,
      trait,
      classes,
    }),

  autoColor: (class_id: number) =>
    getJson<{ class_id: number; color: string }>(`/api/classes/auto_color/${class_id}`),

  // A Complete is a statement about one campaign (the annotations/<campaign>/ dir) on one date.
  // Every read and write is scoped to it, so confirming an image while annotating catkins cannot
  // mark it negative for a disease campaign nobody has looked at yet.
  loadImageStatus: (project_root: string, campaign: string | null, date: string | null) => {
    const params = new URLSearchParams({ project_root });
    if (campaign) params.set("campaign", campaign);
    if (date) params.set("date", date);
    return getJson<{ statuses: Record<string, ImageStatus> }>(
      `/api/classes/image_status?${params.toString()}`,
    );
  },

  setImageStatus: (
    project_root: string,
    image_name: string,
    status: ImageStatus,
    campaign: string | null,
    date: string | null,
  ) =>
    postJson<unknown>("/api/classes/image_status", {
      project_root,
      image_name,
      status,
      campaign,
      date,
    }),

  setImageStatusBulk: (
    project_root: string,
    statuses: Record<string, ImageStatus>,
    campaign: string | null,
    date: string | null,
  ) =>
    postJson<unknown>("/api/classes/image_status/bulk", { project_root, statuses, campaign, date }),

  deriveImageStatus: (body: {
    project_root: string;
    annotations_detect_dir: string | null;
    annotations_segment_dir: string | null;
    image_list: string[];
    complete_override?: string[];
  }) =>
    postJson<{ statuses: Record<string, ImageStatus> }>("/api/classes/image_status/derive", body),
};

// High-contrast palette — matches backend DEFAULT_CLASS_COLORS so JS-side
// auto-color lookups stay in sync with server-written classes.json.
export const DEFAULT_CLASS_COLORS = [
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

export function autoColor(classId: number): string {
  return DEFAULT_CLASS_COLORS[
    ((classId % DEFAULT_CLASS_COLORS.length) + DEFAULT_CLASS_COLORS.length) %
      DEFAULT_CLASS_COLORS.length
  ];
}
