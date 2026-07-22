/** Class registry + per-image-status API helpers. */

import { getJson, postJson } from "@/api/http";

export interface ClassEntry {
  id: number;
  name: string;
  color: string;
}

// "negative" = the breeder marked the image Complete with no objects — a confirmed negative,
// recorded in image_status.json. An empty label file alone is NOT this: it reads as
// "unannotated" until that Complete, which is the whole negative-sample rail.
export type ImageStatus = "complete" | "partial" | "negative" | "unannotated";

export const classesApi = {
  // Class ids are subject-scoped (catkin=0, bush=0 coexist), and the registry lives in the
  // dataset, not the project — so a shared image set carries its own names. Pass the active
  // subject and the dataset_root; the label dirs let the server derive a provisional map when a
  // subject has no saved one yet.
  load: (
    project_root: string,
    subject?: string | null,
    dataset_root?: string | null,
    annotations_detect_dir?: string | null,
    annotations_segment_dir?: string | null,
  ) => {
    const params = new URLSearchParams({ project_root });
    if (subject) params.set("subject", subject);
    if (dataset_root) params.set("dataset_root", dataset_root);
    if (annotations_detect_dir) params.set("annotations_detect_dir", annotations_detect_dir);
    if (annotations_segment_dir) params.set("annotations_segment_dir", annotations_segment_dir);
    return getJson<{ classes: ClassEntry[] }>(`/api/classes/load?${params.toString()}`);
  },

  save: (
    project_root: string,
    subject: string | null,
    classes: ClassEntry[],
    dataset_root?: string | null,
    annotations_detect_dir?: string | null,
    annotations_segment_dir?: string | null,
  ) =>
    postJson<{ status: string; n_classes: number }>("/api/classes/save", {
      project_root,
      subject,
      classes,
      dataset_root,
      annotations_detect_dir,
      annotations_segment_dir,
    }),

  autoColor: (class_id: number) =>
    getJson<{ class_id: number; color: string }>(`/api/classes/auto_color/${class_id}`),

  // A Complete is a statement about one subject (the annotations/<subject>/ dir) on one date.
  // Every read and write is scoped to it, so confirming an image while annotating catkins cannot
  // mark it negative for a disease subject nobody has looked at yet.
  loadImageStatus: (project_root: string, subject: string | null, date: string | null) => {
    const params = new URLSearchParams({ project_root });
    if (subject) params.set("subject", subject);
    if (date) params.set("date", date);
    return getJson<{ statuses: Record<string, ImageStatus> }>(
      `/api/classes/image_status?${params.toString()}`,
    );
  },

  setImageStatus: (
    project_root: string,
    image_name: string,
    status: ImageStatus,
    subject: string | null,
    date: string | null,
  ) =>
    postJson<unknown>("/api/classes/image_status", {
      project_root,
      image_name,
      status,
      subject,
      date,
    }),

  setImageStatusBulk: (
    project_root: string,
    statuses: Record<string, ImageStatus>,
    subject: string | null,
    date: string | null,
  ) =>
    postJson<unknown>("/api/classes/image_status/bulk", { project_root, statuses, subject, date }),

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
