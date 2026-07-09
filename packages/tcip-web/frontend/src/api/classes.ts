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
  load: (project_root: string) =>
    getJson<{ classes: ClassEntry[] }>(
      `/api/classes/load?project_root=${encodeURIComponent(project_root)}`,
    ),

  save: (project_root: string, classes: ClassEntry[]) =>
    postJson<{ status: string; n_classes: number }>("/api/classes/save", { project_root, classes }),

  autoColor: (class_id: number) =>
    getJson<{ class_id: number; color: string }>(`/api/classes/auto_color/${class_id}`),

  loadImageStatus: (project_root: string) =>
    getJson<{ statuses: Record<string, ImageStatus> }>(
      `/api/classes/image_status?project_root=${encodeURIComponent(project_root)}`,
    ),

  setImageStatus: (project_root: string, image_name: string, status: ImageStatus) =>
    postJson<unknown>("/api/classes/image_status", { project_root, image_name, status }),

  setImageStatusBulk: (project_root: string, statuses: Record<string, ImageStatus>) =>
    postJson<unknown>("/api/classes/image_status/bulk", { project_root, statuses }),

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
