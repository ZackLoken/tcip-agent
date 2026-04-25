/** Class registry + per-image-status API helpers. */

export interface ClassEntry {
  id: number;
  name: string;
  color: string;
}

export type ImageStatus = "complete" | "partial" | "unannotated";

export const classesApi = {
  load: (project_root: string) =>
    fetch(`/api/classes/load?project_root=${encodeURIComponent(project_root)}`).then(
      (r) => r.json() as Promise<{ classes: ClassEntry[] }>,
    ),

  save: (project_root: string, classes: ClassEntry[]) =>
    fetch("/api/classes/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_root, classes }),
    }).then((r) => r.json() as Promise<{ status: string; n_classes: number }>),

  autoColor: (class_id: number) =>
    fetch(`/api/classes/auto_color/${class_id}`).then(
      (r) => r.json() as Promise<{ class_id: number; color: string }>,
    ),

  loadImageStatus: (project_root: string) =>
    fetch(
      `/api/classes/image_status?project_root=${encodeURIComponent(project_root)}`,
    ).then((r) => r.json() as Promise<{ statuses: Record<string, ImageStatus> }>),

  setImageStatus: (project_root: string, image_name: string, status: ImageStatus) =>
    fetch("/api/classes/image_status", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_root, image_name, status }),
    }).then((r) => r.json()),

  setImageStatusBulk: (project_root: string, statuses: Record<string, ImageStatus>) =>
    fetch("/api/classes/image_status/bulk", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_root, statuses }),
    }).then((r) => r.json()),

  deriveImageStatus: (body: {
    project_root: string;
    annotations_detect_dir: string | null;
    annotations_segment_dir: string | null;
    image_list: string[];
    complete_override?: string[];
  }) =>
    fetch("/api/classes/image_status/derive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json() as Promise<{ statuses: Record<string, ImageStatus> }>),
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
  return DEFAULT_CLASS_COLORS[((classId % DEFAULT_CLASS_COLORS.length) + DEFAULT_CLASS_COLORS.length) % DEFAULT_CLASS_COLORS.length];
}
