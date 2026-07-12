/** Session-tracking API helpers (annotation_stats.json on disk). */

import { getJson, postJson } from "@/api/http";

export interface SessionEntry {
  user: string;
  started: string;
  ended: string;
  images_annotated: number;
  total_annotations: number;
  total_time_seconds: number;
  avg_seconds_per_annotation: number;
  images: Record<
    string,
    {
      session_seconds: number;
      loaded_annotation_count: number;
      annotations_added: number;
      final_annotation_count: number;
      avg_seconds_per_annotation: number;
    }
  >;
}

export const sessionsApi = {
  start: (project_root: string, user: string) =>
    postJson<unknown>("/api/sessions/start", { project_root, user }),

  load: (project_root: string) =>
    getJson<{ sessions: SessionEntry[]; image_status: Record<string, string> }>(
      `/api/sessions/load?project_root=${encodeURIComponent(project_root)}`,
    ),

  imageEvent: (body: {
    project_root: string;
    image_name: string;
    session_seconds_delta: number;
    annotations_added_delta: number;
    final_annotation_count: number;
    loaded_annotation_count?: number | null;
  }) => postJson<unknown>("/api/sessions/image_event", body),
};
