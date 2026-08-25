/** Session-tracking API helpers (annotation_stats.json on disk). */

import { getJson, postJson } from "@/api/http";
import { ROUTES } from "@/api/routes";

export interface SessionEntry {
  user: string;
  started: string;
  ended: string;
  images_annotated: number;
  total_annotations: number;
  total_time_seconds: number;
  avg_seconds_per_annotation: number;
  // Read-time split of total_time_seconds against image_status.json's current state, not frozen
  // at image_event time: negative_confirmation_seconds + review_seconds + new_annotation_seconds
  // sums back to total_time_seconds.
  negative_confirmation_seconds: number;
  review_seconds: number;
  new_annotation_seconds: number;
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
    postJson<unknown>(ROUTES.postSessionsStart, { project_root, user }),

  load: (project_root: string) =>
    getJson<{ sessions: SessionEntry[] }>(
      `${ROUTES.getSessionsLoad}?project_root=${encodeURIComponent(project_root)}`,
    ),

  imageEvent: (body: {
    project_root: string;
    image_name: string;
    session_seconds_delta: number;
    annotations_added_delta: number;
    final_annotation_count: number;
    loaded_annotation_count?: number | null;
    // Where this image's image_status.json entry lives, so a later read can classify this
    // time as review vs. negative-confirmation vs. new-annotation work.
    dataset_root?: string | null;
    subject?: string | null;
    date?: string | null;
  }) => postJson<unknown>(ROUTES.postSessionsImageEvent, body),
};
