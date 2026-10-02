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
  // Read-time split of total_time_seconds against each label document's current marks, not
  // frozen at image_event time: the three parts sum back to total_time_seconds.
  negative_confirmation_seconds: number;
  review_seconds: number;
  new_annotation_seconds: number;
  images: Record<
    string,
    {
      session_seconds: number;
      annotations_added: number;
      final_annotation_count: number;
      avg_seconds_per_annotation: number;
    }
  >;
}

/** Every call records against the project the backend has open. */
export const sessionsApi = {
  start: (user: string) => postJson<unknown>(ROUTES.postSessionsStart, { user }),

  load: () => getJson<{ sessions: SessionEntry[] }>(ROUTES.getSessionsLoad),

  imageEvent: (body: {
    image_name: string;
    session_seconds_delta: number;
    annotations_added_delta: number;
    final_annotation_count: number;
    // Where this image's label document lives, so a later read can classify this time as
    // review vs. negative-confirmation vs. new-annotation work.
    dataset_root?: string | null;
    subject?: string | null;
    date?: string | null;
  }) => postJson<unknown>(ROUTES.postSessionsImageEvent, body),
};
