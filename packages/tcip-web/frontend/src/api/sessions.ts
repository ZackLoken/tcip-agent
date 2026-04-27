/** Session-tracking API helpers (annotation_stats.json on disk). */

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
    fetch("/api/sessions/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_root, user }),
    }).then((r) => r.json()),

  end: (project_root: string) =>
    fetch("/api/sessions/end", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_root }),
    }).then((r) => r.json()),

  load: (project_root: string) =>
    fetch(`/api/sessions/load?project_root=${encodeURIComponent(project_root)}`).then(
      (r) => r.json() as Promise<{ sessions: SessionEntry[]; image_status: Record<string, string> }>,
    ),

  imageEvent: (body: {
    project_root: string;
    image_name: string;
    session_seconds_delta: number;
    annotations_added_delta: number;
    final_annotation_count: number;
    loaded_annotation_count?: number | null;
  }) =>
    fetch("/api/sessions/image_event", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json()),
};
