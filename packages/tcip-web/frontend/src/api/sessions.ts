/** Session-tracking API helpers (annotation_stats.json on disk). */

import { getJson, postJson } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type { ImageEventPayload, SessionSummary } from "@/api/types.generated";

/** Every call records against the project the backend has open. */
export const sessionsApi = {
  load: () => getJson<{ sessions: SessionSummary[] }>(ROUTES.getSessionsLoad),

  imageEvent: (body: ImageEventPayload) => postJson<unknown>(ROUTES.postSessionsImageEvent, body),
};
