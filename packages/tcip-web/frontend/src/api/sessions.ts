/** Session-tracking API helpers (a workspace project's annotation-stats record). */

import { getJson, postJson } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type {
  ImageEventPayload,
  SessionRef,
  SessionSummary,
  SessionWrite,
} from "@/api/types.generated";

export const sessionsApi = {
  load: () => getJson<{ sessions: SessionSummary[] }>(ROUTES.getSessionsLoad),

  imageEvent: (body: ImageEventPayload) =>
    postJson<SessionWrite>(ROUTES.postSessionsImageEvent, body),

  /** End the session `ref` names as the page may be leaving: a beacon where the browser has one
   *  and accepts it, else a keepalive fetch whose answer nothing reads. */
  endAsPageLeaves: (ref: SessionRef) => {
    const payload = JSON.stringify(ref);
    try {
      if (
        navigator.sendBeacon &&
        navigator.sendBeacon(
          ROUTES.postSessionsEnd,
          new Blob([payload], { type: "application/json" }),
        )
      ) {
        return;
      }
    } catch {
      // A beacon the browser refuses falls through to the keepalive fetch.
    }
    try {
      void fetch(ROUTES.postSessionsEnd, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload,
        keepalive: true,
      }).catch(() => {
        // Nothing reads the answer.
      });
    } catch {
      // Nothing reads the answer.
    }
  },
};
