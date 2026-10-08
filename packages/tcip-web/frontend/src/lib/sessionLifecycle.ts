/**
 * The page's one ordered line to the backend's session record. Each held contribution is posted
 * after the one queued before it has answered, and an end of the recorded session runs after
 * every contribution queued before it, so the answer that names a session is adopted before
 * anything acts on it and no answer arrives after the end that consumed it.
 */

import { committedOf, isAuditEntryNotWritten, StructuredRefusalError } from "@/api/http";
import { sessionsApi } from "@/api/sessions";
import type { ImageEventPayload, SessionWrite } from "@/api/types.generated";
import { useStore } from "@/store";
import { holds, noticeUnconfirmed } from "@/store/slices/registryStatus";

let line: Promise<void> = Promise.resolve();
const queued = new Set<string>();
const REFUSED_CHANNEL = "refused-visits";

function after(step: () => Promise<void> | void): void {
  line = line.then(step, step);
}

/** Forget the recorded session when it is the session `started` stamps; a different recorded
 *  session is kept. */
function forgetSession(started: string): void {
  const { recordedSession, setRecordedSession } = useStore.getState();
  if (recordedSession?.started === started) setRecordedSession(null);
}

/** Retire `contribution` and act on the session its answer says it is recorded in: one still
 *  unended becomes the recorded session while the contribution's project is the open one, one
 *  that has ended is forgotten (`forgetSession`), and an answer naming none, for a contribution
 *  with nothing recorded, changes nothing. */
function adopt(contribution: ImageEventPayload, written: SessionWrite): void {
  const { retireContribution, setRecordedSession, openProject } = useStore.getState();
  retireContribution(contribution);
  const place = written.session;
  if (place === null) return;
  if (place.ended) {
    forgetSession(place.started);
  } else if (contribution.project_id === openProject?.id) {
    setRecordedSession({
      project_id: contribution.project_id,
      started: place.started,
      user: contribution.user,
    });
  }
}

/** Dispose of one posted contribution by its answer while it is still held; the answer to one
 *  retired meanwhile updates nothing. One the backend accepted, or committed and could not
 *  record the line of (the gap toasted), is adopted. One it refused by name (a 4xx answer other
 *  than that gap) is retired, the session its request named forgotten (`forgetSession`), since
 *  that named session is what a refusal answers about, so the next visit opens a fresh one, and
 *  its refusal added to the one notice of refused visits (`pushRefusal`). One with
 *  no answer or an indeterminate one stays held to be sent again under the same identity, and
 *  the notice of unconfirmed visits names every image now unconfirmed. */
function dispose(contribution: ImageEventPayload, posted: Promise<SessionWrite>): Promise<void> {
  const held = () => holds(useStore.getState().heldContributions, contribution);
  return posted.then(
    (written) => {
      if (held()) adopt(contribution, written);
    },
    (e: unknown) => {
      if (!held()) return;
      const s = useStore.getState();
      const committed = committedOf<SessionWrite>(e);
      if (committed !== null) {
        adopt(contribution, committed);
        s.pushToast(e instanceof Error ? e.message : String(e));
      } else if (
        !isAuditEntryNotWritten(e) &&
        e instanceof StructuredRefusalError &&
        e.status < 500
      ) {
        s.retireContribution(contribution);
        if (contribution.started !== null) forgetSession(contribution.started);
        s.pushRefusal(REFUSED_CHANNEL, contribution.image_name, e.message);
      } else {
        noticeUnconfirmed(s, true);
      }
    },
  );
}

/** Send `contribution` when it is still held, marking it unconfirmed until an answer settles
 *  whether it was recorded. */
function send(contribution: ImageEventPayload): Promise<void> | void {
  const { heldContributions, unconfirmedContributions } = useStore.getState();
  if (!holds(heldContributions, contribution)) return;
  if (!holds(unconfirmedContributions, contribution)) {
    useStore.setState({ unconfirmedContributions: [...unconfirmedContributions, contribution] });
  }
  return dispose(contribution, sessionsApi.imageEvent(contribution));
}

/** Queue every held visit not already queued, in the order they were held, whichever project
 *  each is of: the backend admits one naming a session by that session, and one naming none by
 *  the open project. */
export function postHeldContributions(): void {
  for (const contribution of useStore.getState().heldContributions) {
    const id = contribution.contribution_id;
    if (queued.has(id)) continue;
    queued.add(id);
    after(() => Promise.resolve(send(contribution)).finally(() => queued.delete(id)));
  }
}

/** End the recorded session once every contribution queued before this call has answered, and
 *  forget it, so a second call sends nothing; nothing is sent when no session is recorded. */
export function endRecordedSession(): void {
  after(() => {
    const { recordedSession, setRecordedSession } = useStore.getState();
    if (!recordedSession) return;
    setRecordedSession(null);
    sessionsApi.endAsPageLeaves({
      project_id: recordedSession.project_id,
      started: recordedSession.started,
    });
  });
}

/** Hold the open visit's contribution so far, its image staying open with its clock stopped
 *  until the switch departs it or resumes it, and post every held contribution, resolving once
 *  none is held: each landed, was refused by name, or was retired by a departure that arrived
 *  meanwhile (one that also supersedes the switch). A switch this page starts so finishes its
 *  visits before it asks the backend to open the next project; rejects, naming how many are
 *  still held, when any had no answer or an indeterminate one. */
export async function settleVisits(): Promise<void> {
  useStore.getState().pauseSessionInterval();
  postHeldContributions();
  await line;
  const held = useStore.getState().heldContributions.length;
  if (held) {
    throw new Error(
      `${held} image visit(s) could not be sent to the backend; nothing was opened, try again`,
    );
  }
}
