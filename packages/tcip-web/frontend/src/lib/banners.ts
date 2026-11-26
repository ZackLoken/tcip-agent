/**
 * The agent's banners as entries of the notifications area: each banner a panel delivers is one
 * notice until a person dismisses it there, and that dismissal is sent to the backend, which then
 * sends the banner to no page again.
 */

import { api } from "@/api/client";
import { committedOf, messageOf, refusalOf } from "@/api/http";
import {
  PANEL_EVENT_BANNER,
  PROJECT_NOT_OPEN,
  type Panel,
  type PanelEnvelope,
  type ProjectNotOpenRefusal,
} from "@/api/types.generated";
import { TAB_LABELS } from "@/lib/tabLabels";
import { useStore } from "@/store";
import type { EntryContext } from "@/store/slices/notifications";

/** One banner as its entry holds it: the panel it was pushed to, its event's id, its text, and
 *  its entry's context, the project it was posted for and where it was pushed. */
interface Note {
  panel: Panel;
  eventId: string;
  text: string;
  context: { project: string } & EntryContext;
}

/** Where a banner was pushed, as the person reads it: the agent's note for a tab, or for the
 *  app as a whole. */
function noteTarget(panel: Panel): string {
  return `Agent's note for ${panel === "app" ? "the app" : TAB_LABELS[panel]}`;
}

/** The key the entry of the banner `eventId` names stands on. */
const noteKey = (eventId: string): string => `agent-note:${eventId}`;

/** The key a failed dismissal of the banner `eventId` names stands on. */
const dismissalKey = (eventId: string): string => `agent-note-dismissal:${eventId}`;

/** Send the dismissal of `note`, as the committed person, while its project is the open one;
 *  `retry` sends it again. A dismissal not sent, answered, refused because another project is
 *  open, or committed without its line ends the failure standing on its key, the last raising the
 *  refusal's sentence as a warning; any other refusal or no answer raises a failure on that key
 *  offering `retry`. */
async function sendDismissal(note: Note, retry: () => void): Promise<void> {
  const { panel, eventId, text, context } = note;
  const { openProject, user, releaseNotification } = useStore.getState();
  const key = dismissalKey(eventId);
  if (openProject?.id !== context.project) {
    releaseNotification(key);
    return;
  }
  try {
    await api.events.dismiss(panel, { user, project_id: context.project, event_id: eventId });
  } catch (e) {
    const { raiseNotification } = useStore.getState();
    if (committedOf(e) !== null) {
      releaseNotification(key);
      raiseNotification({
        kind: "notice",
        level: "warning",
        text: { sentence: messageOf(e) },
        context,
      });
    } else if (refusalOf<ProjectNotOpenRefusal>(e, PROJECT_NOT_OPEN)) {
      releaseNotification(key);
    } else {
      raiseNotification({
        kind: "failure",
        level: "error",
        text: {
          sentence: `Dismissing the agent's note "${text}" failed (${messageOf(e)}), so it may come back after a reload. Retry to dismiss it.`,
        },
        context,
        key,
        action: { label: "Retry", run: retry },
      });
    }
    return;
  }
  releaseNotification(key);
}

/** List the banner `envelope` carries as a notice of the project it was posted for, naming where
 *  it was pushed, standing on its event until a person dismisses it, which sends its dismissal;
 *  a banner whose text is blank lists nothing. False for any other event. */
export function takeBanner({ panel, event, event_id, project_id }: PanelEnvelope): boolean {
  if (event.event_type !== PANEL_EVENT_BANNER) return false;
  if (!event.text.trim()) return true;
  const note: Note = {
    panel,
    eventId: event_id,
    text: event.text,
    context: { project: project_id, target: noteTarget(panel) },
  };
  const dismiss = (): void => void sendDismissal(note, dismiss);
  useStore.getState().raiseNotification({
    kind: "notice",
    level: "info",
    text: { sentence: note.text },
    context: note.context,
    key: noteKey(event_id),
    onDismiss: dismiss,
  });
  return true;
}
