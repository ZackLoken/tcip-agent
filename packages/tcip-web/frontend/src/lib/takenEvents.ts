import type { PanelEnvelope } from "@/api/types.generated";

/** The id of every panel event this page has taken, for as long as the page lives. */
const taken = new Set<string>();

/** `handler`, taking each panel event once for the page's life: an envelope carrying an event the
 *  page has taken is dropped before `handler` sees it, and an event is taken once `handler`
 *  returns for it, so one `handler` throws on stays untaken. */
export function takingOnce(
  handler: (envelope: PanelEnvelope) => void,
): (envelope: PanelEnvelope) => void {
  return (envelope) => {
    if (taken.has(envelope.event_id)) return;
    handler(envelope);
    taken.add(envelope.event_id);
  };
}
