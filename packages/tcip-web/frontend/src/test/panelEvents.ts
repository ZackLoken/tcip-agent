import type {
  AnnotateFocus,
  AuditGapRefusal,
  Banner,
  CanvasStateRequest,
  LabelsWritten,
  PanelDismissed,
  PanelEnvelope,
  ProjectNotOpenRefusal,
} from "@/api/types.generated";
import produced from "@/test/panelEvents.json";
import type { Refusal } from "@/test/refusals";

/** A broadcast envelope carrying the event `E`. */
type Carrying<E> = PanelEnvelope & { event: E };

/** One delivery of each scene's envelope, keyed by scene. */
interface Deliveries {
  banner: Carrying<Banner>;
  banner_second: Carrying<Banner>;
  banner_blank: Carrying<Banner>;
  labels_written: Carrying<LabelsWritten>;
  labels_written_second: Carrying<LabelsWritten>;
  app_banner: Carrying<Banner>;
  annotate_focus: Carrying<AnnotateFocus>;
  annotate_focus_troubled: Carrying<AnnotateFocus>;
  canvas_state_request: Carrying<CanvasStateRequest>;
  labels_written_declared: Carrying<LabelsWritten>;
}

/** The envelopes `tools/generate_panel_events_fixture.py` received from each panel's socket once
 *  the panel events were posted through the events route: two banners on the Results panel and
 *  one whose text is blank, the labels of two images written on the Annotate panel, on the `app`
 *  panel a banner, an annotate focus on the open fixture's photographs, one on its capture whose
 *  label document will not read, and a canvas state request, and last the first image's labels
 *  written on the Annotate panel again by a sender declaring an agent client, keyed by scene.
 *  `live` is what a socket open at the posts received, `at_connect` what a socket connecting
 *  after them received, and `after_dismissal` what a socket connecting once the first Results
 *  banner was dismissed received, one event carrying one id in each. `dismissals` holds the
 *  dismissal route's answers: the refusal of a dismissal of the first Results banner naming a
 *  project other than the open one (`another_project`) and of one naming no person (`no_one`),
 *  that banner's dismissal by the person who opened the project (`dismissed`), and the refusal of
 *  the second Results banner's dismissal whose audit line was not written (`audit_gap`). */
// A JSON import reads each literal as a string, which the literal unions admit only through
// `unknown`.
const DELIVERIES = produced as unknown as {
  live: Deliveries;
  at_connect: Deliveries;
  after_dismissal: Omit<Deliveries, "banner">;
  dismissals: {
    another_project: Refusal<ProjectNotOpenRefusal>;
    no_one: Refusal<string>;
    dismissed: PanelDismissed;
    audit_gap: Refusal<AuditGapRefusal & { committed: true; what: PanelDismissed }>;
  };
};

/** Each scene's envelope as a socket open at the post received it. */
export const PANEL_EVENTS = DELIVERIES.live;

/** Each scene's envelope as a socket connecting after the posts received it. */
export const PANEL_EVENTS_AT_CONNECT = DELIVERIES.at_connect;

/** Each scene's envelope as a socket connecting after the first Results banner's dismissal
 *  received it; that banner is not among them. */
export const PANEL_EVENTS_AFTER_DISMISSAL = DELIVERIES.after_dismissal;

/** The dismissal route's answers, keyed as `DELIVERIES` states. */
export const DISMISSALS = DELIVERIES.dismissals;
