import type {
  AnnotateFocus,
  Banner,
  CanvasStateRequest,
  LabelsWritten,
  PanelEnvelope,
} from "@/api/types.generated";
import produced from "@/test/panelEvents.json";

/** A broadcast envelope carrying the event `E`. */
type Carrying<E> = PanelEnvelope & { event: E };

/** One delivery of each scene's envelope, keyed by scene. */
interface Deliveries {
  banner: Carrying<Banner>;
  banner_second: Carrying<Banner>;
  labels_written: Carrying<LabelsWritten>;
  labels_written_second: Carrying<LabelsWritten>;
  app_banner: Carrying<Banner>;
  annotate_focus: Carrying<AnnotateFocus>;
  annotate_focus_troubled: Carrying<AnnotateFocus>;
  canvas_state_request: Carrying<CanvasStateRequest>;
}

/** The envelopes `tools/generate_panel_events_fixture.py` received from each panel's socket once
 *  the panel events were posted through the events route: two banners on the Results panel, the
 *  labels of two images written on the Annotate panel, and on the `app` panel a banner, an
 *  annotate focus on the open fixture's photographs, one on its capture whose label document will
 *  not read, and a canvas state request, keyed by scene. `live` is what a socket open at the posts
 *  received and `at_connect` what a socket connecting after them received, one event carrying one
 *  id in both. */
// A JSON import reads each literal as a string, which the literal unions admit only through
// `unknown`.
const DELIVERIES = produced as unknown as {
  live: Deliveries;
  at_connect: Deliveries;
};

/** Each scene's envelope as a socket open at the post received it. */
export const PANEL_EVENTS = DELIVERIES.live;

/** Each scene's envelope as a socket connecting after the posts received it. */
export const PANEL_EVENTS_AT_CONNECT = DELIVERIES.at_connect;
