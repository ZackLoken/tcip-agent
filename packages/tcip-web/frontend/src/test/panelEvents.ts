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

/** The envelopes `tools/generate_panel_events_fixture.py` received from each panel's socket once
 *  the panel events were posted through the events route: a banner on the Results panel, the
 *  labels of an image written on the Annotate panel, and on the `app` panel an annotate focus on
 *  the open fixture's photographs, one on its capture whose label document will not read, and a
 *  canvas state request, keyed by scene. */
// A JSON import reads each literal as a string, which the literal unions admit only through
// `unknown`.
export const PANEL_EVENTS = produced as unknown as {
  banner: Carrying<Banner>;
  labels_written: Carrying<LabelsWritten>;
  annotate_focus: Carrying<AnnotateFocus>;
  annotate_focus_troubled: Carrying<AnnotateFocus>;
  canvas_state_request: Carrying<CanvasStateRequest>;
};
