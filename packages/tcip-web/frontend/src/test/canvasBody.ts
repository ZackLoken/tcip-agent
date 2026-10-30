import { subjectColor } from "@/api/subjects";
import { buildAnnotateShapes, canvasStateBody, type CanvasStateBody } from "@/lib/canvasSync";
import { useStore } from "@/store";
import { RAW_SELECTION, TEST_PROJECT } from "@/test/store";

/** The push body of the store's canvas over the produced capture's first image of the test
 *  project, built by the canvas's own builder: with its shapes as the mirror draws them when
 *  `full`, else a heartbeat. */
export function canvasBody(full = false): CanvasStateBody {
  const { canvas, gui, user } = useStore.getState();
  const { mode, active_subject: activeSubject } = gui;
  return canvasStateBody({
    projectId: TEST_PROJECT.id,
    image: RAW_SELECTION.images[0],
    canvas,
    viewport: null,
    mode,
    activeSubject,
    cutArmed: false,
    user,
    classes: [],
    shapes: full
      ? buildAnnotateShapes({
          ...canvas,
          mode,
          activeSubject,
          visible: true,
          colorFor: subjectColor,
        })
      : null,
  });
}
