/** Cycling the Annotate toolbar's drawing tool (its `m` shortcut and any other stepper). */

import type { Mode } from "@/store/types";

/** The drawing tools, in the order `m` cycles them. */
export const MODES: Mode[] = ["box", "polygon", "point"];

/** The mode `m` advances to: Point -> Box -> Polygon -> Point, the toolbar's left-to-right order
 *  (the array above is a rotation of the same cycle, so the transitions are unchanged). */
export function nextMode(mode: Mode): Mode {
  return MODES[(MODES.indexOf(mode) + 1) % MODES.length];
}
