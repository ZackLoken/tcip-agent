/** Cycling the Annotate toolbar's drawing tool (its `m` shortcut and any other stepper). */

import type { Mode } from "@/store/types";

/** The mode `m` advances to: Point -> Box -> Polygon -> Point, the toolbar's left-to-right order
 *  (the array below is a rotation of the same cycle, so the transitions are unchanged). */
export function nextMode(mode: Mode): Mode {
  const order: Mode[] = ["box", "polygon", "point"];
  return order[(order.indexOf(mode) + 1) % order.length];
}
