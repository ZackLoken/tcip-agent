import { Circle } from "react-konva";

import type { PolygonShape } from "@/store/types";

/** A dashed ring over the nearest existing vertex within snap range of the cursor, so a placed
 *  vertex's landing spot is visible before the click that commits it. */
export function SnapIndicator({
  cursor,
  polygons,
  scale,
  radius,
}: {
  cursor: [number, number];
  polygons: PolygonShape[];
  scale: number;
  /** Snap radius in image-space units (screen px / scale); passed in rather than read from a
   *  module constant here, since the caller's SNAP_RADIUS_CANVAS also drives the real snapping
   *  math and must stay the one source of that value. */
  radius: number;
}) {
  let best: [number, number] | null = null;
  let bestD = radius;
  for (const poly of polygons) {
    for (const ring of poly.rings) {
      for (const [x, y] of ring) {
        const d = Math.hypot(x - cursor[0], y - cursor[1]);
        if (d < bestD) {
          bestD = d;
          best = [x, y];
        }
      }
    }
  }
  if (!best) return null;
  const r = 7 / scale; // ring the snap target just outside the vertex handle (~vertex-sized, not 2×)
  return (
    <Circle
      x={best[0]}
      y={best[1]}
      radius={r}
      stroke="#FFE7B1"
      strokeWidth={1.5 / scale}
      dash={[2.5 / scale, 2.5 / scale]}
    />
  );
}
