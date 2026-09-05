import { memo } from "react";
import { Circle, Line } from "react-konva";

import { HaloLabel } from "@/components/HaloLabel";
import { dashPattern } from "@/lib/authorshipSymbology";
import type { PolygonShape } from "@/store/types";

export const PolygonOverlay = memo(function PolygonOverlay({
  polygon,
  stroke,
  width,
  vertexRadius,
  showVertices,
  labelSize,
  label,
  showLabel,
  dashed,
}: {
  polygon: PolygonShape;
  stroke: string;
  width: number;
  vertexRadius: number;
  showVertices: boolean;
  labelSize: number;
  label: string;
  showLabel?: boolean;
  /** A tool's own polygon that no person has accepted draws dotted; every other polygon is solid. */
  dashed?: "tool";
}) {
  /** Every ring of the annotation draws, in the instance's own stroke: the shape a reviewer
   *  confirms is all of it, not the first contour. Selection/hover styling is shared, so touching
   *  any part lights up all of them: that shared highlight is what reads as "these are one
   *  object". */
  const rings = polygon.rings.filter((ring) => ring.length >= 2);
  if (!rings.length) return null;
  const [x0, y0] = rings[0][0];
  const dash = dashed ? dashPattern(dashed, width) : undefined;
  return (
    <>
      {rings.map((ring, ri) => (
        <Line
          key={`r-${ri}`}
          points={ring.flat()}
          closed
          stroke={stroke}
          strokeWidth={width}
          dash={dash}
        />
      ))}
      {showVertices &&
        rings.map((ring, ri) =>
          ring.map(([x, y], i) => (
            <Circle
              key={`v-${ri}-${i}`}
              x={x}
              y={y}
              radius={vertexRadius}
              fill={stroke}
              stroke="#ffffff"
              strokeWidth={width * 0.5}
            />
          )),
        )}
      {/* One label per annotation, not per ring: a two-part shape is one annotation. */}
      {showLabel && <HaloLabel x={x0} y={y0} text={label} fill={stroke} size={labelSize} />}
    </>
  );
});
