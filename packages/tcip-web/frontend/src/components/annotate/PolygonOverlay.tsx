import { memo } from "react";
import { Circle, Line } from "react-konva";

import { HaloLabel } from "@/components/HaloLabel";
import { dashFor, FOCUS_HALO, type LineStyle } from "@/lib/symbology";
import type { PolygonShape } from "@/store/types";

export const PolygonOverlay = memo(function PolygonOverlay({
  polygon,
  stroke,
  width,
  style,
  vertexRadius,
  showVertices,
  labelSize,
  label,
  showLabel,
  focused,
}: {
  polygon: PolygonShape;
  stroke: string;
  width: number;
  style: LineStyle;
  vertexRadius: number;
  showVertices: boolean;
  labelSize: number;
  label: string;
  showLabel?: boolean;
  /** The focused item draws the halo under every ring's own stroke. */
  focused?: boolean;
}) {
  /** Every ring of the annotation draws, in the instance's own stroke: the shape a reviewer
   *  confirms is all of it, not the first contour. Selection/hover styling is shared, so touching
   *  any part lights up all of them: that shared highlight is what reads as "these are one
   *  object". */
  const rings = polygon.rings.filter((ring) => ring.length >= 2);
  if (!rings.length) return null;
  const [x0, y0] = rings[0][0];
  const dash = dashFor(style, width);
  return (
    <>
      {focused &&
        rings.map((ring, ri) => (
          <Line
            key={`halo-${ri}`}
            points={ring.flat()}
            closed
            stroke={FOCUS_HALO.color}
            strokeWidth={width * FOCUS_HALO.widthFactor}
            opacity={FOCUS_HALO.opacity}
          />
        ))}
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
