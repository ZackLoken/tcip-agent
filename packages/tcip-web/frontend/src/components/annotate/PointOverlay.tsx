import { memo } from "react";
import { Circle, Line } from "react-konva";

import { HaloLabel } from "@/components/HaloLabel";
import { dashFor, FOCUS_HALO, type LineStyle } from "@/lib/symbology";
import type { PointShape } from "@/store/types";

/**
 * A placed point: four short ticks converging on the coordinate, plus a filled core in the
 * item's color with a white keyline. The ticks are the point of the mark: they say "this exact
 * location" the way an instrument's reticle does, and they are what separates a point from the two
 * things it could otherwise be mistaken for on this canvas: a very small box or a collapsed polygon
 * (both hollow outlines) and a polygon vertex handle (a bare filled dot).
 */
export const PointOverlay = memo(function PointOverlay({
  point,
  stroke,
  style,
  coreR,
  tickInner,
  tickOuter,
  lineW,
  labelSize,
  label,
  showLabel,
  focused,
}: {
  point: PointShape;
  stroke: string;
  style: LineStyle;
  coreR: number;
  tickInner: number;
  tickOuter: number;
  lineW: number;
  labelSize: number;
  label: string;
  showLabel?: boolean;
  /** The focused item draws the halo as a ring under its core. */
  focused?: boolean;
}) {
  const { x, y } = point;
  const ticks: [number, number, number, number][] = [
    [x, y - tickInner, x, y - tickOuter],
    [x, y + tickInner, x, y + tickOuter],
    [x - tickInner, y, x - tickOuter, y],
    [x + tickInner, y, x + tickOuter, y],
  ];
  const dash = dashFor(style, lineW);
  return (
    <>
      {focused && (
        <Circle
          x={x}
          y={y}
          radius={tickOuter}
          stroke={FOCUS_HALO.color}
          strokeWidth={lineW * FOCUS_HALO.widthFactor}
          opacity={FOCUS_HALO.opacity}
        />
      )}
      {ticks.map(([x1, y1, x2, y2], i) => (
        <Line
          key={`t-${i}`}
          points={[x1, y1, x2, y2]}
          stroke={stroke}
          strokeWidth={lineW}
          dash={dash}
        />
      ))}
      <Circle x={x} y={y} radius={coreR} fill={stroke} stroke="#ffffff" strokeWidth={lineW * 0.6} />
      {showLabel && (
        <HaloLabel x={x + tickOuter} y={y} text={label} fill={stroke} size={labelSize} />
      )}
    </>
  );
});
