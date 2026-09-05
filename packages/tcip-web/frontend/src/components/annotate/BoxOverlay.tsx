import { memo } from "react";
import { Rect } from "react-konva";

import { HaloLabel } from "@/components/HaloLabel";
import { dashPattern, type DashKind } from "@/lib/authorshipSymbology";
import type { Box } from "@/store/types";

/** Per-shape memo: dragVertex/dragBox replace the whole polygons/boxes array on each RAF tick
 *  (slice() keeps the unchanged elements' identity), so an unrelated shape's props are
 *  referentially equal and it skips re-render, containing a one-shape drag to that shape. */
export const BoxOverlay = memo(function BoxOverlay({
  box,
  stroke,
  width,
  labelSize,
  label,
  showLabel,
  selected,
  handleR,
  dashed,
}: {
  box: Box;
  stroke: string;
  width: number;
  labelSize: number;
  label: string;
  /** Labels are hover/selection-only; the legend is the standing symbology reference. */
  showLabel?: boolean;
  selected?: boolean;
  handleR?: number;
  dashed?: DashKind;
}) {
  const corners: [number, number][] = [
    [box.x1, box.y1],
    [box.x2, box.y1],
    [box.x2, box.y2],
    [box.x1, box.y2],
  ];
  return (
    <>
      <Rect
        x={box.x1}
        y={box.y1}
        width={box.x2 - box.x1}
        height={box.y2 - box.y1}
        stroke={stroke}
        strokeWidth={width}
        dash={dashed ? dashPattern(dashed, width) : undefined}
      />
      {selected &&
        handleR &&
        corners.map(([cx, cy], i) => (
          <Rect
            key={`h-${i}`}
            x={cx - handleR}
            y={cy - handleR}
            width={handleR * 2}
            height={handleR * 2}
            fill="#ffffff"
            stroke={stroke}
            strokeWidth={width * 0.6}
          />
        ))}
      {showLabel && <HaloLabel x={box.x1} y={box.y1} text={label} fill={stroke} size={labelSize} />}
    </>
  );
});
