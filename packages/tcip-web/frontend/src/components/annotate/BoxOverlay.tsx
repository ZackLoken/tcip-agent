import { memo } from "react";
import { Rect } from "react-konva";

import { HaloLabel } from "@/components/HaloLabel";
import { dashFor, FOCUS_HALO, type LineStyle } from "@/lib/symbology";
import type { Box } from "@/store/types";

/** Per-shape memo: a drag replaces the whole boxes array on each RAF tick while the unchanged
 *  boxes keep their identity, so a committed box the drag leaves alone skips re-render. A
 *  polygon's derived box is built anew each render and never skips. */
export const BoxOverlay = memo(function BoxOverlay({
  box,
  stroke,
  width,
  style,
  labelSize,
  label,
  showLabel,
  focused,
  handleR,
}: {
  box: Box;
  stroke: string;
  width: number;
  style: LineStyle;
  labelSize: number;
  label: string;
  showLabel?: boolean;
  /** The focused item draws the halo under its own stroke; with `handleR`, its corner handles. */
  focused?: boolean;
  handleR?: number;
}) {
  const corners: [number, number][] = [
    [box.x1, box.y1],
    [box.x2, box.y1],
    [box.x2, box.y2],
    [box.x1, box.y2],
  ];
  const geometry = {
    x: box.x1,
    y: box.y1,
    width: box.x2 - box.x1,
    height: box.y2 - box.y1,
  };
  return (
    <>
      {focused && (
        <Rect
          {...geometry}
          stroke={FOCUS_HALO.color}
          strokeWidth={width * FOCUS_HALO.widthFactor}
          opacity={FOCUS_HALO.opacity}
        />
      )}
      <Rect {...geometry} stroke={stroke} strokeWidth={width} dash={dashFor(style, width)} />
      {focused &&
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
