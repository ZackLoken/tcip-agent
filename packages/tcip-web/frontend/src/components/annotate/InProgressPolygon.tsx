import { Circle, Group, Line } from "react-konva";

import type { DraftStroke } from "@/lib/draftStrokes";
import { dashAt, DRAFT_DASH } from "@/lib/symbology";

/** The strokes of a polygon being drawn or a cut being placed: dashed laid segments with a dot at
 *  each laid vertex, and a thinner dashed tail to the cursor. */
export function InProgressPolygon({
  strokes,
  strokeW,
  vertR,
}: {
  strokes: DraftStroke[];
  strokeW: number;
  vertR: number;
}) {
  return (
    <>
      {strokes.map(({ points, color, vertices }, i) => {
        const width = vertices ? strokeW : strokeW * 0.6;
        return (
          <Group key={i}>
            <Line
              points={points.flat()}
              stroke={color}
              strokeWidth={width}
              dash={dashAt(DRAFT_DASH, width)}
            />
            {vertices &&
              points.map(([x, y], v) => (
                <Circle
                  key={v}
                  x={x}
                  y={y}
                  radius={vertR}
                  fill={color}
                  stroke="#ffffff"
                  strokeWidth={strokeW * 0.5}
                />
              ))}
          </Group>
        );
      })}
    </>
  );
}
