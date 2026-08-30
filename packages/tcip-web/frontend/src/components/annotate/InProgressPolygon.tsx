import { Circle, Line } from "react-konva";

/** The polygon being drawn: its committed vertices plus a dashed rubber-band to the cursor. */
export function InProgressPolygon({
  points,
  cursor,
  stroke,
  strokeW,
  vertR,
}: {
  points: [number, number][];
  cursor: [number, number] | null;
  stroke: string;
  strokeW: number;
  vertR: number;
}) {
  const dash = [strokeW * 4, strokeW * 4];
  return (
    <>
      <Line points={points.flat()} stroke={stroke} strokeWidth={strokeW} dash={dash} />
      {cursor && (
        <Line
          points={[...points[points.length - 1], ...cursor]}
          stroke={stroke}
          strokeWidth={strokeW * 0.6}
          dash={dash}
        />
      )}
      {points.map(([x, y], i) => (
        <Circle
          key={`cp-${i}`}
          x={x}
          y={y}
          radius={vertR}
          fill={stroke}
          stroke="#ffffff"
          strokeWidth={strokeW * 0.5}
        />
      ))}
    </>
  );
}
