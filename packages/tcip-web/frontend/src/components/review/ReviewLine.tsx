import { Line } from "react-konva";

export function ReviewLine({
  points,
  stroke,
  lw,
  weight,
  dashed,
  fill,
}: {
  points: [number, number][];
  stroke: string;
  lw: number;
  weight: number;
  dashed?: boolean;
  fill?: string;
}) {
  if (points.length < 2) return null;
  return (
    <Line
      points={points.flat()}
      closed
      stroke={stroke}
      strokeWidth={weight * lw}
      dash={dashed ? [8 * lw, 4 * lw] : undefined}
      fill={fill}
    />
  );
}
