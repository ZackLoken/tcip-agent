import { Rect } from "react-konva";

export function ReviewRect({
  box,
  stroke,
  lw,
  weight,
  dashed,
  fill,
}: {
  box: [number, number, number, number];
  stroke: string;
  lw: number;
  weight: number;
  dashed?: boolean;
  fill?: string;
}) {
  const [x1, y1, x2, y2] = box;
  return (
    <Rect
      x={x1}
      y={y1}
      width={x2 - x1}
      height={y2 - y1}
      stroke={stroke}
      strokeWidth={weight * lw}
      dash={dashed ? [8 * lw, 4 * lw] : undefined}
      fill={fill}
    />
  );
}
