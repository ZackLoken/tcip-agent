import { Circle, Line } from "react-konva";

/** A point annotation under review: the Annotate canvas' reticle in the detection's outcome colour.
 *  Same mark in both tabs, so a location a reviewer accepts is drawn the way it was placed, and no
 *  box is drawn around it, which would show the reviewer an extent the annotation does not claim. */
export function ReviewPoint({
  point,
  stroke,
  lw,
  weight,
}: {
  point: [number, number];
  stroke: string;
  lw: number;
  weight: number;
}) {
  const [x, y] = point;
  const inner = 6.5 * lw;
  const outer = 11 * lw;
  const ticks: [number, number, number, number][] = [
    [x, y - inner, x, y - outer],
    [x, y + inner, x, y + outer],
    [x - inner, y, x - outer, y],
    [x + inner, y, x + outer, y],
  ];
  return (
    <>
      {ticks.map(([x1, y1, x2, y2], i) => (
        <Line key={i} points={[x1, y1, x2, y2]} stroke={stroke} strokeWidth={weight * lw} />
      ))}
      <Circle
        x={x}
        y={y}
        radius={4 * lw}
        fill={stroke}
        stroke="#FFFFFF"
        strokeWidth={weight * 0.5 * lw}
      />
    </>
  );
}
