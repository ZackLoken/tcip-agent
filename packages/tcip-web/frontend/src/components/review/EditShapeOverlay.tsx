import { Circle, Line, Rect } from "react-konva";

import { useStore } from "@/store";
import type { EditShape } from "@/lib/editGeometry";

export function EditShapeOverlay({ edit, color }: { edit: EditShape; color: string }) {
  const scale = useStore((s) => s.gui.view.scale);
  const lw = 1 / (scale || 1);
  const hs = 5 * lw; // handle half-size
  if (edit.kind === "box") {
    const [x1, y1, x2, y2] = edit.box;
    const corners: [number, number][] = [
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2],
    ];
    return (
      <>
        <Rect
          x={x1}
          y={y1}
          width={x2 - x1}
          height={y2 - y1}
          stroke={color}
          strokeWidth={2.5 * lw}
          fill={`${color}14`}
        />
        {corners.map(([cx, cy], i) => (
          <Rect
            key={i}
            x={cx - hs}
            y={cy - hs}
            width={hs * 2}
            height={hs * 2}
            fill="#FFFFFF"
            stroke={color}
            strokeWidth={1.5 * lw}
          />
        ))}
      </>
    );
  }
  if (edit.points.length < 2) return null;
  return (
    <>
      <Line
        points={edit.points.flat()}
        closed
        stroke={color}
        strokeWidth={2.5 * lw}
        fill={`${color}14`}
      />
      {edit.points.map(([px, py], i) => (
        <Circle
          key={i}
          x={px}
          y={py}
          radius={4.5 * lw}
          fill="#FFFFFF"
          stroke={color}
          strokeWidth={1.5 * lw}
        />
      ))}
    </>
  );
}
