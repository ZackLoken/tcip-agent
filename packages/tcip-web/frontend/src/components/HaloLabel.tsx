import { Text } from "react-konva";

/**
 * A label with a dark halo behind it (a blurred black copy, then the real fill on top) so an
 * annotation or detection name stays legible over any part of the underlying image. Shared by the
 * Annotate and Review canvases so the same shape's name reads the same on both.
 */
export function HaloLabel({
  x,
  y,
  text,
  fill,
  size,
}: {
  x: number;
  y: number;
  text: string;
  fill: string;
  size: number;
}) {
  return (
    <>
      <Text
        x={x + 2}
        y={y - size - 2}
        text={text}
        fill="#000000"
        fontSize={size}
        fontStyle="bold"
        shadowColor="#000000"
        shadowBlur={size * 0.2}
        shadowOffset={{ x: 0, y: 0 }}
        shadowOpacity={0.9}
      />
      <Text x={x + 2} y={y - size - 2} text={text} fill={fill} fontSize={size} fontStyle="bold" />
    </>
  );
}
