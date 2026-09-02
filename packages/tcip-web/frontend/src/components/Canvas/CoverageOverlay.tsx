/**
 * The coverage lattice drawn on the annotation canvas itself: a Konva layer content in image
 * coordinates, culled to the viewport, listening={false} like every other canvas layer. Cell
 * fills and markers read from records the caller (AnnotateTab) already resolved through
 * useCoverageGrid/useRegionCompleteness; nothing here derives a cell state on its own. State is
 * never conveyed by fill color alone: every marked cell also carries a count, a stroke-width
 * distinction, or a strike-through, so the overlay reads correctly without color.
 */

import { Group, Line, Rect, Text } from "react-konva";

import type { GridCell } from "@/lib/coverage";
import { cellsIntersecting } from "@/lib/coverage";
import type { PixelRect } from "@/lib/viewGeometry";

const LABEL_FONT_PX = 10;

/** Smallest on-screen cell edge, in screen px, at which a cell's name and saved-annotation count
 *  are drawn: four times the overlay's own 10px label font, enough room for a short cell name
 *  and its count without crowding the cell edge. A plain, documented default (the same idiom as
 *  reference_grid.derive_large_raster_grid_tile_size's divisions=16), pending a real annotation
 *  session to check the grain against. */
export const CELL_LABEL_FLOOR_PX = LABEL_FONT_PX * 4;

const SWEPT_RGB = "80, 119, 84";
const ATTEST_RGB = "230, 151, 107";
const BORDER_RGB = "231, 229, 220";

export function CoverageOverlay(props: {
  cells: GridCell[];
  /** The visible image region in image coords; null (not yet measured) draws nothing. */
  viewport: PixelRect | null;
  /** Screen px per image px, so strokes and labels hold a constant screen size. */
  scale: number;
  swept: ReadonlySet<string>;
  activeComplete: ReadonlySet<string>;
  activeStale: ReadonlySet<string>;
  otherComplete: ReadonlySet<string>;
  /** The active subject's saved-annotation count per cell. */
  annotationCounts: Record<string, number>;
}) {
  if (!props.viewport || props.scale <= 0) return null;
  const s = props.scale;
  const strokeW = 1 / s;
  const visible = cellsIntersecting(props.cells, props.viewport);

  return (
    <>
      {visible.map((cell) => {
        const w = cell.x1 - cell.x0;
        const h = cell.y1 - cell.y0;
        const active = props.activeComplete.has(cell.name);
        const stale = props.activeStale.has(cell.name);
        const other = !active && !stale && props.otherComplete.has(cell.name);
        const attested = active || stale || other;
        const count = props.annotationCounts[cell.name] ?? 0;
        const labelFits = w * s >= CELL_LABEL_FLOOR_PX && h * s >= CELL_LABEL_FLOOR_PX;
        return (
          <Group key={cell.name}>
            {props.swept.has(cell.name) && (
              <Rect
                x={cell.x0}
                y={cell.y0}
                width={w}
                height={h}
                fill={`rgba(${SWEPT_RGB}, 0.16)`}
              />
            )}
            <Rect
              x={cell.x0}
              y={cell.y0}
              width={w}
              height={h}
              stroke={`rgba(${BORDER_RGB}, 0.3)`}
              strokeWidth={strokeW}
            />
            {attested && (
              <Rect
                x={cell.x0 + strokeW}
                y={cell.y0 + strokeW}
                width={Math.max(0, w - 2 * strokeW)}
                height={Math.max(0, h - 2 * strokeW)}
                stroke={`rgba(${ATTEST_RGB}, ${active || stale ? 0.9 : 0.35})`}
                strokeWidth={strokeW * (active || stale ? 1.6 : 1)}
              />
            )}
            {stale && (
              <Line
                points={[cell.x0, cell.y0, cell.x1, cell.y1]}
                stroke={`rgba(${ATTEST_RGB}, 0.9)`}
                strokeWidth={strokeW * 1.4}
              />
            )}
            {labelFits && (
              <Text
                x={cell.x0 + 2 / s}
                y={cell.y0 + 2 / s}
                text={count > 0 ? `${cell.name} (${count})` : cell.name}
                fontSize={LABEL_FONT_PX / s}
                fill={`rgba(${BORDER_RGB}, 0.85)`}
              />
            )}
          </Group>
        );
      })}
    </>
  );
}
