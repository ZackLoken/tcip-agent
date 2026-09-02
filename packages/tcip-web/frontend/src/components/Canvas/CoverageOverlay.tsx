/**
 * The coverage lattice drawn on the annotation canvas itself: a Konva layer content in image
 * coordinates, culled to the viewport, listening={false} like every other canvas layer. Cell
 * fills and markers read from records the caller (AnnotateTab) already resolved through
 * useCoverageGrid/useRegionCompleteness; nothing here derives a cell state on its own. State is
 * never conveyed by fill color alone: swept also carries a dashed border, attested a solid one
 * (brighter and wider for the active subject than another's), stale a strike-through, and a
 * saved count is a label rather than a third fill. The overlay itself names none of these; the
 * canvas chrome carries the key (see CoverageChrome).
 */

import { Group, Line, Rect, Text } from "react-konva";

import type { GridCell } from "@/lib/coverage";
import { cellsIntersecting } from "@/lib/coverage";
import type { PixelRect } from "@/lib/viewGeometry";

const LABEL_FONT_PX = 10;

/** Arial's own per-glyph advance widths (a fraction of font size), the font Konva's Text
 *  defaults to with no fontFamily set: the metric the label floor below measures against. */
const GLYPH_ADVANCE_EM: Record<"digit" | "upper" | "space" | "paren", number> = {
  digit: 0.556,
  upper: 0.667,
  space: 0.278,
  paren: 0.333,
};

function glyphAdvanceEm(ch: string): number {
  if (/[0-9]/.test(ch)) return GLYPH_ADVANCE_EM.digit;
  if (/[A-Z]/.test(ch)) return GLYPH_ADVANCE_EM.upper;
  if (ch === " ") return GLYPH_ADVANCE_EM.space;
  if (ch === "(" || ch === ")") return GLYPH_ADVANCE_EM.paren;
  return GLYPH_ADVANCE_EM.upper;
}

/** A label's rendered width at `fontPx`, from `GLYPH_ADVANCE_EM` rather than a live canvas
 *  measurement: jsdom carries no 2D canvas context, so a measurement taken here would read zero
 *  under test and the floor it feeds would silently vanish. */
export function measureLabelWidth(text: string, fontPx: number): number {
  let em = 0;
  for (const ch of text) em += glyphAdvanceEm(ch);
  return em * fontPx;
}

/** The widest label this platform's own lattice can produce: `derive_large_raster_grid_tile_size`
 *  (reference_grid.py) caps a large raster's lattice at 16 divisions of the long edge, so the
 *  widest cell name is a single-letter column plus a two-digit row ("P16"); a three-digit
 *  saved-annotation count is the assumed practical ceiling for the parenthesised suffix. */
const WIDEST_LABEL = "P16 (137)";

const LABEL_INSET_PX = 2;

/** Smallest on-screen cell edge, in screen px, at which a cell's name and saved-annotation count
 *  are drawn: the widest label this lattice can produce, measured at the label font, plus the
 *  inset the label is drawn at on each side. Below this a cell's own edge cannot hold its own
 *  worst-case label without overrunning into the next cell. */
export const CELL_LABEL_FLOOR_PX =
  Math.ceil(measureLabelWidth(WIDEST_LABEL, LABEL_FONT_PX)) + 2 * LABEL_INSET_PX;

/** #507754, tcip-accent (tailwind.config.ts): the recorded sweep fill. */
const SWEPT_RGB = "80, 119, 84";
/** #E6976B, tcip-warn: the attested border and the stale strike. */
const ATTEST_RGB = "230, 151, 107";
/** #E7E5DC, tcip-fg: the per-cell border and label. */
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
              <>
                <Rect
                  x={cell.x0}
                  y={cell.y0}
                  width={w}
                  height={h}
                  fill={`rgba(${SWEPT_RGB}, 0.16)`}
                />
                <Rect
                  x={cell.x0 + strokeW}
                  y={cell.y0 + strokeW}
                  width={Math.max(0, w - 2 * strokeW)}
                  height={Math.max(0, h - 2 * strokeW)}
                  stroke={`rgba(${SWEPT_RGB}, 0.7)`}
                  strokeWidth={strokeW}
                  dash={[4 * strokeW, 3 * strokeW]}
                />
              </>
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
