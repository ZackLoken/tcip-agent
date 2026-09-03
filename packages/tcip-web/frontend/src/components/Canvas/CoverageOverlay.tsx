/**
 * The coverage lattice drawn on the annotation canvas itself: a Konva layer content in image
 * coordinates, culled to the viewport, listening={false} like every other canvas layer. Cell
 * fills and markers read from records the caller (AnnotateTab) already resolved through
 * useCoverageGrid/useRegionCompleteness; nothing here derives a cell state on its own. State is
 * never conveyed by fill color alone: a recorded swept cell carries a fill plus a dashed border,
 * a pending one (seen locally, not yet acknowledged by the server, see coverageTracker.ts's
 * `pending`) carries a short-dash border and no fill, attested carries a solid border (the
 * active subject's a dotted stroke never marks the other), stale a strike-through, and a saved
 * count is a label rather than a third fill. Every border, dashed stroke and label draws
 * with a two-tone halo (a dark line under the light one, the HaloLabel pattern already used for
 * shape names elsewhere on this canvas) so a mark reads on ground bright or dark, orchard mosaics
 * included. The overlay itself names none of these; the canvas chrome carries the key (see
 * CoverageChrome).
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

const LABEL_INSET_PX = 2;

/** A three-digit saved-annotation count is the assumed practical ceiling for a label's
 *  parenthesised suffix ("(137)"). */
const WIDEST_COUNT_SUFFIX = " (137)";

/** Smallest on-screen cell edge, in screen px, at which a cell's name and saved-annotation count
 *  are drawn: the widest name actually present in `cells` (a set-zoom lattice's own row/column
 *  extent varies with the zoom and the raster, so no fixed name length bounds every lattice this
 *  platform can derive), plus the assumed count suffix, measured at the label font, plus the
 *  inset the label is drawn at on each side. Below this a cell's own edge cannot hold the
 *  lattice's own worst-case label without overrunning into the next cell. A uniform floor over
 *  the whole lattice, not a per-cell one, so neighboring cells of the same size never disagree
 *  on whether a label is shown. */
export function cellLabelFloorPx(cells: { name: string }[]): number {
  const widestName = cells.reduce((w, c) => Math.max(w, c.name.length), 1);
  const widestLabel = "M".repeat(widestName) + WIDEST_COUNT_SUFFIX;
  return Math.ceil(measureLabelWidth(widestLabel, LABEL_FONT_PX)) + 2 * LABEL_INSET_PX;
}

/** #C9A24B, tcip-season-3 (tailwind.config.ts): the recorded sweep fill. Off green deliberately:
 *  tcip-accent (the platform's own SI_GREEN) reads against an orchard mosaic's own foliage. */
const SWEPT_RGB = "201, 162, 75";
/** #E6976B, tcip-warn: the attested border and the stale strike. */
const ATTEST_RGB = "230, 151, 107";
/** #E7E5DC, tcip-fg: the per-cell border and label, the light half of the two-tone halo. */
const BORDER_RGB = "231, 229, 220";
/** #1E1E1E, tcip-bg: the dark half of the halo under every border, dashed stroke and label, so
 *  a mark reads against bright or green ground the light tone alone would vanish into. */
const HALO_RGB = "30, 30, 30";

export function CoverageOverlay(props: {
  cells: GridCell[];
  /** The visible image region in image coords; null (not yet measured) draws nothing. */
  viewport: PixelRect | null;
  /** Screen px per image px, so strokes and labels hold a constant screen size. */
  scale: number;
  swept: ReadonlySet<string>;
  /** Swept cells not yet acknowledged by the server: drawn with a short-dash stroke and no
   *  fill instead of the recorded fill, so pending and recorded never differ by fill alone. */
  pending: ReadonlySet<string>;
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
  const labelFloorPx = cellLabelFloorPx(props.cells);

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
        const labelFits = w * s >= labelFloorPx && h * s >= labelFloorPx;
        const dash: [number, number] = [4 * strokeW, 3 * strokeW];
        const pendingDash: [number, number] = [1.5 * strokeW, 1.5 * strokeW];
        const swept = props.swept.has(cell.name);
        const pending = swept && props.pending.has(cell.name);
        const labelX = cell.x0 + 2 / s;
        const labelY = cell.y0 + 2 / s;
        const labelText = count > 0 ? `${cell.name} (${count})` : cell.name;
        const labelSize = LABEL_FONT_PX / s;
        return (
          <Group key={cell.name}>
            {swept && (
              <>
                {!pending && (
                  <Rect
                    x={cell.x0}
                    y={cell.y0}
                    width={w}
                    height={h}
                    fill={`rgba(${SWEPT_RGB}, 0.18)`}
                  />
                )}
                <Rect
                  x={cell.x0 + strokeW}
                  y={cell.y0 + strokeW}
                  width={Math.max(0, w - 2 * strokeW)}
                  height={Math.max(0, h - 2 * strokeW)}
                  stroke={`rgba(${HALO_RGB}, 0.85)`}
                  strokeWidth={strokeW * 2.2}
                  dash={pending ? pendingDash : dash}
                />
                <Rect
                  x={cell.x0 + strokeW}
                  y={cell.y0 + strokeW}
                  width={Math.max(0, w - 2 * strokeW)}
                  height={Math.max(0, h - 2 * strokeW)}
                  stroke={`rgba(${SWEPT_RGB}, 0.95)`}
                  strokeWidth={strokeW}
                  dash={pending ? pendingDash : dash}
                />
              </>
            )}
            <Rect
              x={cell.x0}
              y={cell.y0}
              width={w}
              height={h}
              stroke={`rgba(${HALO_RGB}, 0.55)`}
              strokeWidth={strokeW * 2}
            />
            <Rect
              x={cell.x0}
              y={cell.y0}
              width={w}
              height={h}
              stroke={`rgba(${BORDER_RGB}, 0.85)`}
              strokeWidth={strokeW}
            />
            {attested && (
              <Rect
                x={cell.x0 + strokeW}
                y={cell.y0 + strokeW}
                width={Math.max(0, w - 2 * strokeW)}
                height={Math.max(0, h - 2 * strokeW)}
                stroke={`rgba(${ATTEST_RGB}, ${active || stale ? 0.9 : 0.6})`}
                strokeWidth={strokeW * (active || stale ? 1.6 : 1.2)}
                dash={other ? [1.3 * strokeW, 1.7 * strokeW] : undefined}
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
              <>
                <Text
                  x={labelX}
                  y={labelY}
                  text={labelText}
                  fontSize={labelSize}
                  fill={`rgba(${HALO_RGB}, 0.9)`}
                  fontStyle="bold"
                  shadowColor={`rgba(${HALO_RGB}, 1)`}
                  shadowBlur={labelSize * 0.3}
                  shadowOffset={{ x: 0, y: 0 }}
                  shadowOpacity={0.9}
                />
                <Text
                  x={labelX}
                  y={labelY}
                  text={labelText}
                  fontSize={labelSize}
                  fill={`rgba(${BORDER_RGB}, 0.95)`}
                />
              </>
            )}
          </Group>
        );
      })}
    </>
  );
}
