/**
 * Coverage minimap for a multi-cell raster: a small thumbnail of the whole image with per-cell
 * swept shading and the current viewport rectangle; clicking or dragging jumps the canvas to
 * the cell under the pointer. Collapsed to a pill by default so it never covers canvas content
 * unless opened; the pill sits bottom-right, completing the canvas' floating-chrome grammar
 * (legend bottom-left, attributes top-right).
 */

import { useEffect, useRef, useState } from "react";

import { api } from "@/api/client";
import { useDisclosure } from "@/hooks/useDisclosure";
import { indexCells, sweptFractionBlocks, type GridCell, type GridGeometry } from "@/lib/coverage";
import type { PixelRect } from "@/lib/viewGeometry";

// Swept shading uses the app accent (tcip-accent #507754) so covered work reads as the brand
// green; the viewport rectangle uses the foreground paper tone (tcip-fg #E7E5DC).
const SWEPT_RGB = "80, 119, 84";
const VIEWPORT_STROKE = "#E7E5DC";

export function CoverageMinimap(props: {
  imagePath: string;
  composite: { bands?: string; stretch?: string };
  grid: GridGeometry;
  cells: GridCell[];
  swept: ReadonlySet<string>;
  /** Bumps when the swept set changes (the set mutates in place); triggers a redraw. */
  sweptVersion: number;
  /** The visible image region in image coords, or null when unknown. */
  viewport: PixelRect | null;
  onJump: (cell: GridCell) => void;
}) {
  // Closed by default: the map must not cover canvas content unless asked for.
  const { open, toggle } = useDisclosure("tcip.annotate.minimapOpen");
  const boxRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const draggingRef = useRef(false);
  const [cssWidth, setCssWidth] = useState(0);

  useEffect(() => {
    if (!open || !boxRef.current) return;
    const measure = () => {
      const r = boxRef.current!.getBoundingClientRect();
      if (r.width > 1) setCssWidth(r.width);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(boxRef.current);
    return () => ro.disconnect();
  }, [open]);

  const aspect = props.grid.width > 0 ? props.grid.height / props.grid.width : 1;
  const dpr = typeof devicePixelRatio === "number" && devicePixelRatio > 0 ? devicePixelRatio : 1;
  const deviceW = Math.max(1, Math.round(cssWidth * dpr));
  const deviceH = Math.max(1, Math.round(cssWidth * aspect * dpr));

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!open || !canvas || cssWidth <= 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, deviceW, deviceH);
    const sx = deviceW / props.grid.width;
    const sy = deviceH / props.grid.height;

    const index = indexCells(props.cells);
    const cellDevicePx = props.grid.tile_size * Math.min(sx, sy);
    // Below 2 device pixels a cell cannot render distinguishably (the visibility floor), so
    // shading aggregates k x k cell blocks by swept fraction instead.
    const k = cellDevicePx >= 2 ? 1 : Math.ceil(2 / Math.max(cellDevicePx, 1e-6));
    if (k === 1) {
      ctx.fillStyle = `rgba(${SWEPT_RGB}, 0.5)`;
      for (const cell of props.cells) {
        if (!props.swept.has(cell.name)) continue;
        ctx.fillRect(
          cell.x0 * sx,
          cell.y0 * sy,
          (cell.x1 - cell.x0) * sx,
          (cell.y1 - cell.y0) * sy,
        );
      }
    } else {
      const blocks = sweptFractionBlocks(index.cols, index.rows, k, (col, row) => {
        const cell = index.at(col, row);
        return !!cell && props.swept.has(cell.name);
      });
      for (let by = 0; by < blocks.rows; by++) {
        for (let bx = 0; bx < blocks.cols; bx++) {
          const fraction = blocks.fractions[by * blocks.cols + bx];
          if (fraction <= 0) continue;
          const x0 = index.colX[bx * k][0];
          const x1 = index.colX[Math.min(index.cols - 1, (bx + 1) * k - 1)][1];
          const y0 = index.rowY[by * k][0];
          const y1 = index.rowY[Math.min(index.rows - 1, (by + 1) * k - 1)][1];
          ctx.fillStyle = `rgba(${SWEPT_RGB}, ${0.5 * fraction})`;
          ctx.fillRect(x0 * sx, y0 * sy, (x1 - x0) * sx, (y1 - y0) * sy);
        }
      }
    }

    if (props.viewport) {
      ctx.strokeStyle = VIEWPORT_STROKE;
      ctx.lineWidth = dpr;
      ctx.strokeRect(
        props.viewport.x0 * sx,
        props.viewport.y0 * sy,
        (props.viewport.x1 - props.viewport.x0) * sx,
        (props.viewport.y1 - props.viewport.y0) * sy,
      );
    }
  }, [
    open,
    cssWidth,
    deviceW,
    deviceH,
    dpr,
    props.cells,
    props.swept,
    props.sweptVersion,
    props.viewport,
    props.grid,
  ]);

  const jumpAt = (e: React.PointerEvent) => {
    const el = boxRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) return;
    const x = ((e.clientX - r.left) / r.width) * props.grid.width;
    const y = ((e.clientY - r.top) / r.height) * props.grid.height;
    const cell = props.cells.find((c) => x >= c.x0 && x < c.x1 && y >= c.y0 && y < c.y1);
    if (cell) props.onJump(cell);
  };

  return (
    <div className="absolute bottom-3 right-3 z-20 flex flex-col items-end gap-2">
      {open && (
        <div className="w-56 overflow-hidden rounded-md border border-tcip-border bg-tcip-panel/95 shadow-lg backdrop-blur">
          <div className="flex items-center justify-between px-2 py-1">
            <span className="text-[10px] font-semibold uppercase tracking-wide text-tcip-muted">
              Coverage
            </span>
            <button
              type="button"
              onClick={toggle}
              aria-label="Close coverage map"
              title="Close"
              className="text-tcip-muted hover:text-tcip-fg"
            >
              ✕
            </button>
          </div>
          <div
            ref={boxRef}
            className="relative cursor-pointer select-none"
            onPointerDown={(e) => {
              draggingRef.current = true;
              e.currentTarget.setPointerCapture(e.pointerId);
              jumpAt(e);
            }}
            onPointerMove={(e) => {
              if (draggingRef.current) jumpAt(e);
            }}
            onPointerUp={() => {
              draggingRef.current = false;
            }}
          >
            {cssWidth > 0 && (
              <img
                src={api.images.url(props.imagePath, {
                  ...props.composite,
                  // 2x the component's own measured CSS width, so it stays crisp on dense displays.
                  max_width: Math.max(1, Math.round(2 * cssWidth)),
                })}
                alt="Whole-image coverage map"
                className="block w-full"
                draggable={false}
              />
            )}
            <canvas
              ref={canvasRef}
              width={deviceW}
              height={deviceH}
              className="pointer-events-none absolute inset-0 h-full w-full"
            />
          </div>
        </div>
      )}
      {!open && (
        <button
          type="button"
          onClick={toggle}
          className="flex items-center gap-1.5 rounded-full border border-tcip-border bg-tcip-panel/90 px-2.5 py-1 text-[11px] text-tcip-muted backdrop-blur hover:border-tcip-border-hover hover:text-tcip-fg"
        >
          <svg width="12" height="12" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <rect
              x="1.5"
              y="1.5"
              width="13"
              height="13"
              rx="1"
              stroke="currentColor"
              strokeWidth="1.3"
            />
            <path
              d="M6 1.5v13M10 1.5v13M1.5 6h13M1.5 10h13"
              stroke="currentColor"
              strokeWidth="1"
            />
          </svg>
          Coverage
        </button>
      )}
    </div>
  );
}
