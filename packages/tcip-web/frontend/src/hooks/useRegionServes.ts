/**
 * The cell-aligned region serves the current viewport needs when the user zooms past the base
 * bitmap's resolution on a large raster. Fetch plans come from planRegionFetches (resolution
 * tiers plus the canvas-derived fan-out caps) against the zoom-independent region-serving grid
 * (`servingCells`), never the coverage lattice: the two lattices are unrelated. A coverage cell
 * (`coverageCells`) marks as served at native once every serving cell that intersects it has
 * itself been served at native, so a coverage cell larger than a serving cell (a low set zoom)
 * still marks, never only one sized to fit inside a single serving cell, and the coverage
 * tracker's own facts stay keyed to the lattice the breeder actually sees. A band-composited
 * serve (a bands param in force, server-derived stretch) takes no region overlay: base and
 * overlay must never be two renderings of the same pixels.
 */

import { useRef } from "react";

import { api } from "@/api/client";
import type { CanvasRegion } from "@/components/Canvas/CanvasStage";
import {
  cellsIntersecting,
  planRegionFetches,
  servedCellAtNative,
  type GridCell,
} from "@/lib/coverage";
import { measureCanvasHost, computeViewport } from "@/lib/canvasSync";
import type { LoadedImage } from "@/lib/imageLoader";
import type { ViewState } from "@/store/types";

export function useRegionServes(args: {
  imagePath: string | null;
  imgW: number;
  imgH: number;
  view: ViewState;
  servingCells: GridCell[];
  servingTileSize: number | null;
  coverageCells: GridCell[];
  baseFacts: LoadedImage | null;
  composite: { bands?: string; stretch?: string };
  onCellServedAtNative?: (cellName: string) => void;
}): CanvasRegion[] {
  const prevRef = useRef<{ signature: string; regions: CanvasRegion[] }>({
    signature: "",
    regions: [],
  });
  // Serving cells (by name) served at native resolution so far for the open image: reset
  // whenever the image changes, since the serving grid itself is per-image.
  const nativeServingRef = useRef<{ imagePath: string | null; served: Set<string> }>({
    imagePath: null,
    served: new Set(),
  });
  if (nativeServingRef.current.imagePath !== args.imagePath) {
    nativeServingRef.current = { imagePath: args.imagePath, served: new Set() };
  }

  const install = (signature: string, regions: CanvasRegion[]): CanvasRegion[] => {
    if (prevRef.current.signature === signature) return prevRef.current.regions;
    prevRef.current = { signature, regions };
    return regions;
  };

  const served = args.baseFacts?.ok ? args.baseFacts.servedSize : null;
  if (
    !args.imagePath ||
    args.servingCells.length === 0 ||
    !args.servingTileSize ||
    args.composite.bands !== undefined ||
    !served ||
    args.imgW <= 0 ||
    served.w >= args.imgW
  ) {
    return install("", []);
  }
  const host = measureCanvasHost();
  if (!host) return install("", []);
  const viewport = computeViewport(args.view, host, args.imgW, args.imgH);
  if (!viewport) return install("", []);

  const plan = planRegionFetches({
    cells: args.servingCells,
    viewport: {
      x0: viewport.x,
      y0: viewport.y,
      x1: viewport.x + viewport.w,
      y1: viewport.y + viewport.h,
    },
    scale: args.view.scale,
    baseScale: served.w / args.imgW,
    host,
    tileSize: args.servingTileSize,
  });
  if (!plan || plan.length === 0) return install("", []);

  const imagePath = args.imagePath;
  const onServed = args.onCellServedAtNative;
  const coverageCells = args.coverageCells;
  const servingCells = args.servingCells;
  const nativeServing = nativeServingRef.current.served;
  const regions = plan.map((p) => {
    const { cell } = p;
    const url = api.images.url(imagePath, {
      ...args.composite,
      x0: cell.x0,
      y0: cell.y0,
      x1: cell.x1,
      y1: cell.y1,
      max_width: p.maxWidth,
    });
    return {
      key: cell.name,
      url,
      x: cell.x0,
      y: cell.y0,
      width: cell.x1 - cell.x0,
      height: cell.y1 - cell.y0,
      onLoaded: (facts: LoadedImage) => {
        if (!facts.ok || !servedCellAtNative(cell, facts.servedSize)) return;
        if (nativeServing.has(cell.name)) return;
        nativeServing.add(cell.name);
        for (const covCell of coverageCells) {
          const intersecting = cellsIntersecting(servingCells, covCell);
          if (intersecting.length > 0 && intersecting.every((sc) => nativeServing.has(sc.name))) {
            onServed?.(covCell.name);
          }
        }
      },
    };
  });
  return install(regions.map((r) => `${r.key}|${r.url}`).join(","), regions);
}
