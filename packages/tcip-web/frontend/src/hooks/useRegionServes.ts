/**
 * The reads the current view needs when the user zooms past the base bitmap's resolution on a
 * large raster. The server is asked, for this display, which reads serve the visible native-pixel
 * rect: the native cells it intersects, the tiles of the overview level it plans the view off, or
 * one display read of the rect; each read names its level, its rect in that level's grid and
 * where it lies in native pixels. Nothing here sizes a read or budgets one. A band-composited serve (a bands param in force, server-derived stretch) takes
 * no region overlay: base and overlay must never be two renderings of the same pixels.
 */

import { useEffect, useMemo, useState } from "react";

import { api } from "@/api/client";
import type { ServingCell } from "@/api/types.generated";
import type { CanvasRegion } from "@/components/Canvas/CanvasStage";
import { measureCanvasHost, computeViewport } from "@/lib/canvasSync";
import type { LoadedImage } from "@/lib/imageLoader";
import type { ViewState } from "@/store/types";

export function useRegionServes(args: {
  imagePath: string | null;
  imgW: number;
  imgH: number;
  view: ViewState;
  displayPixels: number;
  baseFacts: LoadedImage | null;
  composite: { bands?: string; stretch?: string };
}): CanvasRegion[] {
  const [answer, setAnswer] = useState<{
    path: string;
    display: number;
    reads: ServingCell[];
  } | null>(null);

  const { imagePath, imgW, imgH, displayPixels } = args;
  const { bands, stretch } = args.composite;
  const served = args.baseFacts?.ok ? args.baseFacts.servedSize : null;
  // The base bitmap's resolution is its coarser axis: a one-pixel-wide frame reduces only in height.
  const baseScale = served && imgW > 0 && imgH > 0 ? Math.min(served.w / imgW, served.h / imgH) : 1;
  const host = measureCanvasHost();
  const viewport =
    imagePath && host && baseScale < 1 && bands === undefined
      ? args.view.scale > baseScale && computeViewport(args.view, host, imgW, imgH)
      : null;
  const x0 = viewport ? Math.floor(viewport.x) : 0;
  const y0 = viewport ? Math.floor(viewport.y) : 0;
  const x1 = viewport ? Math.min(imgW, Math.ceil(viewport.x + viewport.w)) : 0;
  const y1 = viewport ? Math.min(imgH, Math.ceil(viewport.y + viewport.h)) : 0;
  const key = viewport ? `${imagePath}|${displayPixels}|${x0},${y0},${x1},${y1}` : "";

  useEffect(() => {
    if (!key || !imagePath) return;
    let canceled = false;
    void api.images.viewReads(imagePath, displayPixels, { x0, y0, x1, y1 }).then(
      (plan) => {
        if (!canceled) setAnswer({ path: imagePath, display: displayPixels, reads: plan.reads });
      },
      () => {
        // No answer serves the base bitmap alone: region serving is an optimization.
      },
    );
    return () => {
      canceled = true;
    };
  }, [key, imagePath, displayPixels, x0, y0, x1, y1]);

  // The last answer for this image stays on screen while the next view's answer is in flight, so a
  // pan replaces only the reads that changed; a stale answer never lands (the effect cancels it).
  const reads =
    key && answer && answer.path === imagePath && answer.display === displayPixels
      ? answer.reads
      : null;
  return useMemo(
    () =>
      !imagePath || !reads
        ? []
        : reads.map((cell) => ({
            key: `${cell.level}:${cell.name}`,
            url: api.images.url(imagePath, displayPixels, {
              bands,
              stretch,
              x0: cell.x0,
              y0: cell.y0,
              x1: cell.x1,
              y1: cell.y1,
              level: cell.level || undefined,
            }),
            x: cell.nx0,
            y: cell.ny0,
            width: cell.nx1 - cell.nx0,
            height: cell.ny1 - cell.ny0,
          })),
    [imagePath, reads, displayPixels, bands, stretch],
  );
}
