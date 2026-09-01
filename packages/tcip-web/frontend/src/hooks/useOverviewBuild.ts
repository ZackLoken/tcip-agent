import { useEffect, useRef, useState } from "react";

import { api } from "@/api/client";
import { OVERVIEWS_REQUIRED } from "@/api/types.generated";

/** How often the build's progress is read back. Long enough that a minutes-long pyramid build
 *  is not polled thousands of times, short enough that the bar moves while it runs. */
const POLL_MS = 700;

/** How long the reported progress may sit at one value before the build is called stalled.
 *
 *  A build that is working reports a rising fraction; one whose worker died reports the same
 *  number forever, and without this the viewer waits on it forever. A documented threshold: no
 *  measurement of how far apart a real build's progress reports fall on a large raster, so it is set well past
 *  what a single pyramid level should take to show any movement at all. */
export const STALL_MS = 180_000;

export interface OverviewBuildState {
  /** A build for this image is running: the viewer has a wait to show, not a failure. */
  building: boolean;
  /** Completion fraction in [0, 1], as the build reports it. */
  progress: number;
  /** Why the build stopped, when it did. */
  error: string | null;
  /** Bumps once the pyramid exists, so the image request can be made again. */
  reloadToken: number;
}

const IDLE: OverviewBuildState = { building: false, progress: 0, error: null, reloadToken: 0 };

/**
 * Turn a refused image load into the build that makes it servable.
 *
 * A raster past the server's display bound has no resolution a whole view can be served at until
 * its reduced-resolution pyramid exists; the server refuses such a request naming that condition
 * in X-TCIP-Image-Error. `imageError` is that already-read condition from the shared image
 * loader's failed load (null while loading or served), so no second request is made here; only
 * the overviews condition starts a build.
 *
 * One build is started per image URL: a build that fails is reported, never retried in a loop, and
 * one whose reported progress stops moving for `STALL_MS` is reported too rather than polled on.
 */
export function useOverviewBuild(
  imageUrl: string | null,
  imagePath: string | null,
  imageError: string | null,
): OverviewBuildState {
  const [state, setState] = useState<OverviewBuildState>(IDLE);
  const attempted = useRef<string | null>(null);

  useEffect(() => {
    if (attempted.current !== imageUrl) setState(IDLE);
  }, [imageUrl]);

  useEffect(() => {
    if (imageError !== OVERVIEWS_REQUIRED || !imageUrl || !imagePath) return;
    if (attempted.current === imageUrl) return;
    attempted.current = imageUrl;

    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let lastProgress = -1;
    let lastMoved = Date.now();

    const poll = (jobId: string) => {
      timer = setTimeout(() => {
        void api.images.overviewJob(jobId).then(
          (job) => {
            if (cancelled) return;
            if (job.status === "completed") {
              setState((s) => ({
                building: false,
                progress: 1,
                error: null,
                reloadToken: s.reloadToken + 1,
              }));
              return;
            }
            if (job.status === "failed") {
              setState((s) => ({ ...s, building: false, error: job.error ?? "build failed" }));
              return;
            }
            if (job.progress !== lastProgress) {
              lastProgress = job.progress;
              lastMoved = Date.now();
            } else if (Date.now() - lastMoved > STALL_MS) {
              setState((s) => ({
                ...s,
                building: false,
                error: `the build stopped reporting progress (held at ${Math.round(
                  job.progress * 100,
                )}% for over ${Math.round(STALL_MS / 1000)}s)`,
              }));
              return;
            }
            setState((s) => ({ ...s, progress: job.progress }));
            poll(jobId);
          },
          (e: unknown) => {
            if (!cancelled) {
              setState((s) => ({
                ...s,
                building: false,
                error: e instanceof Error ? e.message : String(e),
              }));
            }
          },
        );
      }, POLL_MS);
    };

    setState((s) => ({ ...s, building: true, progress: 0, error: null }));
    void (async () => {
      try {
        const job = await api.images.buildOverviews(imagePath);
        if (cancelled) return;
        if (job.status === "completed") {
          setState((s) => ({
            building: false,
            progress: 1,
            error: null,
            reloadToken: s.reloadToken + 1,
          }));
          return;
        }
        poll(job.job_id);
      } catch (e) {
        if (!cancelled) {
          setState((s) => ({
            ...s,
            building: false,
            error: e instanceof Error ? e.message : String(e),
          }));
        }
      }
    })();

    return () => {
      cancelled = true;
      if (timer != null) clearTimeout(timer);
    };
  }, [imageError, imageUrl, imagePath]);

  return state;
}
