import { useEffect, useRef, useState } from "react";

import { api } from "@/api/client";
import { messageOf } from "@/api/http";
import type { OverviewJob } from "@/api/types.generated";
import { OVERVIEWS_REQUIRED } from "@/api/types.generated";
import { pollJob } from "@/hooks/jobPoll";

/** How long the reported progress may sit at one value before the build is called stalled.
 *
 *  A build that is working reports a growing sidecar; one whose worker died reports the same
 *  number forever, and without this the viewer waits on it forever. A documented threshold: no
 *  measurement of how far apart a real build's progress reports fall on a large raster, so it is set well past
 *  what a single pyramid level should take to show any movement at all. */
export const STALL_MS = 180_000;

export interface OverviewBuildState {
  /** A build for this image is running: the viewer has a wait to show, not a failure. */
  building: boolean;
  /** The sidecar's size on disk as the build last reported it: 0 until it reports, no total
   *  known while it runs, and a build refused over an existing pyramid reports none. */
  bytesWritten: number;
  /** Why the build stopped, when it did. */
  error: string | null;
  /** Bumps once the pyramid exists, so the image request can be made again. */
  reloadToken: number;
}

const IDLE: OverviewBuildState = { building: false, bytesWritten: 0, error: null, reloadToken: 0 };

/** A sidecar byte count as a person reads it, in decimal megabytes. */
export function writtenLabel(bytes: number): string {
  return `${(bytes / 1_000_000).toFixed(1)} MB written`;
}

/**
 * Turn a refused image load into the build that makes it servable.
 *
 * `imageError` is the condition the shared image loader read from a failed load's
 * X-TCIP-Image-Error (null while loading or served); the overviews condition starts a build of
 * `imagePath`'s reduced-resolution pyramid, at most one per `imageUrl`. The state answers the
 * build while it runs, its error once it fails or its progress holds for `STALL_MS`, and a bumped
 * `reloadToken` once it completes.
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

    let canceled = false;
    let stop: (() => void) | null = null;
    let lastWritten = -1;
    let lastMoved = Date.now();
    const failed = (e: unknown) =>
      setState((s) => ({ ...s, building: false, error: messageOf(e) }));

    const onRead = (job: OverviewJob) => {
      if (!job.live) {
        setState((s) =>
          job.status === "completed"
            ? {
                building: false,
                bytesWritten: job.bytes_written,
                error: null,
                reloadToken: s.reloadToken + 1,
              }
            : { ...s, building: false, error: job.error ?? `build ${job.status}` },
        );
      } else if (job.bytes_written !== lastWritten) {
        lastWritten = job.bytes_written;
        lastMoved = Date.now();
        setState((s) => ({ ...s, bytesWritten: job.bytes_written }));
      } else if (Date.now() - lastMoved > STALL_MS) {
        stop?.();
        setState((s) => ({
          ...s,
          building: false,
          error: `the build stopped reporting progress (held at ${writtenLabel(
            job.bytes_written,
          )} for over ${Math.round(STALL_MS / 1000)}s)`,
        }));
      }
    };

    setState((s) => ({ ...s, building: true, bytesWritten: 0, error: null }));
    void (async () => {
      try {
        const job = await api.images.buildOverviews({ path: imagePath });
        if (!canceled) stop = pollJob(job.job_id, api.images.overviewJob, onRead, failed);
      } catch (e) {
        if (!canceled) failed(e);
      }
    })();

    return () => {
      canceled = true;
      stop?.();
    };
  }, [imageError, imageUrl, imagePath]);

  return state;
}
