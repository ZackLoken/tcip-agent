/** The open project's background jobs, listed above the tabs whatever tab is active. */

import { useEffect } from "react";

import { messageOf } from "@/api/http";
import { inferenceApi } from "@/api/inference";
import { trainingApi } from "@/api/training";
import { pollLoop } from "@/hooks/pollLoop";
import { useStore } from "@/store";
import type { JobListings, ListingName } from "@/store/slices/jobListings";
import { RUN_REFRESH_MS } from "@/tabs/trainingMetrics";

/** How often the inference job listing is read; 3000 ms is the hand-set interval the Inference tab
 *  polled it at, with no measurement behind it. */
const INFERENCE_REFRESH_MS = 3000;

/** How each listing is read, as the store holds its answer, and how often. */
const LISTINGS: {
  [N in ListingName]: { read: () => Promise<JobListings[N]["answer"]>; intervalMs: number };
} = {
  training: { read: () => trainingApi.listRuns(), intervalMs: RUN_REFRESH_MS },
  inference: {
    read: async () =>
      Object.fromEntries((await inferenceApi.listJobs()).jobs.map((j) => [j.job_id, j])),
    intervalMs: INFERENCE_REFRESH_MS,
  },
};

/** Read the listing `name` for the project `project` names until `project` changes or the
 *  caller unmounts, holding each answer and each failure, and reading now whenever
 *  `askListing(name)` is called. A failed read keeps the listing last answered. */
function useListing<N extends ListingName>(name: N, project: string | null): void {
  useEffect(() => {
    const { read, intervalMs } = LISTINGS[name];
    const { holdListing } = useStore.getState();
    const loop = pollLoop({
      read,
      intervalMs,
      readFirstNow: true,
      onAnswer: (answer) => {
        holdListing(name, project, answer, null);
        return true;
      },
      onFailure: (error) => {
        holdListing(name, project, useStore.getState().jobListings[name].answer, messageOf(error));
        return true;
      },
    });
    const unsubscribe = useStore.subscribe((s, prev) => {
      if (s.listingAsks[name] !== prev.listingAsks[name]) loop.readNow();
    });
    return () => {
      unsubscribe();
      loop.stop();
    };
  }, [name, project]);
}

/** List the open project's training runs and inference jobs into the store for as long as the
 *  caller is mounted, each read waiting for the last one's answer. */
export function useJobListings(): void {
  const project = useStore((s) => s.openProject?.id ?? null);
  useListing("training", project);
  useListing("inference", project);
}
