import type { StateCreator } from "zustand";

import type { InferenceJob, TrainingListing } from "@/api/types.generated";
import type { AppState } from "@/store/appState";

/** One listing as read for the open project: what its last answered read held, and the message
 *  of its last read while that read failed, null once one answers. */
export interface ListingRead<T> {
  answer: T;
  failure: string | null;
}

/** The background jobs of the open project as the shell's reader lists them: the training
 *  listing, and the inference jobs by id. */
export interface JobListings {
  training: ListingRead<TrainingListing>;
  inference: ListingRead<Record<string, InferenceJob>>;
}

export type ListingName = keyof JobListings;

/** Every listing empty, as before any read for a project. */
export const NO_JOB_LISTINGS: JobListings = {
  training: { answer: { runs: [], sweeps: [] }, failure: null },
  inference: { answer: {}, failure: null },
};

/** What one listing's last answer for a project told: the ids it listed, and each job still
 *  watched with its name as the person reads it and the state last read. */
export interface JobsTold {
  listed: string[];
  watched: Record<string, { name: string; state: string }>;
}

export interface JobListingsSlice {
  /** Emptied whenever the open project changes, so one project's jobs never show for another. */
  jobListings: JobListings;
  /** How many times each listing was asked to be read now; the reader reads on each change. */
  listingAsks: Record<ListingName, number>;
  /** What each listing's last answer told, by project id, for the page's session: a change of
   *  the open project keeps it, so a project opened again is told what changed meanwhile. */
  jobsTold: Record<string, Partial<Record<ListingName, JobsTold>>>;
  /** Hold `answer` and `failure` as `name`'s listing read for the project `project` names and
   *  return true; a read for a project that is no longer the open one is dropped, returning
   *  false. */
  holdListing: <N extends ListingName>(
    name: N,
    project: string | null,
    answer: JobListings[N]["answer"],
    failure: string | null,
  ) => boolean;
  /** Ask the reader to read `name` now, or once more as soon as its read in flight answers. */
  askListing: (name: ListingName) => void;
  /** Keep `told` as what `name`'s last answer for the project `project` told. */
  keepJobsTold: (project: string, name: ListingName, told: JobsTold) => void;
}

export const createJobListingsSlice: StateCreator<AppState, [], [], JobListingsSlice> = (
  set,
  get,
) => ({
  jobListings: NO_JOB_LISTINGS,
  listingAsks: { training: 0, inference: 0 },
  jobsTold: {},
  holdListing: (name, project, answer, failure) => {
    const held = (get().openProject?.id ?? null) === project;
    if (held) set((s) => ({ jobListings: { ...s.jobListings, [name]: { answer, failure } } }));
    return held;
  },
  askListing: (name) =>
    set((s) => ({ listingAsks: { ...s.listingAsks, [name]: s.listingAsks[name] + 1 } })),
  keepJobsTold: (project, name, told) =>
    set((s) => ({
      jobsTold: { ...s.jobsTold, [project]: { ...s.jobsTold[project], [name]: told } },
    })),
});
