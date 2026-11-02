/** Training-tab specific REST + WebSocket helpers. */

import { getJson, postJson, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type {
  CompareBestPayload,
  Comparison,
  ExperimentComparePayload,
  Launched,
  PersonPayload,
  Ranking,
  RelaunchPayload,
  SplitChoices,
  TensorboardLaunch,
  TrainingDetail,
  TrainingFrame,
  TrainingListing,
} from "@/api/types.generated";
import { createReconnectingSocket, jsonFrameHandlers } from "@/lib/reconnectingSocket";

export const trainingApi = {
  listSplitChoices: (experiment_id: string) =>
    getJson<SplitChoices>(ROUTES.getTrainingRunsByExperimentIdSplits(experiment_id)),

  /** Start a new run or sweep from the recorded one ``relaunched_from`` names, on the partition
   *  ``selection_dir`` holds, or the recorded one for null. */
  relaunch: (body: RelaunchPayload) => postJson<Launched>(ROUTES.postTrainingRuns, body),

  listRuns: () => getJson<TrainingListing>(ROUTES.getTrainingRuns),

  getRun: (experiment_id: string) =>
    getJson<TrainingDetail>(ROUTES.getTrainingRunsByExperimentId(experiment_id)),

  launchTensorboard: (experiment_id: string) =>
    postJson<TensorboardLaunch>(
      ROUTES.postTrainingRunsByExperimentIdTensorboard(experiment_id),
      {},
    ),

  cancel: (experiment_id: string, body: PersonPayload) =>
    postJson<TrainingDetail>(ROUTES.postTrainingRunsByExperimentIdCancel(experiment_id), body),

  compare: (body: ExperimentComparePayload) =>
    postJson<Comparison>(ROUTES.postTrainingCompare, body),

  compareBest: (body: CompareBestPayload) =>
    postJson<Ranking>(ROUTES.postTrainingCompareBest, body),
};

/**
 * Open a live metrics stream for a training run or trial of the open project, auto-reconnecting
 * with capped backoff.
 * The server replays all rows from the start on each (re)connect, so the consumer must
 * dedupe by epoch. A ``batch`` frame carries the latest per-batch row a read found, for a run
 * that logs them. A ``status`` frame is terminal; an ``error`` frame names an id no record claims
 * (selected at its launch moment, before the record exists, or simply unknown), and the socket
 * keeps reconnecting under backoff until a status arrives. An error frame never resets the
 * backoff, so the reconnect delay grows to the cap while the run stays unknown.
 */
export function openTrainingStream(
  experiment_id: string,
  onMessage: (msg: TrainingFrame) => void,
): () => void {
  const url = wsUrl(ROUTES.socketTrainingRunsByExperimentIdStream(experiment_id));
  const socket = createReconnectingSocket({
    url,
    ...jsonFrameHandlers<TrainingFrame>(
      onMessage,
      (frame) => frame.type === "status",
      (frame) => frame.type !== "error",
    ),
  });
  socket.start();
  return () => socket.stop();
}
