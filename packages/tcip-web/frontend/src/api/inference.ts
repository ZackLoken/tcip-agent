/** Inference + Results API helpers for the Inference and Results tabs. */

import { getJson, postForBlob, postJson, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import {
  type Assessments,
  type BuildMappingPayload,
  type CanopySegmentDisclosure,
  type ConfirmedRevision,
  type ConfirmRevisionPayload,
  type CountExport,
  type DeliveryEventRecord,
  type DeliveryEvents,
  EXPORT_HEADER,
  type ExportCountCsvPayload,
  type ExportCsvPayload,
  type InferenceFrame,
  type InferenceJob,
  type InferenceJobs,
  type LaunchInferencePayload,
  type LoadMappingPayload,
  type MappingNames,
  type PersonPayload,
  type PhenologyAnswer,
  type PhenologyPayload,
  type PlantMappingDisclosure,
  type PlantRegistryDisclosure,
  type RegisteredModels,
  type SavedExport,
  type ServedMapping,
  type TraitsListing,
} from "@/api/types.generated";
import { createReconnectingSocket, jsonFrameHandlers } from "@/lib/reconnectingSocket";

export const inferenceApi = {
  launch: (body: LaunchInferencePayload) =>
    postJson<InferenceJob>(ROUTES.postInferenceLaunch, body),

  listJobs: () => getJson<InferenceJobs>(ROUTES.getInferenceJobs),

  cancel: (jobId: string, body: PersonPayload) =>
    postJson<InferenceJob>(ROUTES.postInferenceJobsByJobIdCancel(jobId), body),
};

/**
 * Open a live progress stream for an inference job, auto-reconnecting with capped
 * backoff if the socket drops mid-run. The server sends a single ``final`` frame at a
 * terminal state and closes; once seen we stop reconnecting.
 */
export function openInferenceStream(
  jobId: string,
  onMessage: (frame: InferenceFrame) => void,
): () => void {
  const socket = createReconnectingSocket({
    url: wsUrl(ROUTES.socketInferenceJobsByJobIdStream(jobId)),
    ...jsonFrameHandlers<InferenceFrame>(onMessage, (frame) => frame.type === "final"),
  });
  socket.start();
  return () => socket.stop();
}

export type PlantMappingUnion = NonNullable<DeliveryEventRecord["plant_mapping"]>;

/** Whether `pm` is the canopy-segment disclosure, narrowed first. */
export function isCanopySegmentDisclosure(pm: PlantMappingUnion): pm is CanopySegmentDisclosure {
  return "canopy_segments" in pm;
}

/** Whether `pm` is the whole-raster registry disclosure, narrowed after the canopy shape. */
export function isPlantRegistryDisclosure(pm: PlantMappingUnion): pm is PlantRegistryDisclosure {
  return !isCanopySegmentDisclosure(pm) && "plant_registry" in pm;
}

/** Whether `pm` is a walked mapping's disclosure, narrowed last. */
export function isPlantMappingDisclosure(pm: PlantMappingUnion): pm is PlantMappingDisclosure {
  return "name" in pm && "record_sha256" in pm;
}

export const resultsApi = {
  registeredModels: () => getJson<RegisteredModels>(ROUTES.getResultsModelsRegistered),

  // Every assessment recorded under the open project, each naming the checkpoint it judged.
  assessments: () => getJson<Assessments>(ROUTES.getResultsAssessments),

  // The open project's own trait records, each revision with whether its confirmation stands.
  traits: () => getJson<TraitsListing>(ROUTES.getResultsTraits),

  // Refuses with 409 when the hash is not the revision's own: the revision shown was not the one
  // on file.
  confirmTraitRevision: (body: ConfirmRevisionPayload) =>
    postJson<ConfirmedRevision>(ROUTES.postResultsTraitsConfirm, body),

  buildPlantMapping: (body: BuildMappingPayload) =>
    postJson<ServedMapping>(ROUTES.postResultsPlantMappingBuild, body),

  // Refuses with 404 when nothing is stored under the name.
  loadPlantMapping: (body: LoadMappingPayload) =>
    postJson<ServedMapping>(ROUTES.postResultsPlantMappingLoad, body),

  // Every mapping name persisted under the open project.
  listPlantMappings: () => getJson<MappingNames>(ROUTES.getResultsPlantMappingList),

  phenologyMeasurement: (body: PhenologyPayload) =>
    postJson<PhenologyAnswer>(ROUTES.postResultsPhenologyMeasurement, body),

  /** Every delivery event the open project holds: what shipped, under which trait and kind. */
  deliveryEvents: () => getJson<DeliveryEvents>(ROUTES.getResultsDeliveryEvents),

  /** The CSV the server computes for `body`'s phenology export, beside the export its answer
   *  reports. */
  downloadCsv: async (body: ExportCsvPayload): Promise<{ blob: Blob; exported: SavedExport }> => {
    const { blob, headers } = await postForBlob(ROUTES.postResultsExportCsv, body);
    return { blob, exported: reportedExport<SavedExport>(headers) };
  },

  /** The count CSV the server computes for `body`, beside the export its answer reports. */
  downloadCountCsv: async (
    body: ExportCountCsvPayload,
  ): Promise<{ blob: Blob; exported: CountExport }> => {
    const { blob, headers } = await postForBlob(ROUTES.postResultsExportCountCsv, body);
    return { blob, exported: reportedExport<CountExport>(headers) };
  },
};

/** The export an answer's `headers` report under `EXPORT_HEADER`; an answer without it refuses,
 *  naming it. */
export function reportedExport<T extends SavedExport>(headers: Headers): T {
  const sent = headers.get(EXPORT_HEADER);
  if (sent === null) throw new Error(`the export answer carries no ${EXPORT_HEADER} header`);
  return JSON.parse(sent) as T;
}
