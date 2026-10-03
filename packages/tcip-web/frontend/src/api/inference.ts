/** Inference + Results API helpers for the Inference and Results tabs. */

import { getJson, postForBlob, postJson, StructuredRefusalError, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type {
  CanopySegmentDisclosure,
  ConfirmRevisionPayload,
  DeliveryEventRecord as StoredDeliveryEventRecord,
  ExportCountCsvPayload,
  ExportCsvPayload,
  JobStatus,
  MatchTolerance,
  PhenologyPayload,
  PlantMappingDisclosure,
  PlantRegistryDisclosure,
  Stated,
  TraitRevision,
} from "@/api/types.generated";
import { createReconnectingSocket, jsonFrameHandlers } from "@/lib/reconnectingSocket";

/** One entry from the project's trained-model registry: the fields of it this UI reads. */
export interface RegisteredModel {
  name: string;
  checkpoint_path: string;
  tags?: string[];
  experiment_id?: string | null;
}

export type InferenceStatus = JobStatus;

export interface InferenceJob {
  job_id: string;
  status: InferenceStatus;
  done: number;
  total: number;
  images_dir: string;
  output_dir: string;
  error: string | null;
  // Set when a line the publishing library writes for this run could not be written; the
  // predictions are on disk regardless.
  audit_warning: string | null;
}

/** A run over one capture date's images into a bucket directory that does not exist yet; every
 *  execution value ``stated`` leaves unset the platform derives from the checkpoint, and an
 *  ``assessment_id`` runs that assessment's execution record and publishes under it. */
export interface LaunchInferenceBody {
  checkpoint_path: string;
  dataset_root: string;
  date: string;
  output_dir: string;
  stated: Stated;
  assessment_id: string | null;
  user: string;
}

export const inferenceApi = {
  launch: (body: LaunchInferenceBody) =>
    postJson<{
      status: string;
      job_id: string;
      images_dir: string;
      output_dir: string;
    }>(ROUTES.postInferenceLaunch, body),

  listJobs: () => getJson<{ jobs: InferenceJob[] }>(ROUTES.getInferenceJobs),

  cancel: (jobId: string, user: string) =>
    postJson<{ job_id: string; status: string; cancel_requested: boolean }>(
      ROUTES.postInferenceJobsByJobIdCancel(jobId),
      { user },
    ),
};

/**
 * Open a live progress stream for an inference job, auto-reconnecting with capped
 * backoff if the socket drops mid-run. The server sends a single ``final`` frame at a
 * terminal state and closes; once seen we stop reconnecting (the job is done, not lost).
 */
export function openInferenceStream(
  jobId: string,
  onMessage: (msg: Record<string, unknown>) => void,
): () => void {
  const socket = createReconnectingSocket({
    url: wsUrl(ROUTES.socketInferenceJobsByJobIdStream(jobId)),
    ...jsonFrameHandlers<Record<string, unknown>>(onMessage, (frame) => frame.type === "final"),
  });
  socket.start();
  return () => socket.stop();
}

export interface PlantMappingDateSummary {
  n_images: number;
  n_mapped: number;
  n_unattributed: number;
  // null on a date with no recorded distance, never a fabricated zero.
  avg_distance_m: number | null;
}

export interface PlantMappingSummary {
  per_date: { [date: string]: PlantMappingDateSummary };
  totals: { n_dates: number; n_images: number; n_mapped: number; n_unattributed: number };
}

// One persisted plant mapping as the build and load doors answer it.
export interface ServedPlantMapping {
  name: string;
  summary: PlantMappingSummary;
  unreadable: Record<string, string[]>;
  nn_tolerance_m: MatchTolerance;
  max_match_distance_m: number;
}

export interface PerPlantRow {
  plant_id: string;
  accession: string | null;
  date: string;
  n_images: number;
  n_total: number;
  n_positive: number;
  n_unclassified: number;
  n_missing: number;
  // null when a detection of this date lacks the state's attribute or an image is missing, never
  // a fabricated ratio.
  ratio: number | null;
}

// The fixed fields below are the columns every phenology delivery carries; the trait's milestone
// date and bound columns arrive as additional keys the response's own `columns` name.
export interface OnsetRow {
  plant_id: string;
  accession: string | null;
  n_dates: number;
  n_dates_unclassified: number;
  n_dates_missing_images: number;
  // Dates with a non-zero-detection observation; a complete plant may still have none.
  n_observed_dates: number;
  // Whether every one of the plant's dates carries the state's attribute on every detection and is
  // fully observed.
  complete: boolean;
  [milestoneColumn: string]: string | number | boolean | null;
}

/** One milestone's columns: its date column and the column holding that date's bound. */
export interface MilestoneColumn {
  date: string;
  bound: string;
}

/** The count-export door's own response headers: present on every response, the acknowledging
 *  user an empty string (never a rendering of null/undefined) when the delivery is validated. */
export interface ExportCountCsvHeaders {
  savedTo: string;
  validated: boolean;
  acknowledgedBy: string;
}

// One door returns both projections from one server-side measurement, so no surface can render
// either projection bare, and a milestone date and the curve it was read off cannot disagree.
export interface PhenologyMeasurementResponse {
  curves: { rows: PerPlantRow[]; n_plants: number };
  milestones: { rows: OnsetRow[]; columns: MilestoneColumn[] };
  // Whether an assessment answers for every delivered bucket; when not, unvalidated_reason is the
  // gate's refusal and result_sha256 each projection's digest a breeder's acknowledgment of
  // exporting it binds to.
  validated: boolean;
  unvalidated_reason: string | null;
  result_sha256: Record<"curves" | "milestones", string> | null;
  // The missingness rule the rows were computed under.
  require_all_dates_complete: boolean;
  trait: string;
  trait_revision: number;
  trait_revision_sha256: string;
  // False when no bucket's scope declares the positive state's attribute: the ratios are then not
  // a valid phenology measurement.
  positive_class_assessed: boolean;
  // What this delivery could not verify, not merely what it did not read: a bare date omitted,
  // absent, or archived (predictions still counted), or "date/name" for one uncheckable capture.
  captures_unverified: string[];
  // A plant CSV the mapping was built from that moved since. Empty when nothing was unverified.
  plant_csvs_unverified: string[];
  // This delivery's own delivered dates, and its unattributed-capture count scoped to them
  // (never the mapping's own n_dates_missing_images span).
  dates_delivered: string[];
  images_unattributed: number;
}

/** One revision as the trait routes serve it: the stored revision and whether its confirmation
 *  stands. */
export type ServedTraitRevision = TraitRevision & { confirmed: boolean };

/** One trait's stored record as the traits route serves it, with the number of the revision a
 *  delivery reads (`null` while none is confirmed). */
export interface ServedTraitRecord {
  trait: string;
  revisions: ServedTraitRevision[];
  latest_confirmed: number | null;
}

/** The traits route's one read: every trait's record, each trait whose record its schema refuses,
 *  and crops.yml's own definition of every phenotype a revision delivers. */
export interface TraitsListing {
  traits: ServedTraitRecord[];
  unreadable: { trait: string; reason: string }[];
  definitions: Record<string, string>;
}

/** A delivery door's refusal that a trait's delivered number has no confirmed meaning. */
export interface OperationalizationRefusal {
  kind: "operationalization";
  message: string;
}

/** The operationalization refusal a thrown error carries, read by its detail's `kind`, or null for
 *  every other failure. */
export function operationalizationRefusalOf(e: unknown): OperationalizationRefusal | null {
  if (!(e instanceof StructuredRefusalError)) return null;
  const detail = e.detail;
  if (detail.kind !== "operationalization") return null;
  return {
    kind: "operationalization",
    message: typeof detail.message === "string" ? detail.message : e.message,
  };
}

/** A delivery door's own refusal: the delivery gate's sentence naming what answers for no
 *  delivered bucket, and the digest of the result it computed when a breeder's acknowledgment can
 *  ship it unvalidated (null when none can). */
export interface DeliveryRefusal {
  kind: "delivery";
  message: string;
  result_sha256: string | null;
}

/** The delivery refusal a thrown error carries, read by kind off its parsed detail, or null for
 *  every other failure. */
export function deliveryRefusalOf(e: unknown): DeliveryRefusal | null {
  if (!(e instanceof StructuredRefusalError)) return null;
  const detail = e.detail;
  if (detail.kind !== "delivery") return null;
  return {
    kind: "delivery",
    message: typeof detail.message === "string" ? detail.message : e.message,
    result_sha256: typeof detail.result_sha256 === "string" ? detail.result_sha256 : null,
  };
}

/** The Inference tab's launch refusing the bucket directory a live job is still writing: the
 *  requested path and that job. */
export interface BucketExistsRefusal {
  kind: "bucket_exists";
  message: string;
  date: string | null;
  requested_output_dir: string | null;
  job_id: string;
}

/** The launch's bucket refusal a thrown error carries, read by kind off its parsed detail, or
 *  null for every other failure. */
export function bucketRefusalOf(e: unknown): BucketExistsRefusal | null {
  if (!(e instanceof StructuredRefusalError)) return null;
  const detail = e.detail;
  if (detail.kind !== "bucket_exists" || typeof detail.job_id !== "string") return null;
  return {
    kind: "bucket_exists",
    message: typeof detail.message === "string" ? detail.message : e.message,
    date: typeof detail.date === "string" ? detail.date : null,
    requested_output_dir:
      typeof detail.requested_output_dir === "string" ? detail.requested_output_dir : null,
    job_id: detail.job_id,
  };
}

export type PlantMappingUnion =
  PlantMappingDisclosure | PlantRegistryDisclosure | CanopySegmentDisclosure;

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

/** One completed delivery as the backend serves it: the stored record, and the key its cited
 *  mapping loads under (set only alongside `plant_mapping`). */
export type DeliveryEventRecord = StoredDeliveryEventRecord & {
  plant_mapping_resolved_key?: string | null;
};

export const resultsApi = {
  registeredModels: () => getJson<{ models: RegisteredModel[] }>(ROUTES.getResultsModelsRegistered),

  // The open project's own trait records, so a tab resolves which trait it works on from the
  // project instead of assuming one, and the Setup tab shows each revision for confirmation.
  traits: () => getJson<TraitsListing>(ROUTES.getResultsTraits),

  // Refuses with 409 when the hash is not the revision's own: the revision shown was not the one
  // on file.
  confirmTraitRevision: (body: ConfirmRevisionPayload) =>
    postJson<ServedTraitRevision & { audit_warning: string | null }>(
      ROUTES.postResultsTraitsConfirm,
      body,
    ),

  buildPlantMapping: (body: {
    name: string;
    images_root: string;
    plant_registry: string;
    dates?: string[];
    nn_tolerance_m?: number;
    supersede?: boolean;
    user: string;
  }) => postJson<ServedPlantMapping>(ROUTES.postResultsPlantMappingBuild, body),

  // Refuses with 404 when nothing is stored under the name.
  loadPlantMapping: (name: string) =>
    postJson<ServedPlantMapping>(ROUTES.postResultsPlantMappingLoad, { name }),

  // Every mapping name persisted under the open project, for the Results tab's name picker.
  listPlantMappings: () => getJson<{ names: string[] }>(ROUTES.getResultsPlantMappingList),

  phenologyMeasurement: (body: PhenologyPayload) =>
    postJson<PhenologyMeasurementResponse>(ROUTES.postResultsPhenologyMeasurement, body),

  /** Every delivery event the open project holds: what shipped, under which trait and kind. */
  deliveryEvents: () =>
    getJson<{ records: DeliveryEventRecord[] }>(ROUTES.getResultsDeliveryEvents),

  // The server computes what it exports, never a caller-composed table of rows. Its own request
  // shape is distinct from the measurement request's shape, never a spread of it.
  downloadCsv: async (body: ExportCsvPayload): Promise<Blob> =>
    (await postForBlob(ROUTES.postResultsExportCsv, body)).blob,

  // Unlike downloadCsv, this reports from the response headers: there is no prior screen
  // measurement for a count, so the headers travel back beside the blob rather than discarded.
  downloadCountCsv: async (
    body: ExportCountCsvPayload,
  ): Promise<{ blob: Blob; headers: ExportCountCsvHeaders }> => {
    const { blob, headers } = await postForBlob(ROUTES.postResultsExportCountCsv, body);
    return {
      blob,
      headers: {
        savedTo: headers.get("X-TCIP-Saved-To") ?? "",
        validated: headers.get("X-TCIP-Validated") === "true",
        acknowledgedBy: decodeURIComponent(headers.get("X-TCIP-Acknowledged-By") ?? ""),
      },
    };
  },
};
