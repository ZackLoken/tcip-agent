/** Inference + Results API helpers for the Inference and Results tabs. */

import { getJson, postForBlob, postJson, StructuredRefusalError, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import {
  ACKNOWLEDGED_BY_HEADER,
  type Assessments,
  type BucketExistsRefusal,
  type BuildMappingPayload,
  type CanopySegmentDisclosure,
  type ConfirmRevisionPayload,
  type DeliveryEvents,
  type ExportCountCsvPayload,
  type ExportCsvPayload,
  type InferenceFrame,
  type InferenceJob,
  type InferenceJobs,
  type LaunchInferencePayload,
  type MatchRadius,
  type PhenologyPayload,
  type PlantRegistryDisclosure,
  type RegisteredModels,
  SAVED_TO_HEADER,
  type ServedDeliveryEvent,
  type ServedPlantMappingDisclosure,
  type TraitRevision,
  VALIDATED_HEADER,
} from "@/api/types.generated";
import { createReconnectingSocket, jsonFrameHandlers } from "@/lib/reconnectingSocket";

export const inferenceApi = {
  launch: (body: LaunchInferencePayload) =>
    postJson<InferenceJob>(ROUTES.postInferenceLaunch, body),

  listJobs: () => getJson<InferenceJobs>(ROUTES.getInferenceJobs),

  cancel: (jobId: string, user: string) =>
    postJson<InferenceJob>(ROUTES.postInferenceJobsByJobIdCancel(jobId), { user }),
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

export interface PlantMappingDateSummary {
  n_images: number;
  n_mapped: number;
  n_unattributed: number;
  // null on a date with no recorded distance.
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
  // Each date's radius, absent for a date no capture of which carries a position.
  match_radius: Record<string, MatchRadius>;
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
  // null when a detection of this date lacks the state's attribute or an image is missing.
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
 *  user an empty string when the delivery is validated. */
export interface ExportCountCsvHeaders {
  savedTo: string;
  validated: boolean;
  acknowledgedBy: string;
}

// The measurement door's answer: both projections of one server-side measurement.
export interface PhenologyMeasurementResponse {
  curves: { rows: PerPlantRow[]; n_plants: number };
  milestones: { rows: OnsetRow[]; columns: MilestoneColumn[] };
  // Whether an assessment answers for every delivered bucket; when not, unvalidated_reason is the
  // gate's refusal and result_sha256 each projection's digest a breeder's acknowledgment of
  // exporting it binds to.
  validated: boolean;
  unvalidated_reason: string | null;
  result_sha256: Record<"curves" | "milestones", string> | null;
  trait: string;
  trait_revision: number;
  trait_revision_sha256: string;
  // False when no bucket's scope declares the positive state's attribute: the ratios are then not
  // a valid phenology measurement.
  positive_class_assessed: boolean;
  // What this delivery could not verify: a bare date omitted, absent, or archived (predictions
  // still counted), or "date/name" for one uncheckable capture.
  captures_unverified: string[];
  // A plant CSV the mapping was built from that moved since. Empty when nothing was unverified.
  plant_csvs_unverified: string[];
  // This delivery's own delivered dates, and its unattributed-capture count scoped to them.
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

/** The launch's bucket refusal a thrown error carries, read by kind off its parsed detail, or
 *  null for every other failure. */
export function bucketRefusalOf(e: unknown): BucketExistsRefusal | null {
  if (!(e instanceof StructuredRefusalError) || e.detail.kind !== "bucket_exists") return null;
  return e.detail as unknown as BucketExistsRefusal;
}

export type PlantMappingUnion = NonNullable<ServedDeliveryEvent["plant_mapping"]>;

/** Whether `pm` is the canopy-segment disclosure, narrowed first. */
export function isCanopySegmentDisclosure(pm: PlantMappingUnion): pm is CanopySegmentDisclosure {
  return "canopy_segments" in pm;
}

/** Whether `pm` is the whole-raster registry disclosure, narrowed after the canopy shape. */
export function isPlantRegistryDisclosure(pm: PlantMappingUnion): pm is PlantRegistryDisclosure {
  return !isCanopySegmentDisclosure(pm) && "plant_registry" in pm;
}

/** Whether `pm` is a walked mapping's served disclosure, narrowed last. */
export function isPlantMappingDisclosure(
  pm: PlantMappingUnion,
): pm is ServedPlantMappingDisclosure {
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
    postJson<ServedTraitRevision & { audit_warning: string | null }>(
      ROUTES.postResultsTraitsConfirm,
      body,
    ),

  buildPlantMapping: (body: BuildMappingPayload) =>
    postJson<ServedPlantMapping>(ROUTES.postResultsPlantMappingBuild, body),

  // Refuses with 404 when nothing is stored under the name.
  loadPlantMapping: (name: string) =>
    postJson<ServedPlantMapping>(ROUTES.postResultsPlantMappingLoad, { name }),

  // Every mapping name persisted under the open project.
  listPlantMappings: () => getJson<{ names: string[] }>(ROUTES.getResultsPlantMappingList),

  phenologyMeasurement: (body: PhenologyPayload) =>
    postJson<PhenologyMeasurementResponse>(ROUTES.postResultsPhenologyMeasurement, body),

  /** Every delivery event the open project holds: what shipped, under which trait and kind. */
  deliveryEvents: () => getJson<DeliveryEvents>(ROUTES.getResultsDeliveryEvents),

  /** The CSV the server computes for `body`'s phenology export. */
  downloadCsv: async (body: ExportCsvPayload): Promise<Blob> =>
    (await postForBlob(ROUTES.postResultsExportCsv, body)).blob,

  /** The count CSV the server computes for `body`, beside the response headers reporting it. */
  downloadCountCsv: async (
    body: ExportCountCsvPayload,
  ): Promise<{ blob: Blob; headers: ExportCountCsvHeaders }> => {
    const { blob, headers } = await postForBlob(ROUTES.postResultsExportCountCsv, body);
    return {
      blob,
      headers: {
        savedTo: sentHeader(headers, SAVED_TO_HEADER),
        validated: sentHeader(headers, VALIDATED_HEADER) === "true",
        acknowledgedBy: decodeURIComponent(sentHeader(headers, ACKNOWLEDGED_BY_HEADER)),
      },
    };
  },
};

/** The header `name` the count-export door sends on every answer; an answer without it refuses,
 *  naming it. */
function sentHeader(headers: Headers, name: string): string {
  const value = headers.get(name);
  if (value === null) throw new Error(`the count-export answer carries no ${name} header`);
  return value;
}
