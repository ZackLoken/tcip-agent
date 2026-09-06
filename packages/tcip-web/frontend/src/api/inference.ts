/** Inference + Results API helpers for the Inference and Results tabs. */

import { decodeRefusal, getJson, postJson, StructuredRefusalError, wsUrl } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type { JobStatus } from "@/api/types.generated";
import { createReconnectingSocket, jsonFrameHandlers } from "@/lib/reconnectingSocket";

/**
 * One entry from the project's trained-model registry, as the browser reads it.
 *
 * The backend's model_registry owns the entry and writes every key on it; the fields below are
 * the ones this UI uses, and tests/test_registry_entry_shape_agreement.py holds each of them
 * against an entry the real registry wrote, so a renamed or dropped key fails there instead of
 * arriving here as undefined.
 */
export interface RegisteredModel {
  name: string;
  checkpoint_path: string;
  tags?: string[];
  experiment_id?: string | null;
}

// "interrupted" (one of JobStatus's members) is a job rehydrated after a restart: its worker
// thread is gone and it is not resumable.
export type InferenceStatus = JobStatus;

export interface InferenceJob {
  job_id: string;
  status: InferenceStatus;
  done: number;
  total: number;
  images_dir: string;
  output_dir: string;
  error: string | null;
  warning: string | null;
}

/** A run is named, not spelled: the backend resolves both the images dir and the prediction
 *  bucket from (dataset_root, model_name, date) through its own layout resolver. Everything the
 *  platform derives per checkpoint (conf, IoU, tile geometry, cross-tile merge) is left off. */
export interface LaunchInferenceBody {
  checkpoint_path: string;
  dataset_root: string;
  model_name: string;
  date?: string | null;
  tile?: boolean;
}

export const inferenceApi = {
  launch: (body: LaunchInferenceBody) =>
    postJson<{
      status: string;
      job_id: string;
      images_dir: string;
      output_dir: string;
      bucket_redirected: boolean;
      requested_output_dir: string | null;
    }>(ROUTES.postInferenceLaunch, body),

  listJobs: () => getJson<{ jobs: InferenceJob[] }>(ROUTES.getInferenceJobs),

  cancel: (jobId: string) =>
    postJson<{ job_id: string; status: string; cancel_requested: boolean }>(
      ROUTES.postInferenceJobsByJobIdCancel(jobId),
      {},
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

// The persisted mapping record's own resolved match radius: never recomputed by a caller, and
// `source` names which of build_mapping's four branches produced `value`.
export interface PlantMappingTolerance {
  value: number;
  source: string;
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
  // null when this date was not fully classified/observed, never a fabricated ratio.
  ratio: number | null;
}

// Milestone column names are derived from the threaded trait's own spec, not hardcoded to
// any one trait: the fixed fields below are the columns every phenology delivery carries
// regardless of trait; the trait-specific milestone/date columns (e.g. <trait>_50per_date)
// arrive as additional keys and are read generically (see ResultsTab.tsx's milestoneColumns
// helper).
export interface OnsetRow {
  plant_id: string;
  accession: string | null;
  n_dates: number;
  n_dates_unclassified: number;
  n_dates_missing_images: number;
  // Dates with a real, non-zero-detection observation: a plant can be fully classified and fully
  // observed (0 unclassified, 0 missing) while still never having detected anything, e.g. before
  // emergence; that reads as "no observations", not "valid".
  n_observed_dates: number;
  [milestoneColumn: string]: string | number | null;
}

// The inputs a phenology measurement is computed from. Every Results door takes this same shape; none
// accepts rows, so no caller-composed table can be mistaken for (or declared to be) a delivery.
export interface PhenologyRequest {
  project_root: string;
  mapping_name: string;
  predictions_by_date: Record<string, string>;
  trait: string;
  // Show unvalidated numbers on screen instead of refusing. A display choice, never an
  // acknowledgement: it never applies to a CSV, which has its own export request shape.
  show_unvalidated?: boolean;
}

/** The breeder's own act of shipping a phenology export unvalidated: the reason only, since
 *  acknowledged_by is resolved server-side from the request's own user. */
export interface AcknowledgementRequest {
  reason: string;
}

/** The export door's own request shape: it shares the four fields PhenologyRequest also carries
 *  but does not inherit it (no show_unvalidated, a display-only choice this door never honors). */
export interface ExportCsvRequest {
  project_root: string;
  mapping_name: string;
  predictions_by_date: Record<string, string>;
  trait: string;
  payload: "curves" | "milestones";
  filename?: string;
  user?: string;
  acknowledgement?: AcknowledgementRequest | null;
}

/** The bucket regime of a per-image count delivery: an existing, reviewed prediction bucket. */
export interface PerImageCountDeliveryRequest {
  kind: "per_image_count";
  predictions_dir: string;
  trait: string;
}

/** A per-plant count delivery from a persisted whole-raster prediction bucket plus a
 *  registered plant registry. `nn_tolerance_m` is not exposed here: this door serves the
 *  derived tolerance, or the `canopy_subject` regime. */
export interface OrthomosaicPlantCountsDeliveryRequest {
  kind: "orthomosaic_plant_counts";
  predictions_dir: string;
  raster_path: string;
  plant_registry: string;
  delivered_phenotype: string;
  crop?: string;
  pipeline_version?: string;
  canopy_subject?: string;
}

/** The count-export door's own request shape: a discriminated `delivery` naming which of the
 *  two stranded count kinds this posts, the same acknowledgement shape `ExportCsvRequest` uses. */
export interface ExportCountCsvRequest {
  project_root: string;
  delivery: PerImageCountDeliveryRequest | OrthomosaicPlantCountsDeliveryRequest;
  filename: string;
  user?: string;
  acknowledgement?: AcknowledgementRequest | null;
}

/** The count-export door's own response headers: present on every response, an empty string
 *  (never a rendering of null/undefined) when nothing was unvalidated or acknowledged. */
export interface ExportCountCsvHeaders {
  savedTo: string;
  deliveryEventRecorded: boolean;
  unvalidatedDimensions: string;
  acknowledgedBy: string;
}

// One door returns both projections from one server-side measurement, so no surface can render
// either projection bare, and a milestone date and the curve it was read off cannot disagree.
export interface PhenologyMeasurementResponse {
  curves: { rows: PerPlantRow[]; n_plants: number; positive_class_id: number | null };
  milestones: { rows: OnsetRow[] };
  // Per-dimension state floored exactly as the delivered CSV's columns would be: a dimension with
  // no column of its own (tile_size) floors every dimension that does. Never stronger than the file.
  validated: Record<string, string>;
  // Each dimension's own unfloored state, e.g. { operating_point: "validated_held_out" }: for a
  // reader wanting one dimension's real outcome regardless of an unrelated dimension's failure.
  validated_raw: Record<string, string>;
  // True when any dimension lacked on-disk evidence, including one an acknowledgement cleared,
  // which is exactly when these numbers must not be rendered as validated.
  has_unvalidated_dimensions: boolean;
  validity_detail: Record<string, unknown>;
  // False when nothing was ever classified along the trait's positive-class axis: the ratios are
  // then not a valid phenology measurement (run + validate the classifier first).
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

/** One entry of what a trait delivers, in the crop vocabulary's own wording, never paraphrased. */
export interface DeliveredPhenotypeDefinition {
  name: string;
  definition: string;
}

/** A field a confirmation covered whose live value has moved since, with both values. */
export interface OperationalizationSupersession {
  field: string;
  confirmed_value: unknown;
  current_value: unknown;
}

/**
 * What a trait's delivered number means for one delivery kind, as the browser reads it.
 *
 * The agent states the record and the breeder confirms it; nothing here is authored in the GUI.
 * `confirmed_current` and `superseded` are computed by the backend from the same comparison the
 * delivery doors run, never re-derived here. `record_seen` is the content hash the confirmation
 * posts back, so a click lands on the text that was displayed.
 */
export type OperationalizationRecord = {
  trait: string;
  delivery_kind: string;
  statement: string;
  mechanism: string;
  measured_subject: string;
  delivered_phenotypes: string[];
  delivered_value_keys: string[];
  stated_by: string;
  stated_at: string;
  relayed_note: string;
  agent_client_name: string | null;
  agent_client_version: string | null;
  agent_session: string | null;
  terminal_session: string | null;
  harness_session: string | null;
  harness_effort_at_connect: string | null;
  confirmed_by: string | null;
  confirmed_at: string | null;
  identity_from_request: boolean | null;
  confirmed_current: boolean;
  superseded: OperationalizationSupersession[];
  registry_problem: string | null;
  delivers: DeliveredPhenotypeDefinition[];
  record_seen: string;
};

/** The four fields the confirmation writer owns, every one of them null after a withdrawal, plus
 *  the audit-append warning (A8): a confirmation that lands but whose audit line does not is still
 *  a 200, never a refusal, the same shape the trait-spec confirmation route now also carries. */
export interface OperationalizationConfirmation {
  confirmed_by: string | null;
  confirmed_at: string | null;
  identity_from_request: boolean | null;
  confirmed_fields: Record<string, unknown> | null;
  audit_warning: string | null;
}

export interface ConfirmOperationalizationBody {
  project_root: string;
  trait: string;
  delivery_kind: string;
  user?: string;
  confirmed?: boolean;
  record_seen: string;
}

/** A delivery door's refusal that a trait's delivered number has no confirmed meaning. */
export interface OperationalizationRefusal {
  kind: "operationalization";
  state: number | null;
  trait: string;
  delivery_kind: string;
  message: string;
  registry_problem: string | null;
}

/**
 * The operationalization refusal a thrown error carries, or null for every other failure.
 *
 * Read by kind off the parsed detail, so this family can never be matched by the prose regex the
 * unvalidated-evidence refusal still dispatches on.
 */
export function operationalizationRefusalOf(e: unknown): OperationalizationRefusal | null {
  if (!(e instanceof StructuredRefusalError)) return null;
  const detail = e.detail;
  if (detail.kind !== "operationalization") return null;
  return {
    kind: "operationalization",
    state: typeof detail.state === "number" ? detail.state : null,
    trait: String(detail.trait ?? ""),
    delivery_kind: String(detail.delivery_kind ?? ""),
    message: typeof detail.message === "string" ? detail.message : e.message,
    registry_problem: typeof detail.registry_problem === "string" ? detail.registry_problem : null,
  };
}

/** A count door's own delivery-gate refusal: the gate's reason plus this door's own remedy,
 *  beside every counts-bearing fact the core had in hand (image_count, n_detections, and the
 *  like, door-dependent, so this carries the raw detail rather than a fixed field set). */
export interface DeliveryGateRefusal {
  kind: "delivery_gate";
  message: string;
  unvalidated_dimensions: string;
  [fact: string]: unknown;
}

/**
 * The delivery-gate refusal a thrown error carries, or null for every other failure.
 *
 * Read by kind off the parsed detail, the same dispatch `operationalizationRefusalOf` uses for
 * its own family, so a delivery-gate refusal is never matched by a prose regex either.
 */
export function deliveryGateRefusalOf(e: unknown): DeliveryGateRefusal | null {
  if (!(e instanceof StructuredRefusalError)) return null;
  const detail = e.detail;
  if (detail.kind !== "delivery_gate") return null;
  return {
    ...detail,
    kind: "delivery_gate",
    message: typeof detail.message === "string" ? detail.message : e.message,
    unvalidated_dimensions:
      typeof detail.unvalidated_dimensions === "string" ? detail.unvalidated_dimensions : "",
  };
}

/** The Inference tab's launch refusing a bucket a prior run already published into: the
 *  requested path, the document count and a fresh bucket name, or null when every variant to
 *  the resolver's ceiling is taken (the agent's own remedy is the only way forward then). */
export interface BucketHoldsDocumentsRefusal {
  kind: "bucket_holds_documents";
  message: string;
  date: string | null;
  requested_model_name: string | null;
  requested_output_dir: string | null;
  document_stem_count: number | null;
  suggested_model_name: string | null;
  suggested_output_dir: string | null;
}

/** The launch's other refusal: a job of this process already writing the same (dataset, model,
 *  date), named so the breeder watches it instead of resolving past it into a second job. */
export interface BucketInFlightRefusal {
  kind: "bucket_in_flight";
  message: string;
  date: string | null;
  requested_output_dir: string | null;
  job_id: string | null;
}

export type BucketRefusal = BucketHoldsDocumentsRefusal | BucketInFlightRefusal;

/**
 * Either of the launch's own bucket refusals a thrown error carries, or null for every other
 * failure (including the verdict-exhaustion 409, whose detail is a plain string).
 *
 * Read by kind off the parsed detail, the same dispatch `deliveryGateRefusalOf` uses for its own
 * family, so a bucket refusal is never matched by a prose regex either. The route sends every
 * field below on both refusal kinds; null here is the honest read of one that, despite that,
 * a given detail does not carry.
 */
export function bucketRefusalOf(e: unknown): BucketRefusal | null {
  if (!(e instanceof StructuredRefusalError)) return null;
  const detail = e.detail;
  const date = typeof detail.date === "string" ? detail.date : null;
  const message = typeof detail.message === "string" ? detail.message : e.message;
  if (detail.kind === "bucket_holds_documents") {
    return {
      kind: "bucket_holds_documents",
      message,
      date,
      requested_model_name:
        typeof detail.requested_model_name === "string" ? detail.requested_model_name : null,
      requested_output_dir:
        typeof detail.requested_output_dir === "string" ? detail.requested_output_dir : null,
      document_stem_count:
        typeof detail.document_stem_count === "number" ? detail.document_stem_count : null,
      suggested_model_name:
        typeof detail.suggested_model_name === "string" ? detail.suggested_model_name : null,
      suggested_output_dir:
        typeof detail.suggested_output_dir === "string" ? detail.suggested_output_dir : null,
    };
  }
  if (detail.kind === "bucket_in_flight") {
    return {
      kind: "bucket_in_flight",
      message,
      date,
      requested_output_dir:
        typeof detail.requested_output_dir === "string" ? detail.requested_output_dir : null,
      job_id: typeof detail.job_id === "string" ? detail.job_id : null,
    };
  }
  return null;
}

/** One trait's authoring statement, as the browser reads it. */
export type TraitSpecStatementRecord = {
  trait: string;
  statement_fields: Record<string, unknown> | null;
  rationale: string | null;
  stated_by: string | null;
  stated_at: string | null;
  relayed_note: string | null;
  agent_client_name: string | null;
  agent_client_version: string | null;
  agent_session: string | null;
  terminal_session: string | null;
  harness_session: string | null;
  harness_effort_at_connect: string | null;
  confirmed_by: string | null;
  confirmed_at: string | null;
  identity_from_request: boolean | null;
  confirmed_current: boolean;
  record_seen: string;
};

/** The four fields the trait-spec confirmation writer owns, plus the audit-append warning (A8):
 *  a confirmation that lands but whose audit line does not is still a 200, never a refusal. */
export interface TraitSpecStatementConfirmation {
  confirmed_by: string | null;
  confirmed_at: string | null;
  identity_from_request: boolean | null;
  record_seen: string | null;
  audit_warning: string | null;
}

export interface ConfirmTraitSpecStatementBody {
  project_root: string;
  trait: string;
  record_seen: string;
  user?: string;
  confirmed?: boolean;
}

/** A trait-spec confirmation's refusal that the statement moved since it was displayed. */
export interface TraitSpecAuthoringRefusal {
  kind: "trait_spec_authoring";
  message: string;
  record: TraitSpecStatementRecord;
}

export function traitSpecAuthoringRefusalOf(e: unknown): TraitSpecAuthoringRefusal | null {
  if (!(e instanceof StructuredRefusalError)) return null;
  const detail = e.detail;
  if (detail.kind !== "trait_spec_authoring") return null;
  return {
    kind: "trait_spec_authoring",
    message: typeof detail.message === "string" ? detail.message : e.message,
    record: detail.record as TraitSpecStatementRecord,
  };
}

// The mapping a phenology delivery attributed detections through, plus this delivery's own
// unattributed-capture disclosure (MappingBuild.delivery_disclosure).
export interface PlantMappingDisclosure {
  name: string;
  project_root: string;
  dataset_id: string;
  dataset_root: string;
  built_at: string;
  record_sha256: string;
  nn_tolerance_m: PlantMappingTolerance;
  capture_identity: Record<string, string>;
  captures_unverified: string[];
  plant_csvs_unverified: string[];
  dates_delivered: string[];
  images_unattributed: number;
  images_unattributed_scope: string;
  plant_attribution: string;
}

// The whole-raster counterpart of PlantMappingDisclosure: a nearest-neighbour orthomosaic
// delivery's own registry disclosure (deliver_orthomosaic_plant_counts's own).
export interface PlantRegistryDisclosure {
  plant_registry: { name: string; digest: string };
  project_root: string;
  raster_identity: Record<string, unknown>;
  nn_tolerance_m: PlantMappingTolerance;
  detections_unattributed: number;
  detections_unattributed_scope: string;
  plant_attribution: string;
  plants_outside_raster: string[];
}

// One resolved segment-to-plant tie: the derived clearance a displaced registry position
// would have to exceed to leave this segment.
export interface SegmentTieDisclosure {
  segment_index: number;
  plot_name: string;
  clearance_m: number;
}

// The label document a canopy-segment delivery read its boundaries from.
export interface CanopySegmentsDocument {
  path: string;
  sha256: string;
  subject: string;
  n_segments: number;
}

// A canopy-segment delivery's unattributed-detection count, broken out by source; the sum of
// these three is CanopySegmentDisclosure.detections_unattributed, stated there as derived.
export interface UnattributedDetectionsBySource {
  outside_segments: number;
  overlapping_segments: number;
  segment_without_plant: number;
}

// The third disclosure shape: a canopy-segment orthomosaic delivery's own disclosure
// (deliver_orthomosaic_plant_counts's canopy_subject argument).
export interface CanopySegmentDisclosure {
  plant_registry: { name: string; digest: string };
  project_root: string;
  raster_identity: Record<string, unknown>;
  canopy_segments: CanopySegmentsDocument;
  segment_ties: SegmentTieDisclosure[];
  segments_without_plant: number;
  plants_outside_raster: string[];
  plants_without_segment: string[];
  plants_with_ambiguous_detections: string[];
  detections_unattributed: number;
  detections_unattributed_by_source: UnattributedDetectionsBySource;
  detections_unattributed_scope: string;
  plant_attribution: string;
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

/** The `delivery_supersessions` record `supersede_delivery` filed against one event's id, joined
 *  onto that event by the backend (`delivery_events_schema.with_supersessions`). */
export interface DeliverySupersession {
  superseded_event_id: string;
  output_sha256: string | null;
  replacement_event_id: string | null;
  reason: string;
  superseded_by: string;
  superseded_at: string;
}

/** One completed delivery: what shipped, under which trait and kind, and the real per-bucket
 *  verification evidence the delivering door reconciled at the time. Read-only; a delivery event
 *  is a fact recorded after an artifact already shipped, not a statement to confirm. */
export interface DeliveryEventRecord {
  event_id: string;
  trait: string | null;
  delivery_kind: string | null;
  door: string;
  output_path: string | null;
  output_sha256: string | null;
  documents: Record<string, unknown>;
  produced_at: string;
  // Who acknowledged this delivery unvalidated, and why; null on both when nothing was.
  acknowledged_by: string | null;
  acknowledgement_reason: string | null;
  // The plant mapping this delivery attributed detections through, door-conditional (phenology
  // a mapping, deliver_orthomosaic_plant_counts a registry or canopy disclosure, others null).
  plant_mapping: PlantMappingUnion | null;
  // Set only alongside plant_mapping: the name to load to see exactly the cited record (its own
  // name while unmoved, an archived key once superseded, or null when neither resolves).
  plant_mapping_resolved_key?: string | null;
  // The supersession filed against this event, if any (see DeliverySupersession above).
  superseded: DeliverySupersession | null;
}

export const resultsApi = {
  registeredModels: (project_path: string) =>
    getJson<{ models: RegisteredModel[] }>(
      `${ROUTES.getResultsModelsRegistered}?project_path=${encodeURIComponent(project_path)}`,
    ),

  // The project's own registered traits, so the Results tab resolves which trait it is
  // computing for from the project's registry instead of assuming one. Each trait's declared
  // milestone fractions come along, so the tab can tell what there is to compute for it.
  traits: (project_root: string) =>
    getJson<{
      traits: string[];
      milestone_fractions_by_trait: Record<string, number[]>;
      invalid_specs: { file: string; reason: string }[];
    }>(`${ROUTES.getResultsTraits}?project_root=${encodeURIComponent(project_root)}`),

  buildPlantMapping: (body: {
    name: string;
    images_root: string;
    plant_registry: string;
    dates?: string[];
    nn_tolerance_m?: number;
    supersede?: boolean;
  }) =>
    postJson<{
      summary: PlantMappingSummary;
      mapping: unknown;
      unreadable: Record<string, string[]>;
      nn_tolerance_m: PlantMappingTolerance;
      max_match_distance_m: number;
    }>(ROUTES.postResultsPlantMappingBuild, body),

  loadPlantMapping: (name: string) =>
    postJson<{
      mapping: unknown;
      summary: PlantMappingSummary | Record<string, never>;
      nn_tolerance_m: PlantMappingTolerance | null;
      max_match_distance_m: number | null;
    }>(ROUTES.postResultsPlantMappingLoad, { name }),

  // Every mapping name persisted under the open project, for the Results tab's name picker.
  listPlantMappings: () => getJson<{ names: string[] }>(ROUTES.getResultsPlantMappingList),

  phenologyMeasurement: (body: PhenologyRequest) =>
    postJson<PhenologyMeasurementResponse>(ROUTES.postResultsPhenologyMeasurement, body),

  /** Enumerates what exists, keyed by trait plus delivery kind; the served `statement_fields`
   *  names what the `record_seen` hash covers, so the browser holds no list of its own to drift. */
  operationalizations: (project_root: string) =>
    getJson<{ records: OperationalizationRecord[]; statement_fields: string[] }>(
      `${ROUTES.getResultsOperationalizations}?project_root=${encodeURIComponent(project_root)}`,
    ),

  operationalization: (project_root: string, trait: string, delivery_kind: string) =>
    getJson<OperationalizationRecord>(
      `${ROUTES.getResultsOperationalization}?project_root=${encodeURIComponent(project_root)}` +
        `&trait=${encodeURIComponent(trait)}&delivery_kind=${encodeURIComponent(delivery_kind)}`,
    ),

  // Refuses with 409 when the record moved since it was displayed, carrying what is on file now.
  confirmOperationalization: (body: ConfirmOperationalizationBody) =>
    postJson<OperationalizationConfirmation>(ROUTES.postResultsOperationalizationConfirm, body),

  /** Every trait-spec authoring statement this project holds, one row per trait; siblings of
   *  `operationalizations`/`operationalization` above, the same generalized statement shape. */
  traitSpecStatements: (project_root: string) =>
    getJson<{
      records: TraitSpecStatementRecord[];
      unresolved: unknown[];
      statement_fields: string[];
    }>(`${ROUTES.getResultsTraitSpecStatements}?project_root=${encodeURIComponent(project_root)}`),

  traitSpecStatement: (project_root: string, trait: string) =>
    getJson<TraitSpecStatementRecord>(
      `${ROUTES.getResultsTraitSpecStatement}?project_root=${encodeURIComponent(project_root)}` +
        `&trait=${encodeURIComponent(trait)}`,
    ),

  // Refuses with 409 when the statement moved since it was displayed, carrying what is on file now.
  confirmTraitSpecStatement: (body: ConfirmTraitSpecStatementBody) =>
    postJson<TraitSpecStatementConfirmation>(ROUTES.postResultsTraitSpecStatementConfirm, body),

  /** Every delivery event this project holds: what shipped, under which trait and kind. */
  deliveryEvents: (project_root: string) =>
    getJson<{ records: DeliveryEventRecord[] }>(
      `${ROUTES.getResultsDeliveryEvents}?project_root=${encodeURIComponent(project_root)}`,
    ),

  // The server computes what it exports, never a caller-composed table of rows. Its own request
  // shape, never a spread of the measurement request, which no longer shares one with it.
  downloadCsv: async (body: ExportCsvRequest): Promise<Blob> => {
    const resp = await fetch(ROUTES.postResultsExportCsv, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      // The same decoder every JSON call uses, so a structured refusal arrives parsed, not stringified.
      throw await decodeRefusal(resp, `export_csv failed: ${resp.status}`);
    }
    return await resp.blob();
  },

  // Unlike downloadCsv, this reports from the response headers: there is no prior screen
  // measurement for a count, so the headers travel back beside the blob rather than discarded.
  downloadCountCsv: async (
    body: ExportCountCsvRequest,
  ): Promise<{ blob: Blob; headers: ExportCountCsvHeaders }> => {
    const resp = await fetch(ROUTES.postResultsExportCountCsv, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      throw await decodeRefusal(resp, `export_count_csv failed: ${resp.status}`);
    }
    const blob = await resp.blob();
    return {
      blob,
      headers: {
        savedTo: resp.headers.get("X-TCIP-Saved-To") ?? "",
        deliveryEventRecorded: resp.headers.get("X-TCIP-Delivery-Event-Recorded") === "true",
        unvalidatedDimensions: resp.headers.get("X-TCIP-Unvalidated-Dimensions") ?? "",
        acknowledgedBy: decodeURIComponent(resp.headers.get("X-TCIP-Acknowledged-By") ?? ""),
      },
    };
  },
};
