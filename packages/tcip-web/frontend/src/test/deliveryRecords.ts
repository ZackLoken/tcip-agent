import type {
  Acknowledgment,
  Assessments,
  AuditGapRefusal,
  CanopySegmentDisclosure,
  DeliveryEventRecord,
  DeliveryEvents,
  DeliveryRefusal,
  MappingNames,
  OperationalizationRefusal,
  PhenologyAnswer,
  PlantMappingDisclosure,
  PlantRegistryDisclosure,
  RegisteredModels,
  SavedExport,
  ServedDeliveryEvent,
  ServedMapping,
} from "@/api/types.generated";
import produced from "@/test/deliveryEvents.json";
import type { Refusal } from "@/test/refusals";

/** A served delivery event whose stored event's disclosure is `D`, its event `E` beside that. */
type Disclosed<D, E = unknown> = Omit<ServedDeliveryEvent, "event"> & {
  event: Omit<DeliveryEventRecord, "plant_mapping"> & { plant_mapping: D } & E;
};

/** Each scene's one served delivery event, by fixture name. */
interface Records {
  validated: Disclosed<null>;
  acknowledged_mapping: Disclosed<PlantMappingDisclosure, { acknowledgment: Acknowledgment }>;
  archived_mapping: Disclosed<PlantMappingDisclosure>;
  registry: Disclosed<PlantRegistryDisclosure>;
  canopy: Disclosed<CanopySegmentDisclosure>;
}

/**
 * The delivery-events listings `tools/generate_delivery_fixture.py` served from deliveries the
 * platform's own producers made, each holding its one event: `validated` a per-image count
 * delivery its assessment validates, `acknowledged_mapping` a phenology delivery through a plant
 * mapping shipped under a breeder's acknowledgment, `archived_mapping` that same record once a
 * superseding rebuild archived the mapping it cites, and `registry` and `canopy` the orthomosaic
 * door's nearest-plant and canopy-segment deliveries.
 */
export const DELIVERY_LISTINGS = produced.listings as unknown as {
  [N in keyof Records]: { records: [Records[N]] };
};

/** Each listing's one served event, by fixture name. */
export const DELIVERY_RECORDS = Object.fromEntries(
  Object.entries(DELIVERY_LISTINGS).map(([name, listing]) => [name, listing.records[0]]),
) as unknown as Records;

/** What an export door answered: the delivery headers it sent and the file, beside the project's
 *  delivery events once it had, its own the latest. */
interface Export {
  headers: Record<string, string>;
  file: string;
  events: DeliveryEvents;
}

/** The count export of the `validated` project's bucket. */
export const COUNT_EXPORT = produced.count_export as unknown as Export;

/** The count export of the nearest-plant project's counts, unvalidated under a breeder's
 *  acknowledgment. */
export const ACKNOWLEDGED_COUNT_EXPORT = produced.acknowledged_count_export as unknown as Export;

/** The phenology project's curves export, shipped under an acknowledgment. */
export const CURVES_EXPORT = produced.curves_export as unknown as Export;

/** The delivery event `exported` wrote: the latest of the events it was answered beside. */
export function eventOf(exported: Export): DeliveryEventRecord {
  return exported.events.records.at(-1)!.event;
}

/** The `validated` project's served assessments of its trained checkpoint: the one its delivery
 *  rests on, passed, then one at a conf no detection reaches, failed. */
export const PRODUCED_ASSESSMENTS = produced.assessments as unknown as Assessments;

/** The served assessments of a project whose trained checkpoint was assessed once, at a conf no
 *  detection reaches, and failed. */
export const FAILED_ASSESSMENTS = produced.failed_assessments as unknown as Assessments;

/** The `validated` project's registered models: its trained checkpoint, the one its assessments
 *  judged, then a second run's and a checkpoint registered as `other`, neither assessed. */
export const REGISTERED_MODELS = produced.registered_models as unknown as RegisteredModels;

/** The results routes' answers the generator received, each refusal with its status: the count
 *  project's mapping names, none; the count export's and the curves export's refusals once the
 *  delivery's line failed, the count project's listing refused once a delivery record was
 *  damaged, the phenology project's measurement refused by the delivery gate, shown unvalidated
 *  and refused for naming no stored mapping, the refusal of a mapping built while its line failed, of a rebuild of a
 *  mapping a delivery cites, of a measurement of a trait confirmed with no phenology
 *  operationalization, and of a per-plant count export over an unassessed bucket; the phenology
 *  project's mapping names, its mapping loaded and its mappings built (placing every plant,
 *  placing one, at a stated radius, and over a date whose photographs carry no position); the
 *  measurements of three plants, each alone, whose milestone rows read incomplete, unobserved and
 *  right-censored, in that order; and an assessed series' project's deliveries,
 *  none, and its first plant measured validated, the series' reference photographs attributed
 *  to no plant. */
// A JSON import reads each literal as a string, which the literal unions admit only through
// `unknown`.
export const RESULTS_ANSWERS = produced.answers as unknown as {
  no_mapping_names: MappingNames;
  count_export_audit_gap: Refusal<AuditGapRefusal & { committed: SavedExport }>;
  unreadable_deliveries: Refusal<string>;
  unknown_mapping: Refusal<string>;
  export_audit_gap: Refusal<AuditGapRefusal & { committed: SavedExport }>;
  delivery_refusal: Refusal<DeliveryRefusal>;
  phenology: PhenologyAnswer;
  mapping_audit_gap: Refusal<AuditGapRefusal & { committed: null }>;
  mapping_cited: Refusal<string>;
  operationalization_refusal: Refusal<OperationalizationRefusal>;
  count_delivery_refusal: Refusal<DeliveryRefusal>;
  mapping_names: MappingNames;
  mapping_loaded: ServedMapping;
  mapping_built: ServedMapping;
  mapping_one_placed: ServedMapping;
  mapping_stated: ServedMapping;
  mapping_unpositioned: ServedMapping;
  milestones: [PhenologyAnswer, PhenologyAnswer, PhenologyAnswer];
  no_deliveries: DeliveryEvents;
  phenology_validated: PhenologyAnswer;
};
