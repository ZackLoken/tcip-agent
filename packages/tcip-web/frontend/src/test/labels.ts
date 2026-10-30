import type { subjectsApi } from "@/api/subjects";
import type {
  AuditGapRefusal,
  FlagRequest,
  LabelDocumentAnswer,
  OperatingPoint,
  RegistrySave,
  RegistrySubjects,
  Resolution,
  SavedLabels,
  ServedAnnotation,
  ServedFlag,
  ServedProposal,
  ServedProposals,
  VersionConflictRefusal,
} from "@/api/types.generated";
import produced from "@/test/labelAnswers.json";
import type { Refusal } from "@/test/refusals";

/** What `tools/generate_labels_fixture.py` received from the label routes over one image: the
 *  save of a box, a polygon, a point and a geometry-less rating with a flag on the box, the load
 *  of that document, the proposals route over one box staged on it, the save resolving the flag,
 *  and the refusal of a save naming a stale version token; then from the subjects routes, the
 *  registry the load route served once its subject was declared bare, a save growing it, the
 *  refusal of a second one whose line failed, and the registry the load route then served. */
// A JSON import reads each tuple as an array, which no tuple type admits without `unknown`.
export const LABEL_ANSWERS = produced as unknown as {
  saved: SavedLabels;
  loaded: LabelDocumentAnswer;
  proposals: ServedProposals;
  resolved: SavedLabels;
  version_conflict: Refusal<VersionConflictRefusal>;
  registry_bare: Awaited<ReturnType<typeof subjectsApi.load>>;
  registry_saved: RegistrySave;
  registry_stale: Refusal<string>;
  registry_audit_gap: Refusal<AuditGapRefusal & { committed: RegistrySave }>;
  registry_loaded: Awaited<ReturnType<typeof subjectsApi.load>>;
};

const [BOX, POLYGON, POINT, RATING] = LABEL_ANSWERS.loaded.annotations;
const [PROPOSAL] = LABEL_ANSWERS.proposals.proposals;
const [FLAG] = LABEL_ANSWERS.loaded.flags;
const [SUBJECT] = Object.values(LABEL_ANSWERS.registry_loaded.subjects ?? {});

/** The registry the load route served once its subject was declared and before any attribute
 *  was: that subject, by its name. */
export const BARE_REGISTRY: RegistrySubjects = LABEL_ANSWERS.registry_bare.subjects ?? {};

/** The attributes the registry the load route served declares of its subject, by name. */
export const SERVED_ATTRIBUTES = SUBJECT.attributes ?? {};

/** The produced flag with `change` laid over it, raised on the image as a whole and served on no
 *  annotation, with `request` laid over its request. */
export function servedFlag(
  request: Partial<FlagRequest> = {},
  change: Partial<ServedFlag> = {},
): ServedFlag {
  return {
    ...FLAG,
    covered_by: [],
    ...change,
    request: { ...FLAG.request, target: null, ...request },
  };
}

/** The produced flag's resolution with `change` laid over it. */
export function servedResolution(change: Partial<Resolution> = {}): Resolution {
  const [{ resolution }] = LABEL_ANSWERS.resolved.document.flags;
  if (resolution === null) throw new Error("the produced resolving save left its flag open");
  return { ...resolution, ...change };
}

/** The produced annotation whose geometry is the kind `change` states: rings, a box, a point, or
 *  none. */
function producedOfKind(change: Partial<ServedAnnotation>): ServedAnnotation {
  if (change.rings) return POLYGON;
  if (change.bbox) return BOX;
  return change.point ? POINT : RATING;
}

/** The produced annotation of `change`'s geometry kind with `change` laid over it. */
export function servedAnnotation(
  change: Partial<ServedAnnotation> & Pick<ServedAnnotation, "subject">,
): ServedAnnotation {
  return { ...producedOfKind(change), ...change };
}

/** The produced proposals answer holding `proposals`, its operating point with `point` laid over
 *  it. */
export function servedProposals(
  proposals: ServedProposal[],
  point: Partial<OperatingPoint> = {},
): ServedProposals {
  return { operating_point: { ...LABEL_ANSWERS.proposals.operating_point, ...point }, proposals };
}

/** The produced proposal, its geometry the produced annotation's of `change`'s kind, with
 *  `change` laid over it. */
export function servedProposal(
  change: Partial<ServedProposal> & Pick<ServedProposal, "subject">,
): ServedProposal {
  const { bbox, rings, point } = producedOfKind(change);
  return { ...PROPOSAL, bbox, rings, point, ...change };
}
