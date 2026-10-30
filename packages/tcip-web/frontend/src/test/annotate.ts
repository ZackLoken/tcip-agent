import { type LoadedLabels, loadedLabels, type LoadedSave, loadedSave } from "@/api/client";
import type {
  FindAnswer,
  ImageStagings,
  ImageStates,
  LabelDocumentAnswer,
  RegistrySubjects,
  SavedLabels,
  ServedAnnotation,
  ServedProposals,
  Staged,
  SubjectsLoad,
} from "@/api/types.generated";
import produced from "@/test/annotateAnswers.json";
import type { Refusal } from "@/test/refusals";

/** What `tools/generate_annotate_fixture.py` received from the annotate routes over the open
 *  fixture's photographs, each scene on the first after the loads:
 *  - each photograph's document holding nothing, loaded (`loads`, in the selection's order);
 *  - saves of one `subject_a` box (`box`), of nothing (`cleared`) and of one `tip` point
 *    (`point`); every photograph's state of `subject_a` once the second holds a box (`states`);
 *    the document marked complete over its box (`complete`), and its save marked complete holding
 *    nothing (`negative`); every photograph's state again once the fifth is marked complete over a
 *    box (`marked_states`);
 *  - one proposal staged over the third photograph's document holding nothing, served
 *    (`single_proposals`);
 *  - the review: four proposals staged, the first accepted, and two boxes of a person's and a
 *    polygon added (`review_saved`), that document loaded (`review`) and its proposals served
 *    (`review_proposals`); the first box flagged (`review_flagged`); then the proposal a box pairs
 *    with accepted (`accept_paired`), the unpaired proposal accepted (`accept_unpaired`), and a
 *    second bucket's box and polygon proposals served (`shaped_proposals`) and its polygon
 *    accepted (`accept_polygon`); the unpaired proposal accepted again by a save drawing a box
 *    over it first (`accept_drawn`);
 *  - a document whose third box a tool wrote (`tool_two`), its save confirming that box
 *    (`confirmed`), and a save of the three boxes as a person's (`person_boxes`);
 *  - the second photograph's document holding a tool's, a person's, an accepted tool's and an
 *    unattributed box (`authored`), then one holding a tool's polygon (`tool_polygon`);
 *  - the save flagging one box (`flag_raised`) and its document (`flagged`), the save removing the
 *    box (`flag_removed`), and a new box flagged twice with the second flag resolved
 *    (`flags_mixed`);
 *  - over the fourth photograph, its stagings before any propose (`no_stagings`), then through a
 *    stub engine a propose staging two candidates (`staged`), the photograph's stagings
 *    (`stagings`) and the staged bucket's proposals (`staged_proposals`), finds answering two boxes
 *    (`found`), one other (`found_other`) and none (`found_none`), a propose answering nothing
 *    (`nothing_proposed`), and the refusal of one whose engine will not load (`engine_refusal`);
 *  - the photographs' registry declaring the box subject, the point subject and a second subject
 *    (`registry`);
 *  - the refused load of the fifth photograph, its stored document damaged (`unreadable`);
 *  - the name each staging of the single, the review and the polygon proposals published its
 *    bucket under (`buckets`). */
// A JSON import reads each literal as a string, which the literal unions admit only through
// `unknown`.
export const ANNOTATE_ANSWERS = produced as unknown as {
  loads: LabelDocumentAnswer[];
  saves: Record<keyof typeof produced.saves, SavedLabels>;
  states: ImageStates;
  marked_states: ImageStates;
  complete: LabelDocumentAnswer;
  single_proposals: ServedProposals;
  review: LabelDocumentAnswer;
  review_proposals: ServedProposals;
  review_flagged: LabelDocumentAnswer;
  shaped_proposals: ServedProposals;
  tool_two: LabelDocumentAnswer;
  authored: LabelDocumentAnswer;
  tool_polygon: LabelDocumentAnswer;
  flagged: LabelDocumentAnswer;
  flags_mixed: LabelDocumentAnswer;
  no_stagings: ImageStagings;
  staged: Staged;
  stagings: ImageStagings;
  staged_proposals: ServedProposals;
  found: FindAnswer;
  found_other: FindAnswer;
  found_none: FindAnswer;
  nothing_proposed: null;
  engine_refusal: Refusal<string>;
  registry: SubjectsLoad;
  unreadable: Refusal<string>;
  buckets: { single: string; review: string; shaped: string };
};

/** The subjects the produced registry declares, by name. */
const DECLARED = ANNOTATE_ANSWERS.registry.subjects ?? {};

/** The produced registry's subjects `names` name, each as served; a name it declares no subject
 *  of throws. */
export function registryOf(...names: string[]): RegistrySubjects {
  return Object.fromEntries(
    names.map((name) => {
      const subject = DECLARED[name] as RegistrySubjects[string] | undefined;
      if (subject === undefined) throw new Error(`the produced registry declares no ${name}`);
      return [name, subject];
    }),
  );
}

/** The subject of the produced point save. */
export const POINT_SUBJECT = ANNOTATE_ANSWERS.saves.point.document.annotations[0].subject;

/** The produced registry's subject other than the box and point subjects. */
export const [SECOND_SUBJECT] = Object.keys(DECLARED).filter(
  (name) =>
    name !== POINT_SUBJECT && name !== ANNOTATE_ANSWERS.saves.box.document.annotations[0].subject,
);

/** The produced load answer of the photograph at `imagePath`; a path the generator loaded nothing
 *  of throws. */
function loadAnswerOf(imagePath: string): LabelDocumentAnswer {
  const found = ANNOTATE_ANSWERS.loads.find((answer) => answer.image_path === imagePath);
  if (!found) throw new Error(`the generator loaded no document of ${imagePath}`);
  return found;
}

/** The produced load of the photograph at `imagePath`, as the client loads it. */
export function loadOf(imagePath: string): LoadedLabels {
  return loadedLabels(loadAnswerOf(imagePath));
}

/** The produced load of the photograph at `imagePath` holding `annotations`, each placed by the
 *  labels fixture's own annotation of its kind (`test/labels.ts`'s `servedAnnotation`), as the
 *  client loads it. */
export function loadHolding(imagePath: string, annotations: ServedAnnotation[]): LoadedLabels {
  return loadedLabels({ ...loadAnswerOf(imagePath), annotations });
}

/** The produced save answers, as the client adopts each. */
export const LANDED = Object.fromEntries(
  Object.entries(ANNOTATE_ANSWERS.saves).map(([name, saved]) => [name, loadedSave(saved)]),
) as Record<keyof typeof ANNOTATE_ANSWERS.saves, LoadedSave>;

/** The produced loaded documents, as the client loads each. */
export const LOADED = {
  review: loadedLabels(ANNOTATE_ANSWERS.review),
  review_flagged: loadedLabels(ANNOTATE_ANSWERS.review_flagged),
  tool_two: loadedLabels(ANNOTATE_ANSWERS.tool_two),
  authored: loadedLabels(ANNOTATE_ANSWERS.authored),
  tool_polygon: loadedLabels(ANNOTATE_ANSWERS.tool_polygon),
  flagged: loadedLabels(ANNOTATE_ANSWERS.flagged),
  flags_mixed: loadedLabels(ANNOTATE_ANSWERS.flags_mixed),
};
