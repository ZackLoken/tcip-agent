import type { api } from "@/api/client";
import type {
  OpenAnswer,
  SelectAnswer,
  SnapshotProject,
  StateSnapshotMessage,
} from "@/api/types.generated";
import { useStore } from "@/store";
import type { DatasetSelection, GuiState } from "@/store/types";
import produced from "@/test/openFixtures.json";

type Tree = Awaited<ReturnType<typeof api.dataset.tree>>;

/** What `tools/generate_open_fixture.py` received from a backend opening a project: the open
 *  route's answer, the envelope `/ws/state` sent next, the select route's answers for a capture
 *  of five photographs, a capture of one band group and a capture whose label document will not
 *  read, the nav route's answer, three datasets' trees, the select route's answers for the
 *  photographs made for a subject, for the subject an agent's focus names and for that subject
 *  with a bucket, and the photographs' selection under a separator-ended root. */
const PRODUCED = produced as unknown as {
  opened: OpenAnswer;
  envelope: StateSnapshotMessage;
  raw: SelectAnswer;
  grouped: SelectAnswer;
  troubled: SelectAnswer;
  navigated: Awaited<ReturnType<typeof api.dataset.nav>>;
  ds_tree: Tree;
  troubled_tree: Tree;
  empty_tree: Tree;
  raw_subject: SelectAnswer;
  focused: SelectAnswer;
  focused_bucket: SelectAnswer;
  separated: DatasetSelection;
};

/** The dataset trees the generator received: the photographs' dataset (`ds`), the troubled
 *  capture's (`troubled`, its `label_problem` naming the document) and a dataset with no capture
 *  (`empty`). */
export const DATASET_TREES = {
  ds: PRODUCED.ds_tree,
  troubled: PRODUCED.troubled_tree,
  empty: PRODUCED.empty_tree,
};

/** The photographs' dataset's two capture dates, the first with buckets published over it. */
export const [DS_DATE, DS_UNPUBLISHED_DATE] = DATASET_TREES.ds.captures.map((c) => c.name);

/** The first bucket published over {@link DS_DATE}. */
export const DS_BUCKET = DATASET_TREES.ds.buckets_by_date[DS_DATE][0];

/** The select route's answer for a capture holding a label document that will not read, which its
 *  `label_problem` names. */
export const TROUBLED_SELECT_ANSWER = PRODUCED.troubled;

/** The nav route's answer to a step to another image. */
export const NAV_ANSWER = PRODUCED.navigated;

/** The select route's answer for the capture of five photographs. */
export const RAW_SELECT_ANSWER = PRODUCED.raw;

/** The select route's answer for the capture of five photographs made for a subject. */
export const SUBJECT_SELECT_ANSWER = PRODUCED.raw_subject;

/** The subject {@link SUBJECT_SELECT_ANSWER} was made for, the one the annotate scenes save. */
export const SELECTED_SUBJECT = (() => {
  const { subject } = SUBJECT_SELECT_ANSWER.selection;
  if (subject === null) throw new Error("the produced selection names no subject");
  return subject;
})();

/** The select route's answer for the capture of five photographs made for the subject the
 *  produced annotate focus names. */
export const FOCUSED_SELECT_ANSWER = PRODUCED.focused;

/** {@link FOCUSED_SELECT_ANSWER} with the first bucket published over the capture chosen. */
export const FOCUSED_BUCKET_SELECT_ANSWER = PRODUCED.focused_bucket;

/** The selection of the capture of five photographs, `img1.jpg` to `img5.jpg`, the first in view. */
export const RAW_SELECTION = PRODUCED.raw.selection;

/** The selection of a capture of another dataset of the project, holding one band group. */
export const GROUPED_SELECTION = PRODUCED.grouped.selection;

/** The selection of the photographs' capture under its root spelled with a trailing separator:
 *  each image's place is the one the backend's scan found, which no join of the root, the capture
 *  and the image's name rebuilds. */
export const SEPARATED_SELECTION = PRODUCED.separated;

/** The open route's answer the generator received. */
export const PRODUCED_OPEN_ANSWER = PRODUCED.opened;

/** The state envelope the generator received after the open. */
export const PRODUCED_ENVELOPE = PRODUCED.envelope;

/** The project every frontend test opens: the produced answer's. */
export const TEST_PROJECT: SnapshotProject = PRODUCED_OPEN_ANSWER.project;

/** The epoch of the backend the produced envelope came from. */
export const TEST_EPOCH = PRODUCED_ENVELOPE.epoch;

/** {@link TEST_PROJECT} with `change` laid over it: another project for a test needing two. */
export function testProject(change: Partial<SnapshotProject>): SnapshotProject {
  return { ...TEST_PROJECT, ...change };
}

/** The produced envelope with `change` laid over it. */
export function stateEnvelope(change: Partial<StateSnapshotMessage>): StateSnapshotMessage {
  return { ...PRODUCED_ENVELOPE, ...change };
}

/** What a test varies of a produced selection: the fields its capture's layout does not state,
 *  and how many of the capture's images, in order, it keeps. */
export type SelectionChange = Partial<
  Pick<DatasetSelection, "subject" | "bucket" | "current_image_index">
> & { count?: number };

/** `from`, {@link RAW_SELECTION} by default, with `change` laid over it. */
export function selected(
  change: SelectionChange = {},
  from: DatasetSelection = RAW_SELECTION,
): DatasetSelection {
  const { count, ...fields } = change;
  return { ...from, ...fields, images: from.images.slice(0, count) };
}

/** Lay `change` over the selection the store holds, as the browser's own navigation does; a store
 *  holding none throws. */
export function changeSelection(change: Omit<SelectionChange, "count">): void {
  useStore.setState((s) => {
    if (!s.gui.dataset) throw new Error("the store holds no selection to change");
    return { gui: { ...s.gui, dataset: { ...s.gui.dataset, ...change } } };
  });
}

/** Open {@link TEST_PROJECT} in the store on `dataset`, none when it is null, with `gui` over the
 *  rest of the GUI state. */
export function openTestProject(
  dataset: DatasetSelection | null = null,
  gui: Partial<Omit<GuiState, "dataset">> = {},
): void {
  useStore.setState((s) => ({
    gui: { ...s.gui, ...gui, dataset },
    openProject: TEST_PROJECT,
  }));
}
