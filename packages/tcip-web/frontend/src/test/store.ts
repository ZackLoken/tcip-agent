import type {
  OpenAnswer,
  SelectAnswer,
  SnapshotProject,
  StateSnapshotMessage,
} from "@/api/types.generated";
import { useStore } from "@/store";
import type { DatasetSelection, GuiState } from "@/store/types";
import produced from "@/test/openFixtures.json";

/** What `tools/generate_open_fixture.py` received from a backend opening a project: the open
 *  route's answer, the envelope `/ws/state` sent next, the select route's answers for a capture
 *  of five photographs and a capture of one band group, and the photographs' selection under a
 *  separator-ended root. */
const PRODUCED = produced as {
  opened: OpenAnswer;
  envelope: StateSnapshotMessage;
  raw: SelectAnswer;
  grouped: SelectAnswer;
  separated: DatasetSelection;
};

/** The select route's answer for the capture of five photographs. */
export const RAW_SELECT_ANSWER = PRODUCED.raw;

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

/** The produced open answer for the project `change` names: its id and directory laid over the
 *  produced project's, and its display name over the produced one. */
export function openAnswer(
  change: Partial<SnapshotProject> & { display_name?: string },
): OpenAnswer {
  const { display_name = PRODUCED_OPEN_ANSWER.display_name, ...project } = change;
  return { ...PRODUCED_OPEN_ANSWER, project: testProject(project), display_name };
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

/** The produced select answer holding `selection`, with `label_problem`. */
export function selectAnswer(
  selection: DatasetSelection,
  label_problem: string | null = null,
): SelectAnswer {
  return { ...RAW_SELECT_ANSWER, selection, label_problem };
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
