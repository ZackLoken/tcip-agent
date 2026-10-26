import type { OpenAnswer, SnapshotProject, StateSnapshotMessage } from "@/api/types.generated";
import { useStore } from "@/store";
import type { DatasetSelection, GuiState } from "@/store/types";
import produced from "@/test/openFixtures.json";

/** What `tools/generate_open_fixture.py` received from a backend opening a project: the open
 *  route's answer and the envelope `/ws/state` sent next. */
const PRODUCED = produced as { opened: OpenAnswer; envelope: StateSnapshotMessage };

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

/** Open {@link TEST_PROJECT} in the store, with `dataset` laid over the current dataset
 * selection and `gui` over the rest of the GUI state. */
export function openTestProject(
  dataset: Partial<DatasetSelection> = {},
  gui: Partial<Omit<GuiState, "dataset">> = {},
): void {
  useStore.setState((s) => ({
    gui: { ...s.gui, ...gui, dataset: { ...s.gui.dataset, ...dataset } },
    openProject: TEST_PROJECT,
  }));
}
