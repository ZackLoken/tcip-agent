import { useStore } from "@/store";
import type { DatasetSelection, GuiState, OpenProject } from "@/store/types";

/** The project every frontend test opens. */
export const TEST_PROJECT: OpenProject = { id: "a1b2c3d4e5f6", path: "C:/proj" };

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
