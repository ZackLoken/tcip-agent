import { vi } from "vitest";

import { api } from "@/api/client";
import { DATASET_TREES } from "@/test/store";

type DatasetTree = Awaited<ReturnType<typeof api.dataset.tree>>;

/** Answer every dataset-tree read with the produced `tree`, by default the photographs' dataset:
 *  two dates, two buckets published over the first and none over the second. */
export function mockDatasetTree(tree: DatasetTree = DATASET_TREES.ds) {
  return vi.spyOn(api.dataset, "tree").mockResolvedValue(tree);
}
