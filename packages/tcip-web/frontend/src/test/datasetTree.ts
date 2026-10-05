import { vi } from "vitest";

import { api } from "@/api/client";

type DatasetTree = Awaited<ReturnType<typeof api.dataset.tree>>;

/** Answer every dataset-tree read with one dated image set under C:/data holding one bucket,
 * `overrides` laid over it. */
export function mockDatasetTree(overrides: Partial<DatasetTree> = {}) {
  return vi.spyOn(api.dataset, "tree").mockResolvedValue({
    dataset_root: "C:/data",
    dates_with_images: ["2026-01-01"],
    subjects: ["subject_a"],
    subjects_by_date: {},
    buckets_by_date: { "2026-01-01": ["baseline/2026-01-01"] },
    label_problem: null,
    ...overrides,
  });
}
