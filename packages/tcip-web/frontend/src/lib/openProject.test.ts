import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { openProjectById, openWorkspaceProject } from "@/lib/openProject";
import { useStore } from "@/store";

vi.mock("@/api/client", () => ({
  api: {
    projects: { list: vi.fn(), open: vi.fn() },
    dataset: { select: vi.fn() },
  },
}));

import { api } from "@/api/client";
import type { ProjectSummary } from "@/api/client";

function project(overrides: Partial<ProjectSummary> & { id: string }): ProjectSummary & {
  id: string;
} {
  return {
    display_name: `Project ${overrides.id}`,
    site: "north orchard",
    record_problem: null,
    path: `/ws/${overrides.id}`,
    created: 1,
    modified: 1,
    dates: [],
    subjects: [],
    models: [],
    subjects_by_date: {},
    models_by_date: {},
    image_count: 0,
    is_open: false,
    label_problem: null,
    ...overrides,
  };
}

function listed(projects: ProjectSummary[]) {
  vi.mocked(api.projects.list).mockResolvedValue({
    workspace: "/ws",
    open_id: null,
    last_opened_problem: null,
    projects,
  });
}

beforeEach(() => {
  vi.mocked(api.dataset.select).mockResolvedValue({
    status: "ok",
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    selection: { image_list: [], current_image_index: 0 } as any,
  });
  vi.mocked(api.projects.open).mockImplementation(async (id: string) => ({
    id,
    display_name: `Project ${id}`,
    path: `/ws/${id}`,
  }));
});
afterEach(() => vi.clearAllMocks());

describe("openProjectById", () => {
  it("opens on the newest LABELED date, not the newest date (which would be blank)", async () => {
    listed([
      project({
        id: "hz",
        // Agent just ingested a still-unlabeled 2026-03-24; labels live on 2026-02-11.
        dates: ["2026-02-11", "2026-03-24"],
        subjects: ["bush", "subject_a"], // flat list would pick "bush"
        models: ["baseline"],
        subjects_by_date: { "2026-02-11": ["subject_a"], "2026-03-24": [] },
        models_by_date: { "2026-02-11": ["baseline"], "2026-03-24": [] },
      }),
    ]);

    await openProjectById("hz");

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    // Lands on 2026-02-11 (newest date with labels) + its subject, not the empty newest date.
    expect(arg.date).toBe("2026-02-11");
    expect(arg.subject).toBe("subject_a");
    expect(arg.model_name).toBe("baseline");
  });

  it("falls back to the newest date when nothing is labeled yet (empty project)", async () => {
    listed([
      project({
        id: "fresh",
        dates: ["2026-02-11", "2026-03-24"],
        subjects_by_date: { "2026-02-11": [], "2026-03-24": [] },
        models_by_date: { "2026-02-11": [], "2026-03-24": [] },
      }),
    ]);

    await openProjectById("fresh");

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    expect(arg.date).toBe("2026-03-24"); // newest overall (nothing labeled to prefer)
    expect(arg.subject).toBeNull();
  });

  it("opens the project by its id and selects inside the directory the open answered", async () => {
    listed([project({ id: "site-a", dates: ["2026-02-11"] })]);

    await openProjectById("site-a");

    expect(api.projects.open).toHaveBeenCalledWith("site-a");
    expect(vi.mocked(api.dataset.select).mock.calls[0][0].dataset_root).toBe("/ws/site-a");
    expect(useStore.getState().openProject).toEqual({ id: "site-a", path: "/ws/site-a" });
  });

  it("returns null for an id the workspace does not list", async () => {
    listed([]);
    expect(await openProjectById("nope")).toBeNull();
    expect(api.projects.open).not.toHaveBeenCalled();
    expect(api.dataset.select).not.toHaveBeenCalled();
  });
});

describe("openWorkspaceProject", () => {
  it("pushes a toast naming the label document when the selection carries a label_problem", async () => {
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      selection: { image_list: [], current_image_index: 0 } as any,
      label_problem: "C:/data/annotations/2026-02-11/IMG_0000.json does not decode as JSON",
    });
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    await openWorkspaceProject(
      project({ id: "hz", dates: ["2026-02-11"] }),
      "2026-02-11",
      "subject_a",
      "baseline",
    );

    expect(pushToast).toHaveBeenCalledWith(
      "C:/data/annotations/2026-02-11/IMG_0000.json does not decode as JSON",
    );
  });
});
