import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { adoptProjectByName, adoptWorkspaceProject, openProjectByName } from "@/lib/openProject";
import { useStore } from "@/store";

vi.mock("@/api/client", () => ({
  api: {
    projects: { list: vi.fn(), setActive: vi.fn() },
    dataset: { select: vi.fn() },
  },
}));

import { api } from "@/api/client";
import type { ProjectSummary } from "@/api/client";

function project(overrides: Partial<ProjectSummary> & { name: string }): ProjectSummary {
  return {
    path: `/ws/${overrides.name}`,
    created: 1,
    modified: 1,
    dates: [],
    subjects: [],
    models: [],
    subjects_by_date: {},
    models_by_date: {},
    image_count: 0,
    is_active: false,
    ...overrides,
  };
}

beforeEach(() => {
  vi.mocked(api.dataset.select).mockResolvedValue({
    status: "ok",
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    selection: {} as any,
  });
  vi.mocked(api.projects.setActive).mockResolvedValue({ name: "x", path: "/x" });
});
afterEach(() => vi.clearAllMocks());

describe("openProjectByName", () => {
  it("opens on the newest LABELLED date, not the newest date (which would be blank)", async () => {
    const p = project({
      name: "hz",
      // Agent just ingested a still-unlabelled 2026-03-24; labels live on 2026-02-11.
      dates: ["2026-02-11", "2026-03-24"],
      subjects: ["bush", "subject_a"], // flat list would pick "bush"
      models: ["baseline"],
      subjects_by_date: { "2026-02-11": ["subject_a"], "2026-03-24": [] },
      models_by_date: { "2026-02-11": ["baseline"], "2026-03-24": [] },
    });
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: [p],
    });

    await openProjectByName("hz");

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    // Lands on 2026-02-11 (newest date with labels) + its subject, not the empty newest date.
    expect(arg.date).toBe("2026-02-11");
    expect(arg.subject).toBe("subject_a");
    expect(arg.model_name).toBe("baseline");
  });

  it("falls back to the newest date when nothing is labelled yet (empty project)", async () => {
    const p = project({
      name: "fresh",
      dates: ["2026-02-11", "2026-03-24"],
      subjects: [],
      models: [],
      subjects_by_date: { "2026-02-11": [], "2026-03-24": [] },
      models_by_date: { "2026-02-11": [], "2026-03-24": [] },
    });
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: [p],
    });

    await openProjectByName("fresh");

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    expect(arg.date).toBe("2026-03-24"); // newest overall (nothing labelled to prefer)
    expect(arg.subject).toBeNull();
  });

  it("uses the labelled subject/model when the default date has them", async () => {
    const p = project({
      name: "hz2",
      dates: ["2026-02-11"],
      subjects: ["subject_a"],
      models: ["baseline"],
      subjects_by_date: { "2026-02-11": ["subject_a"] },
      models_by_date: { "2026-02-11": ["baseline"] },
    });
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: [p],
    });

    await openProjectByName("hz2");

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    expect(arg.subject).toBe("subject_a");
    expect(arg.model_name).toBe("baseline");
  });

  it("points both roots at the summary's own path, the same place the picker opens", async () => {
    const p = project({
      name: "site-a",
      dates: ["2026-02-11"],
      subjects: ["bush"],
      models: ["baseline"],
      subjects_by_date: { "2026-02-11": ["bush"] },
      models_by_date: { "2026-02-11": ["baseline"] },
    });
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: [p],
    });

    await openProjectByName("site-a");

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    expect(arg.project_root).toBe("/ws/site-a");
    expect(arg.dataset_root).toBe("/ws/site-a");
  });

  it("returns null for an unknown project", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: [],
    });
    expect(await openProjectByName("nope")).toBeNull();
    expect(api.dataset.select).not.toHaveBeenCalled();
  });

  it("writes no active-project marker", async () => {
    const p = project({ name: "hz", dates: ["2026-02-11"] });
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: [p],
    });

    await openProjectByName("hz");

    expect(api.projects.setActive).not.toHaveBeenCalled();
  });
});

describe("adoptProjectByName", () => {
  it("opens the project and writes the active-project marker", async () => {
    const p = project({ name: "hz", dates: ["2026-02-11"] });
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: [p],
    });

    await adoptProjectByName("hz");

    expect(api.dataset.select).toHaveBeenCalledTimes(1);
    expect(api.projects.setActive).toHaveBeenCalledWith("hz");
  });

  it("returns null for an unknown project and writes no marker", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: [],
    });

    expect(await adoptProjectByName("nope")).toBeNull();
    expect(api.projects.setActive).not.toHaveBeenCalled();
  });
});

describe("adoptWorkspaceProject", () => {
  it("opens the project and adopts it", async () => {
    const p = project({ name: "hz", dates: ["2026-02-11"] });

    await adoptWorkspaceProject(p, "2026-02-11", "subject_a", "baseline");

    expect(api.dataset.select).toHaveBeenCalledTimes(1);
    expect(api.projects.setActive).toHaveBeenCalledWith("hz");
  });

  it("still returns the selection and pushes a toast when the marker write is rejected", async () => {
    const p = project({ name: "hz", dates: ["2026-02-11"] });
    vi.mocked(api.projects.setActive).mockRejectedValue(new Error("locked"));
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    const selection = await adoptWorkspaceProject(p, "2026-02-11", "subject_a", "baseline");
    expect(selection).toBeDefined();

    // The marker write is fire-and-forget: await a microtask for its rejection to settle.
    await Promise.resolve();
    expect(pushToast).toHaveBeenCalledWith(expect.stringContaining("hz"));
  });
});
