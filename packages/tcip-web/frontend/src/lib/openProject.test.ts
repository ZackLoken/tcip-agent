import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { openProjectByName } from "@/lib/openProject";

vi.mock("@/api/client", () => ({
  api: { projects: { list: vi.fn() }, dataset: { select: vi.fn() } },
}));

import { api } from "@/api/client";
import type { ProjectSummary } from "@/api/client";

function project(overrides: Partial<ProjectSummary> & { name: string }): ProjectSummary {
  return {
    path: `/ws/${overrides.name}`,
    created: 1,
    modified: 1,
    dates: [],
    traits: [],
    models: [],
    traits_by_date: {},
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
});
afterEach(() => vi.clearAllMocks());

describe("openProjectByName", () => {
  it("opens on the newest LABELLED date, not the newest date (which would be blank)", async () => {
    const p = project({
      name: "hz",
      // Agent just ingested a still-unlabelled 2026-03-24; labels live on 2026-02-11.
      dates: ["2026-02-11", "2026-03-24"],
      traits: ["bush", "catkin"], // flat list would pick "bush"
      models: ["baseline"],
      traits_by_date: { "2026-02-11": ["catkin"], "2026-03-24": [] },
      models_by_date: { "2026-02-11": ["baseline"], "2026-03-24": [] },
    });
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      projects: [p],
    });

    await openProjectByName("hz");

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    // Lands on 2026-02-11 (newest date with labels) + its trait, not the empty newest date.
    expect(arg.date).toBe("2026-02-11");
    expect(arg.annotation_type).toBe("catkin");
    expect(arg.model_name).toBe("baseline");
  });

  it("falls back to the newest date when NOTHING is labelled yet (empty project)", async () => {
    const p = project({
      name: "fresh",
      dates: ["2026-02-11", "2026-03-24"],
      traits: [],
      models: [],
      traits_by_date: { "2026-02-11": [], "2026-03-24": [] },
      models_by_date: { "2026-02-11": [], "2026-03-24": [] },
    });
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      projects: [p],
    });

    await openProjectByName("fresh");

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    expect(arg.date).toBe("2026-03-24"); // newest overall (nothing labelled to prefer)
    expect(arg.annotation_type).toBeNull();
  });

  it("uses the labelled trait/model when the default date has them", async () => {
    const p = project({
      name: "hz2",
      dates: ["2026-02-11"],
      traits: ["catkin"],
      models: ["baseline"],
      traits_by_date: { "2026-02-11": ["catkin"] },
      models_by_date: { "2026-02-11": ["baseline"] },
    });
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      projects: [p],
    });

    await openProjectByName("hz2");

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    expect(arg.annotation_type).toBe("catkin");
    expect(arg.model_name).toBe("baseline");
  });

  it("returns null for an unknown project", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      projects: [],
    });
    expect(await openProjectByName("nope")).toBeNull();
    expect(api.dataset.select).not.toHaveBeenCalled();
  });
});
