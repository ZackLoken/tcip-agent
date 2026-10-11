import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api, type ProjectSummary } from "@/api/client";
import { GUI_STATE_DEFAULTS, type SessionWrite } from "@/api/types.generated";
import { openWithDefaults, startOpen } from "@/lib/openProject";
import { useStore } from "@/store";

vi.mock("@/api/client", () => ({
  api: {
    projects: { list: vi.fn(), open: vi.fn() },
    dataset: { select: vi.fn() },
  },
}));

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
    subjects_by_date: {},
    buckets_by_date: {},
    image_count: 0,
    label_problem: null,
    ...overrides,
  };
}

beforeEach(() => {
  vi.mocked(api.dataset.select).mockResolvedValue({
    status: "ok",
    selection: GUI_STATE_DEFAULTS.dataset,
  });
  useStore.getState().setUser("grower");
  vi.mocked(api.projects.open).mockImplementation(({ id }) =>
    Promise.resolve({ id, display_name: `Project ${id}`, path: `/ws/${id}` }),
  );
});
afterEach(() => vi.clearAllMocks());

describe("openWithDefaults", () => {
  it("opens on the newest LABELED date, not the newest date (which would be blank)", async () => {
    await openWithDefaults(
      project({
        id: "hz",
        // Agent just ingested a still-unlabeled 2026-03-24; labels live on 2026-02-11.
        dates: ["2026-02-11", "2026-03-24"],
        subjects: ["bush", "subject_a"], // flat list would pick "bush"
        subjects_by_date: { "2026-02-11": ["subject_a"], "2026-03-24": [] },
        buckets_by_date: { "2026-02-11": ["baseline"], "2026-03-24": [] },
      }),
    );

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    // Lands on 2026-02-11 (newest date with labels) + its subject, not the empty newest date.
    expect(arg.date).toBe("2026-02-11");
    expect(arg.subject).toBe("subject_a");
    expect(arg.bucket).toBe("baseline");
  });

  it("falls back to the newest date when nothing is labeled yet (empty project)", async () => {
    await openWithDefaults(
      project({
        id: "fresh",
        dates: ["2026-02-11", "2026-03-24"],
        subjects_by_date: { "2026-02-11": [], "2026-03-24": [] },
        buckets_by_date: { "2026-02-11": [], "2026-03-24": [] },
      }),
    );

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    expect(arg.date).toBe("2026-03-24"); // newest overall (nothing labeled to prefer)
    expect(arg.subject).toBeNull();
  });

  it("opens the project by its id and selects inside the directory the open answered", async () => {
    await openWithDefaults(project({ id: "site-a", dates: ["2026-02-11"] }));

    expect(api.projects.open).toHaveBeenCalledWith({ id: "site-a", user: "grower" });
    expect(vi.mocked(api.dataset.select).mock.calls[0][0].dataset_root).toBe("/ws/site-a");
    expect(useStore.getState().openProject).toEqual({ id: "site-a", path: "/ws/site-a" });
  });

  it("opens a project holding no capture with no dataset selected", async () => {
    await openWithDefaults(project({ id: "empty" }));

    expect(api.projects.open).toHaveBeenCalledWith({ id: "empty", user: "grower" });
    expect(api.dataset.select).not.toHaveBeenCalled();
    expect(useStore.getState().gui.dataset.dataset_root).toBeNull();
    expect(useStore.getState().openProject).toEqual({ id: "empty", path: "/ws/empty" });
  });
});

describe("a switch this page starts", () => {
  it("posts the closed visit of the project it leaves before asking to open the next", async () => {
    const { sessionsApi } = await import("@/api/sessions");
    const order: string[] = [];
    vi.spyOn(sessionsApi, "imageEvent").mockImplementation((c) => {
      order.push(`visit ${c.project_id} ${c.started}`);
      return Promise.resolve({
        status: "ok",
        session: { started: c.started ?? "s1", ended: false },
      });
    });
    vi.mocked(api.projects.open).mockImplementation(({ id }) => {
      order.push(`open ${id}`);
      return Promise.resolve({ id, display_name: `Project ${id}`, path: `/ws/${id}` });
    });
    useStore.setState({
      openProject: { id: "valley", path: "/ws/valley" },
      recordedSession: { project_id: "valley", started: "s1", user: "user:grower" },
    });
    useStore.setState({ user: "user:grower" });
    useStore.getState().startImageSessionTracking("a.jpg", Date.now() - 2000);

    await openWithDefaults(project({ id: "ridge" }));

    expect(order).toEqual(["visit valley s1", "open ridge"]);
    expect(useStore.getState().heldContributions).toEqual([]);
  });

  it("asks nothing of the backend, and shows why, when the visit it leaves gets no answer", async () => {
    const { sessionsApi } = await import("@/api/sessions");
    vi.spyOn(sessionsApi, "imageEvent").mockRejectedValue(new TypeError("Failed to fetch"));
    useStore.setState({ openProject: { id: "valley", path: "/ws/valley" }, recordedSession: null });
    useStore.getState().startImageSessionTracking("a.jpg", Date.now() - 2000);

    await openWithDefaults(project({ id: "ridge" }));

    expect(api.projects.open).not.toHaveBeenCalled();
    expect(useStore.getState().heldContributions).toEqual([
      expect.objectContaining({ image_name: "a.jpg", project_id: "valley" }),
    ]);
    expect(useStore.getState().openError?.message).toMatch(/could not be sent/);

    useStore.getState().incrementAnnotationsAdded();
    expect(useStore.getState().sessionTracking).toMatchObject({
      currentImageName: "a.jpg",
      annotationsAddedDelta: 1,
    });
    expect(typeof useStore.getState().sessionTracking?.imageEnterTimeMs).toBe("number");
    useStore.setState({ heldContributions: [], unconfirmedContributions: [] });
  });

  it("sends no open once a snapshot adopting another project superseded it during settlement", async () => {
    const { sessionsApi } = await import("@/api/sessions");
    let answer!: (w: SessionWrite) => void;
    vi.spyOn(sessionsApi, "imageEvent").mockReturnValueOnce(
      new Promise((resolve) => {
        answer = resolve;
      }),
    );
    useStore.setState({ openProject: { id: "valley", path: "/ws/valley" }, recordedSession: null });
    useStore.getState().startImageSessionTracking("a.jpg", Date.now() - 2000);

    const opening = openWithDefaults(project({ id: "ridge" }));
    await Promise.resolve();
    const s = useStore.getState();
    s.mergeSnapshot(s.gui, null, { id: "hill", path: "/ws/hill" }, null);
    answer({ status: "ok", session: { started: "s1", ended: false } });
    await opening;

    expect(api.projects.open).not.toHaveBeenCalled();
  });
});

describe("startOpen", () => {
  it("pushes a toast naming the label document when the selection carries a label_problem", async () => {
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      selection: GUI_STATE_DEFAULTS.dataset,
      label_problem: "label_documents['2026-02-11', 'IMG_0000'] under C:/data: is a list",
    });
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    await startOpen(
      project({ id: "hz", dates: ["2026-02-11"] }),
      "2026-02-11",
      "subject_a",
      "baseline",
    );

    expect(pushToast).toHaveBeenCalledWith(
      "label_documents['2026-02-11', 'IMG_0000'] under C:/data: is a list",
    );
  });
});
