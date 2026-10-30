import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import type { SessionWrite } from "@/api/types.generated";
import { openWithDefaults, startOpen } from "@/lib/openProject";
import { useStore } from "@/store";
import { openedAs, PICKER } from "@/test/projectRecords";
import { SESSION_WRITES, sessionOf } from "@/test/sessions";
import { RAW_SELECT_ANSWER, stateEnvelope, TROUBLED_SELECT_ANSWER } from "@/test/store";

vi.mock("@/api/client", () => ({
  api: {
    projects: { list: vi.fn(), open: vi.fn() },
    dataset: { select: vi.fn() },
  },
}));

const { valley: VALLEY, ridge: RIDGE, empty: EMPTY } = PICKER;
const LISTED = [VALLEY, RIDGE, EMPTY];

beforeEach(() => {
  vi.mocked(api.dataset.select).mockResolvedValue(RAW_SELECT_ANSWER);
  useStore.getState().setUser("grower");
  vi.mocked(api.projects.open).mockImplementation(({ id }) =>
    Promise.resolve(openedAs(LISTED.find((p) => p.record.id === id)!)),
  );
});

afterEach(() => vi.clearAllMocks());

describe("openWithDefaults", () => {
  it("opens on the newest LABELED date, not the newest date (which would be blank)", async () => {
    const { captures, subjects, subjects_by_date, buckets_by_date } = VALLEY.tree;
    const dates = captures.map((c) => c.name);
    const labeled = dates.filter((date) => subjects_by_date[date].length > 0).at(-1)!;
    // The newest date holds no label, and the flat subject list would pick another subject.
    expect(labeled).not.toBe(dates.at(-1));
    expect(subjects[0]).not.toBe(subjects_by_date[labeled][0]);

    await openWithDefaults(VALLEY);

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    expect(arg.date).toBe(labeled);
    expect(arg.subject).toBe(subjects_by_date[labeled][0]);
    expect(arg.bucket).toBe(buckets_by_date[labeled][0]);
  });

  it("falls back to the newest date when nothing is labeled yet (empty project)", async () => {
    const { captures } = RIDGE.tree;
    expect(captures.length).toBeGreaterThan(1);
    expect(RIDGE.tree.subjects).toEqual([]);

    await openWithDefaults(RIDGE);

    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    expect(arg.date).toBe(captures.at(-1)!.name);
    expect(arg.subject).toBeNull();
  });

  it("opens the project by its id and selects inside the directory the open answered", async () => {
    await openWithDefaults(RIDGE);

    expect(api.projects.open).toHaveBeenCalledWith({ id: RIDGE.record.id, user: "grower" });
    expect(vi.mocked(api.dataset.select).mock.calls[0][0].dataset_root).toBe(
      RIDGE.tree.dataset_root,
    );
    expect(useStore.getState().openProject).toEqual(openedAs(RIDGE).project);
  });

  it("opens a project holding no capture with no dataset selected", async () => {
    await openWithDefaults(EMPTY);

    expect(api.projects.open).toHaveBeenCalledWith({ id: EMPTY.record.id, user: "grower" });
    expect(api.dataset.select).not.toHaveBeenCalled();
    expect(useStore.getState().gui.dataset).toBeNull();
    expect(useStore.getState().openProject).toEqual(openedAs(EMPTY).project);
  });
});

describe("a switch this page starts", () => {
  it("posts the closed visit of the project it leaves before asking to open the next", async () => {
    const { sessionsApi } = await import("@/api/sessions");
    const order: string[] = [];
    vi.spyOn(sessionsApi, "imageEvent").mockImplementation((c) => {
      order.push(`visit ${c.project_id} ${c.started}`);
      return Promise.resolve(SESSION_WRITES.first);
    });
    vi.mocked(api.projects.open).mockImplementation(() => {
      order.push(`open ${RIDGE.record.id}`);
      return Promise.resolve(openedAs(RIDGE));
    });
    const { started } = sessionOf(SESSION_WRITES.first);
    useStore.setState({
      openProject: openedAs(VALLEY).project,
      recordedSession: { project_id: VALLEY.record.id, started, user: useStore.getState().user },
    });
    useStore.getState().startImageSessionTracking("a.jpg", Date.now() - 2000);

    await openWithDefaults(RIDGE);

    expect(order).toEqual([`visit ${VALLEY.record.id} ${started}`, `open ${RIDGE.record.id}`]);
    expect(useStore.getState().heldContributions).toEqual([]);
  });

  it("asks nothing of the backend, and shows why, when the visit it leaves gets no answer", async () => {
    const { sessionsApi } = await import("@/api/sessions");
    vi.spyOn(sessionsApi, "imageEvent").mockRejectedValue(new TypeError("Failed to fetch"));
    useStore.setState({ openProject: openedAs(VALLEY).project, recordedSession: null });
    useStore.getState().startImageSessionTracking("a.jpg", Date.now() - 2000);

    await openWithDefaults(RIDGE);

    expect(api.projects.open).not.toHaveBeenCalled();
    expect(useStore.getState().heldContributions).toEqual([
      expect.objectContaining({ image_name: "a.jpg", project_id: VALLEY.record.id }),
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
    const answer = Promise.withResolvers<SessionWrite>();
    vi.spyOn(sessionsApi, "imageEvent").mockReturnValueOnce(answer.promise);
    useStore.setState({ openProject: openedAs(VALLEY).project, recordedSession: null });
    useStore.getState().startImageSessionTracking("a.jpg", Date.now() - 2000);

    const opening = openWithDefaults(RIDGE);
    await Promise.resolve();
    const s = useStore.getState();
    s.mergeSnapshot(
      stateEnvelope({
        state: s.gui,
        version: s.wsVersion,
        project: openedAs(EMPTY).project,
      }),
    );
    answer.resolve(SESSION_WRITES.first);
    await opening;

    expect(api.projects.open).not.toHaveBeenCalled();
  });
});

describe("startOpen", () => {
  it("pushes a toast naming the label document when the selection carries a label_problem", async () => {
    vi.mocked(api.dataset.select).mockResolvedValue(TROUBLED_SELECT_ANSWER);
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");
    const [{ name: date }] = VALLEY.tree.captures;

    await startOpen(VALLEY, date, VALLEY.tree.subjects_by_date[date][0], null);

    expect(pushToast).toHaveBeenCalledWith(TROUBLED_SELECT_ANSWER.label_problem);
  });
});
