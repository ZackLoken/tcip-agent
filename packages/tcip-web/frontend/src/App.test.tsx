import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api, type ProjectSummary } from "@/api/client";
import { sessionsApi } from "@/api/sessions";
import { GUI_STATE_DEFAULTS } from "@/api/types.generated";
import { stateSocket } from "@/api/ws";
import App from "@/App";
import { useActiveTabSync } from "@/hooks/useActiveTabSync";
import { useStore } from "@/store";

// App's own socket/tab-sync effects reach the network; only the tab/panel wiring is under
// test here, so both are stubbed rather than left to hit a backend that isn't running.
vi.mock("@/api/ws", () => ({
  stateSocket: {
    connect: vi.fn(),
    close: vi.fn(),
    subscribePanel: vi.fn(() => () => {}),
  },
}));
vi.mock("@/hooks/useActiveTabSync", () => ({ useActiveTabSync: vi.fn() }));
// App statically imports the Annotate tab (not code-split), which pulls in Konva; jsdom has no
// canvas backend, so the Konva module itself is stubbed the same way AnnotateTab's own tests do.
vi.mock("konva", () => ({ default: {} }));
vi.mock("react-konva", () => ({
  Group: (props: { children?: React.ReactNode }) => <>{props.children}</>,
  Rect: () => null,
  Line: () => null,
  Circle: () => null,
  Text: () => null,
}));
vi.mock("@/components/Canvas/CanvasStage", () => ({
  CanvasStage: (props: { children?: React.ReactNode }) => (
    <div data-testid="canvas-stage">{props.children}</div>
  ),
}));
// The Training tab fires its own run listing on mount regardless of any project being open;
// stubbed so switching to it here exercises only the tab/panel wiring, not a real fetch.
vi.mock("@/api/training", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/training")>();
  return {
    ...actual,
    trainingApi: { ...actual.trainingApi, listRuns: vi.fn().mockResolvedValue({ runs: [] }) },
  };
});

vi.mock("@/api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      projects: { ...actual.api.projects, list: vi.fn(), open: vi.fn() },
      dataset: { ...actual.api.dataset, select: vi.fn() },
    },
  };
});

const VALLEY: ProjectSummary = {
  id: "a1b2c3d4e5f6",
  display_name: "Valley farm",
  site: null,
  record_problem: null,
  path: "/ws/valley",
  created: 1_700_000_000,
  modified: 1_700_000_500,
  dates: ["2026-03-01"],
  subjects: ["subject_a"],
  subjects_by_date: { "2026-03-01": ["subject_a"] },
  buckets_by_date: { "2026-03-01": [] },
  image_count: 3,
  label_problem: null,
};

const VALLEY_OPENED = { id: "a1b2c3d4e5f6", display_name: "Valley farm", path: "/ws/valley" };

const VALLEY_SELECTED = {
  status: "ok",
  label_problem: null,
  selection: {
    dataset_root: "/ws/valley",
    subject: "subject_a",
    date: "2026-03-01",
    image_list: [],
    current_image_index: 0,
    images_dir: null,
    bucket: null,
  },
};

function listProjects(openId: string | null) {
  vi.mocked(api.projects.list).mockResolvedValue({
    workspace: "/ws",
    open_id: openId,
    last_opened_problem: null,
    projects: [VALLEY],
  });
}

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.getState().setUser("grower");
  listProjects(null);
  vi.mocked(stateSocket.connect).mockClear();
  vi.mocked(stateSocket.subscribePanel).mockClear();
  vi.mocked(useActiveTabSync).mockClear();
});

afterEach(cleanup);

describe("App tab/panel wiring", () => {
  it("labels the active tab's panel by the selected tab button, and points that button at it", async () => {
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);

    const tab = await screen.findByRole("tab", { name: /meta/i });
    const panel = screen.getByRole("tabpanel");
    expect(tab).toHaveAttribute("aria-controls", panel.id);
    expect(panel).toHaveAttribute("aria-labelledby", tab.id);
  });

  it("moves the panel's id and labelledby to the newly active tab on selection", async () => {
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);
    await screen.findByRole("tabpanel");

    act(() => useStore.getState().setActiveTab("training"));
    const panel = await screen.findByRole("tabpanel");
    const tab = screen.getByRole("tab", { name: /training/i });
    expect(panel.id).toBe("tabpanel-training");
    expect(panel).toHaveAttribute("aria-labelledby", tab.id);
  });
});

describe("App annotator gate", () => {
  it("runs nothing project-shaped until a name is committed in the picker's field", async () => {
    act(() => useStore.getState().setUser(""));
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);

    expect(await screen.findByText("Open a project")).toBeInTheDocument();
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
    expect(screen.queryByRole("tabpanel")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Toggle agent terminal" })).not.toBeInTheDocument();
    expect(screen.queryByText("no project open")).not.toBeInTheDocument();
    expect(stateSocket.connect).not.toHaveBeenCalled();
    expect(stateSocket.subscribePanel).not.toHaveBeenCalled();
    expect(useActiveTabSync).not.toHaveBeenCalled();

    const field = screen.getByLabelText("Annotator");
    fireEvent.change(field, { target: { value: "g" } });
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
    expect(stateSocket.connect).not.toHaveBeenCalled();

    fireEvent.keyDown(field, { key: "Enter" });
    expect(await screen.findByRole("tabpanel")).toBeInTheDocument();
    expect(screen.getByRole("tablist")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Toggle agent terminal" })).toBeInTheDocument();
    expect(screen.getByText("no project open")).toBeInTheDocument();
    expect(stateSocket.connect).toHaveBeenCalledTimes(1);
    expect(stateSocket.subscribePanel).toHaveBeenCalled();
    expect(useActiveTabSync).toHaveBeenCalled();
    expect(useStore.getState().user).toBe("g");
  });
});

describe("App admission through the picker", () => {
  it("keeps the chosen project and shows a failed open after Open commits the name", async () => {
    vi.mocked(api.projects.open).mockRejectedValue(new Error("backend refused"));
    act(() => useStore.getState().setUser(""));
    render(<App />);

    fireEvent.click(await screen.findByText("Valley farm"));
    fireEvent.change(screen.getByLabelText("Annotator"), { target: { value: "jordan" } });
    fireEvent.click(screen.getByRole("button", { name: "Open project" }));

    expect(await screen.findByRole("tablist")).toBeInTheDocument();
    expect(await screen.findAllByText(/backend refused/)).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Open project" })).toBeEnabled();
    expect(useStore.getState().user).toBe("jordan");
    expect(useStore.getState().openChoice.projectId).toBe("a1b2c3d4e5f6");
    expect(api.projects.open).toHaveBeenCalledTimes(1);
  });

  it("opens the backend's open project when a blank first screen is admitted with Enter", async () => {
    listProjects("a1b2c3d4e5f6");
    vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
    vi.mocked(api.dataset.select).mockResolvedValue(VALLEY_SELECTED);
    act(() => useStore.getState().setUser(""));
    render(<App />);

    const field = screen.getByLabelText("Annotator");
    await screen.findByText("Valley farm");
    expect(api.projects.open).not.toHaveBeenCalled();
    fireEvent.change(field, { target: { value: "jordan" } });
    fireEvent.keyDown(field, { key: "Enter" });

    await waitFor(() =>
      expect(api.projects.open).toHaveBeenCalledWith({ id: "a1b2c3d4e5f6", user: "jordan" }),
    );
    expect(await screen.findByRole("tablist")).toBeInTheDocument();
    await waitFor(() =>
      expect(useStore.getState().openProject).toEqual({ id: "a1b2c3d4e5f6", path: "/ws/valley" }),
    );
    expect(useStore.getState().gui.dataset.dataset_root).toBe("/ws/valley");
    expect(useStore.getState().openError).toBeNull();
    expect(useStore.getState().opening).toBeNull();
  });

  it("toasts a failed open that the selected tab hides the picker behind", async () => {
    vi.mocked(api.projects.open).mockRejectedValue(new Error("backend refused"));
    act(() => useStore.getState().setUser(""));
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);

    fireEvent.click(await screen.findByText("Valley farm"));
    fireEvent.change(screen.getByLabelText("Annotator"), { target: { value: "jordan" } });
    fireEvent.click(screen.getByRole("button", { name: "Open project" }));

    expect(await screen.findByRole("tablist")).toBeInTheDocument();
    expect(screen.queryByText("Open a project")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(useStore.getState().toasts.map((t) => t.message)).toContainEqual(
        expect.stringContaining("backend refused"),
      ),
    );
  });

  it("toasts a failed open whose project a snapshot adopted while the open was pending", async () => {
    let reject!: (e: Error) => void;
    vi.mocked(api.projects.open).mockReturnValue(
      new Promise((_, rej) => {
        reject = rej;
      }),
    );
    act(() => useStore.getState().setUser(""));
    render(<App />);
    fireEvent.click(await screen.findByText("Valley farm"));
    fireEvent.change(screen.getByLabelText("Annotator"), { target: { value: "jordan" } });
    fireEvent.click(screen.getByRole("button", { name: "Open project" }));
    await waitFor(() => expect(useStore.getState().opening?.projectId).toBe("a1b2c3d4e5f6"));
    act(() =>
      useStore
        .getState()
        .mergeSnapshot(GUI_STATE_DEFAULTS, 1, { id: "a1b2c3d4e5f6", path: "/ws/valley" }, "e1"),
    );

    await act(async () => reject(new Error("selection failed")));

    expect(useStore.getState().toasts.map((t) => t.message)).toContainEqual(
      expect.stringContaining("selection failed"),
    );
  });
});

describe("App visit attribution across a change of person", () => {
  const RIDGE: ProjectSummary = {
    ...VALLEY,
    id: "0f9e8d7c6b5a",
    display_name: "Ridge farm",
    path: "/ws/ridge",
  };
  const selectionOf = (root: string) => ({
    status: "ok",
    label_problem: null,
    selection: {
      dataset_root: root,
      subject: "subject_a",
      date: "2026-03-01",
      image_list: ["img1.jpg", "img2.jpg"],
      current_image_index: 0,
      images_dir: `${root}/images/2026-03-01`,
      bucket: null,
    },
  });
  const spies: { mockRestore: () => void }[] = [];

  afterEach(() => spies.splice(0).forEach((s) => s.mockRestore()));

  it("attributes the outgoing visit to the first person and project, the next to the second", async () => {
    spies.push(
      vi.spyOn(api.annotate, "load").mockImplementation((path) =>
        Promise.resolve({
          image_path: path,
          img_width: 1000,
          img_height: 800,
          boxes: [],
          polygons: [],
          points: [],
          imageAnnotations: [],
          completion: {},
          flags: [],
          base_mtime: "1",
        }),
      ),
      vi.spyOn(api.annotate, "save"),
      vi.spyOn(api.images, "bands").mockResolvedValue({ band_count: 3, bands: [] }),
      vi.spyOn(api.images, "viewReads").mockResolvedValue({ reads: [] }),
      vi.spyOn(sessionsApi, "imageEvent").mockResolvedValue({
        status: "ok",
        session: { started: "s1", ended: false },
      }),
    );
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      open_id: null,
      last_opened_problem: null,
      projects: [VALLEY, RIDGE],
    });
    vi.mocked(api.projects.open)
      .mockResolvedValueOnce(VALLEY_OPENED)
      .mockResolvedValueOnce({ id: RIDGE.id!, display_name: "Ridge farm", path: "/ws/ridge" });
    vi.mocked(api.dataset.select)
      .mockResolvedValueOnce(selectionOf("/ws/valley"))
      .mockResolvedValueOnce(selectionOf("/ws/ridge"));
    act(() => useStore.getState().setUser(""));
    render(<App />);

    async function admit(card: string, name: string) {
      fireEvent.click(await screen.findByText(card));
      fireEvent.change(screen.getByLabelText("Annotator"), { target: { value: name } });
      fireEvent.click(screen.getByRole("button", { name: "Open project" }));
      await screen.findByTestId("canvas-stage");
      await waitFor(() =>
        expect(useStore.getState().sessionTracking.currentImageName).toBe("img1.jpg"),
      );
    }

    await admit("Valley farm", "first");
    expect(useStore.getState().sessionTracking.projectId).toBe(VALLEY.id);

    fireEvent.click(screen.getByRole("button", { name: "Switch Project" }));
    await screen.findByText("Open a project");
    await admit("Ridge farm", "second");

    expect(vi.mocked(sessionsApi.imageEvent).mock.calls[0][0]).toMatchObject({
      image_name: "img1.jpg",
      user: "first",
      project_id: VALLEY.id,
    });
    expect(useStore.getState().heldContributions).toEqual([]);
    act(() => {
      const s = useStore.getState();
      s.patchGui({ dataset: { ...s.gui.dataset, current_image_index: 1 } });
    });
    await waitFor(() => expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(2));
    expect(vi.mocked(sessionsApi.imageEvent).mock.calls[1][0]).toMatchObject({
      image_name: "img1.jpg",
      user: "second",
      project_id: RIDGE.id,
    });
  });
});

describe("App session end", () => {
  const RECORDED = { project_id: "a1b2c3d4e5f6", started: "2026-10-07T14:00:00+00:00" };

  it("ends the recorded session by its project and start stamp, once, when the shell goes", async () => {
    const send = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));
    useStore.setState({
      openProject: { id: "a1b2c3d4e5f6", path: "/ws/valley" },
      recordedSession: { ...RECORDED, user: "grower" },
    });
    const view = render(<App />);

    window.dispatchEvent(new Event("pagehide"));
    view.unmount();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(send).toHaveBeenCalledTimes(1);
    expect(send).toHaveBeenCalledWith(
      expect.stringContaining("/api/sessions/end"),
      expect.objectContaining({ body: JSON.stringify(RECORDED), keepalive: true }),
    );
    expect(useStore.getState().recordedSession).toBeNull();
  });

  it("sends nothing when no contribution of this page was recorded", async () => {
    const send = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));
    useStore.setState({ openProject: { id: "a1b2c3d4e5f6", path: "/ws/valley" } });
    const view = render(<App />);

    window.dispatchEvent(new Event("pagehide"));
    view.unmount();
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(send).not.toHaveBeenCalled();
  });
});

describe("App unload guard", () => {
  it("guards a refresh/close while the canvas holds unsaved edits", () => {
    render(<App />);
    act(() => useStore.setState((s) => ({ canvas: { ...s.canvas, dirty: true } })));
    const event = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(true);
  });

  it("does not guard when the canvas is clean", () => {
    render(<App />);
    const event = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(event);
    expect(event.defaultPrevented).toBe(false);
  });
});
