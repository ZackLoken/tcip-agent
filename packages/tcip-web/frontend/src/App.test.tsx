import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import { sessionsApi } from "@/api/sessions";
import type { OpenAnswer, Panel, PanelEnvelope } from "@/api/types.generated";
import App from "@/App";
import { useActiveTabSync } from "@/hooks/useActiveTabSync";
import { useStore } from "@/store";
import { loadOf } from "@/test/annotate";
import { BAND_ANSWERS } from "@/test/bands";
import { PANEL_EVENTS } from "@/test/panelEvents";
import { openedAs, PICKER, PROJECT_FIXTURES } from "@/test/projectRecords";
import { SESSION_ANSWERS, sessionOf } from "@/test/sessions";
import { changeSelection, stateEnvelope, SUBJECT_SELECT_ANSWER } from "@/test/store";

// The state socket and the active-tab sync stand in for App's network effects.
const stateSocket = vi.hoisted(() => ({
  connect: vi.fn(),
  close: vi.fn(),
  subscribePanel: vi.fn<(panel: Panel, handler: (envelope: PanelEnvelope) => void) => () => void>(
    () => () => {},
  ),
}));
vi.mock("@/api/ws", () => ({ stateSocket }));
vi.mock("@/hooks/useActiveTabSync", () => ({ useActiveTabSync: vi.fn() }));
// App statically imports the Annotate tab, which pulls in Konva, and jsdom has no canvas backend.
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
// The Training tab lists its runs on mount, whether or not a project is open; the listing holds
// no run.
vi.mock("@/api/training", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/training")>();
  return {
    ...actual,
    trainingApi: {
      ...actual.trainingApi,
      listRuns: vi.fn().mockResolvedValue({ runs: [], sweeps: [] }),
    },
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

const { valley: VALLEY, ridge: RIDGE } = PICKER;
const VALLEY_NAME = VALLEY.record.display_name;

const VALLEY_OPENED = openedAs(VALLEY);

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.getState().setUser("grower");
  vi.mocked(api.projects.list).mockResolvedValue(PROJECT_FIXTURES.picker);
  stateSocket.connect.mockClear();
  stateSocket.subscribePanel.mockClear();
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

describe("App panel events", () => {
  it("shows the banner its panel's socket delivers above that panel's tab", async () => {
    const { banner } = PANEL_EVENTS;
    act(() => useStore.getState().setActiveTab("results"));
    render(<App />);
    await screen.findByRole("tabpanel");

    const subscribed = stateSocket.subscribePanel.mock.calls.find(([p]) => p === banner.panel);
    act(() => subscribed?.[1](banner));

    expect(screen.getByText(banner.event.text)).toBeInTheDocument();
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

    const field = screen.getByLabelText("Your name");
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

    fireEvent.click(await screen.findByText(VALLEY_NAME));
    fireEvent.change(screen.getByLabelText("Your name"), { target: { value: "jordan" } });
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    expect(await screen.findByRole("tablist")).toBeInTheDocument();
    expect(await screen.findAllByText(/backend refused/)).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Open" })).toBeEnabled();
    expect(useStore.getState().user).toBe("jordan");
    expect(useStore.getState().openChoice.projectId).toBe(VALLEY.record.id);
    expect(api.projects.open).toHaveBeenCalledTimes(1);
  });

  it("admits a blank first screen with Enter and preselects the open project without opening it", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(PROJECT_FIXTURES.picker_open);
    act(() => useStore.getState().setUser(""));
    render(<App />);

    const field = screen.getByLabelText("Your name");
    await screen.findByText(VALLEY_NAME);
    fireEvent.change(field, { target: { value: "jordan" } });
    fireEvent.keyDown(field, { key: "Enter" });

    expect(await screen.findByRole("tablist")).toBeInTheDocument();
    await waitFor(() => expect(useStore.getState().openChoice.projectId).toBe(VALLEY.record.id));
    expect(api.projects.open).not.toHaveBeenCalled();
    expect(useStore.getState().openProject).toBeNull();
    expect(useStore.getState().opening).toBeNull();
  });

  it("toasts a failed open that the selected tab hides the picker behind", async () => {
    vi.mocked(api.projects.open).mockRejectedValue(new Error("backend refused"));
    act(() => useStore.getState().setUser(""));
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);

    fireEvent.click(await screen.findByText(VALLEY_NAME));
    fireEvent.change(screen.getByLabelText("Your name"), { target: { value: "jordan" } });
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    expect(await screen.findByRole("tablist")).toBeInTheDocument();
    expect(screen.queryByText("Open a project")).not.toBeInTheDocument();
    await waitFor(() =>
      expect(useStore.getState().toasts.map((t) => t.message)).toContainEqual(
        expect.stringContaining("backend refused"),
      ),
    );
  });

  it("toasts a failed open whose project a snapshot adopted while the open was pending", async () => {
    const open = Promise.withResolvers<OpenAnswer>();
    vi.mocked(api.projects.open).mockReturnValue(open.promise);
    act(() => useStore.getState().setUser(""));
    render(<App />);
    fireEvent.click(await screen.findByText(VALLEY_NAME));
    fireEvent.change(screen.getByLabelText("Your name"), { target: { value: "jordan" } });
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    await waitFor(() => expect(useStore.getState().opening?.projectId).toBe(VALLEY.record.id));
    act(() =>
      useStore
        .getState()
        .mergeSnapshot(stateEnvelope({ version: 1, project: VALLEY_OPENED.project })),
    );

    await act(() => Promise.resolve(open.reject(new Error("selection failed"))));

    expect(useStore.getState().toasts.map((t) => t.message)).toContainEqual(
      expect.stringContaining("selection failed"),
    );
  });
});

describe("App visit attribution across a change of person", () => {
  const answer = SUBJECT_SELECT_ANSWER;
  const FIRST_IMAGE = answer.selection.images[0].name;
  it("attributes the outgoing visit to the first person and project, the next to the second", async () => {
    vi.spyOn(api.annotate, "load").mockImplementation((path) => Promise.resolve(loadOf(path)));
    vi.spyOn(api.annotate, "save");
    vi.spyOn(api.images, "bands").mockResolvedValue(BAND_ANSWERS.plain);
    vi.spyOn(api.images, "viewReads").mockResolvedValue({ reads: [] });
    vi.spyOn(sessionsApi, "imageEvent").mockResolvedValue(SESSION_ANSWERS.landed);
    vi.mocked(api.projects.open)
      .mockResolvedValueOnce(VALLEY_OPENED)
      .mockResolvedValueOnce(openedAs(RIDGE));
    vi.mocked(api.dataset.select).mockResolvedValueOnce(answer).mockResolvedValueOnce(answer);
    act(() => useStore.getState().setUser(""));
    render(<App />);

    async function admit(card: string, name: string) {
      fireEvent.click(await screen.findByText(card));
      fireEvent.change(screen.getByLabelText("Your name"), { target: { value: name } });
      fireEvent.click(screen.getByRole("button", { name: "Open" }));
      await screen.findByTestId("canvas-stage");
      await waitFor(() =>
        expect(useStore.getState().sessionTracking.currentImageName).toBe(FIRST_IMAGE),
      );
    }

    await admit(VALLEY_NAME, "first");
    expect(useStore.getState().sessionTracking.projectId).toBe(VALLEY.record.id);

    fireEvent.click(screen.getByRole("button", { name: "Switch project" }));
    await screen.findByText("Open a project");
    await admit(RIDGE.record.display_name, "second");

    expect(vi.mocked(sessionsApi.imageEvent).mock.calls[0][0]).toMatchObject({
      image_name: FIRST_IMAGE,
      user: "first",
      project_id: VALLEY.record.id,
    });
    expect(useStore.getState().heldContributions).toEqual([]);
    act(() => changeSelection({ current_image_index: 1 }));
    await waitFor(() => expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(2));
    expect(vi.mocked(sessionsApi.imageEvent).mock.calls[1][0]).toMatchObject({
      image_name: FIRST_IMAGE,
      user: "second",
      project_id: RIDGE.record.id,
    });
  });
});

describe("App session end", () => {
  const RECORDED = {
    project_id: VALLEY.record.id,
    started: sessionOf(SESSION_ANSWERS.landed).started,
  };

  it("ends the recorded session by its project and start stamp, once, when the shell goes", async () => {
    const send = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));
    useStore.setState({
      openProject: VALLEY_OPENED.project,
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
    useStore.setState({ openProject: VALLEY_OPENED.project });
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
