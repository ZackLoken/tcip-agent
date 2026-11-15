import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import { sessionsApi } from "@/api/sessions";
import { subjectsApi } from "@/api/subjects";
import type { OpenAnswer, Panel, PanelEnvelope } from "@/api/types.generated";
import App from "@/App";
import { useActiveTabSync } from "@/hooks/useActiveTabSync";
import { postHeldContributions } from "@/lib/sessionLifecycle";
import { useStore } from "@/store";
import { loadOf } from "@/test/annotate";
import { BAND_ANSWERS } from "@/test/bands";
import { measureWindow, PANE_FOCUS } from "@/test/dock";
import { LABEL_ANSWERS } from "@/test/labels";
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

const { valley: VALLEY, ridge: RIDGE, empty: EMPTY } = PICKER;
const VALLEY_NAME = VALLEY.record.display_name;

const VALLEY_OPENED = openedAs(VALLEY);

/** The sentence of every listed notifications entry, oldest first. */
const listedSentences = () => useStore.getState().notifications.map((e) => e.sentence);

const initialStoreState = useStore.getState();

beforeEach(() => {
  measureWindow();
  useStore.setState(initialStoreState, true);
  useStore.getState().setUser("grower");
  vi.mocked(api.projects.list).mockResolvedValue(PROJECT_FIXTURES.picker);
  stateSocket.connect.mockClear();
  stateSocket.close.mockClear();
  stateSocket.subscribePanel.mockClear();
  vi.mocked(useActiveTabSync).mockClear();
});

afterEach(cleanup);

describe("App tab/panel wiring", () => {
  it("labels the active tab's panel by the selected tab button, and points that button at it", async () => {
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);

    const tab = await screen.findByRole("tab", { name: /meta/i });
    const panel = screen.getByRole("tabpanel", { name: /meta/i });
    expect(tab).toHaveAttribute("aria-controls", panel.id);
    expect(panel).toHaveAttribute("aria-labelledby", tab.id);
  });

  it("moves the panel's id and labelledby to the newly active tab on selection", async () => {
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);
    await screen.findByRole("tabpanel", { name: /meta/i });

    act(() => useStore.getState().setActiveTab("training"));
    const panel = await screen.findByRole("tabpanel", { name: /training/i });
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
    await screen.findByRole("tabpanel", { name: /results/i });

    const subscribed = stateSocket.subscribePanel.mock.calls.find(([p]) => p === banner.panel);
    act(() => subscribed?.[1](banner));

    expect(screen.getByText(banner.event.text)).toBeInTheDocument();
  });
});

describe("App shell failures", () => {
  const CAPTURE = SUBJECT_SELECT_ANSWER.selection;
  const CONTEXT = { project: VALLEY.record.id, target: CAPTURE.date };

  /** Render the shell over Valley farm open on the produced capture. */
  function renderOnCapture() {
    answerAnnotateReads();
    useStore.setState((s) => ({
      openProject: VALLEY_OPENED.project,
      gui: { ...s.gui, dataset: CAPTURE },
    }));
    render(<App />);
  }

  it("lists label documents that will not read as a failure of the project and capture", async () => {
    const { registry_unreadable: loaded } = LABEL_ANSWERS;
    vi.spyOn(subjectsApi, "load").mockResolvedValue(loaded);
    renderOnCapture();

    await waitFor(() =>
      expect(useStore.getState().notifications).toContainEqual(
        expect.objectContaining({
          kind: "failure",
          sentence: `1 label document(s) could not be read: ${loaded.unreadable.join(", ")}`,
          context: CONTEXT,
        }),
      ),
    );
  });

  it("lists a failed registry load as a failure of the project and capture", async () => {
    vi.spyOn(subjectsApi, "load").mockRejectedValue(new TypeError("Failed to fetch"));
    renderOnCapture();

    await waitFor(() =>
      expect(useStore.getState().notifications).toContainEqual(
        expect.objectContaining({
          kind: "failure",
          sentence: "Could not load the registry for this project.",
          context: CONTEXT,
        }),
      ),
    );
  });

  it("lists an agent focus that could not be applied as a failure of the open project", async () => {
    vi.mocked(api.dataset.select).mockRejectedValue(new Error("selection refused"));
    useStore.setState({ openProject: VALLEY_OPENED.project });
    render(<App />);
    await screen.findByRole("tablist");

    const app = stateSocket.subscribePanel.mock.calls.find(([p]) => p === "app");
    act(() => app?.[1](PANEL_EVENTS.annotate_focus));

    await waitFor(() =>
      expect(useStore.getState().notifications).toEqual([
        expect.objectContaining({
          kind: "failure",
          sentence: "Agent tried to focus the Annotate tab, but it couldn't be applied.",
          context: { project: VALLEY.record.id },
        }),
      ]),
    );
  });
});

describe("App annotator gate", () => {
  it("runs nothing project-shaped until a name is committed in the picker's field", async () => {
    act(() => useStore.getState().setUser(""));
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);

    expect(await screen.findByText("Open a project")).toBeInTheDocument();
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
    expect(screen.queryByRole("tabpanel", { name: /meta/i })).not.toBeInTheDocument();
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
    expect(await screen.findByRole("tabpanel", { name: /meta/i })).toBeInTheDocument();
    expect(screen.getByRole("tablist")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Toggle agent terminal" })).toBeInTheDocument();
    expect(screen.getByText("no project open")).toBeInTheDocument();
    expect(stateSocket.connect).toHaveBeenCalledTimes(1);
    expect(stateSocket.subscribePanel).toHaveBeenCalled();
    expect(useActiveTabSync).toHaveBeenCalled();
    expect(useStore.getState().user).toBe("g");
  });
});

describe("App notifications area", () => {
  it("is reached from the name screen's top bar, and shows no lost connection there", async () => {
    act(() => useStore.getState().setUser(""));
    render(<App />);
    await screen.findByText("Open a project");

    const button = screen.getByRole("button", { name: "Notifications" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByTestId("condition-mark")).not.toBeInTheDocument();

    fireEvent.click(button);
    const heading = await PANE_FOCUS.heading();
    expect(heading).toBeVisible();
    expect(button).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/Nothing to report/)).toBeVisible();
    await waitFor(() => expect(heading).toHaveFocus());

    fireEvent.keyDown(heading, { key: "Escape" });
    expect(button).toHaveAttribute("aria-expanded", "false");
    await waitFor(() => expect(screen.getByRole("tab", { name: "Notifications" })).toHaveFocus());
    expect(screen.getByText("Open a project")).toBeInTheDocument();
  });

  it("announces a condition raised while the area was never opened", async () => {
    vi.spyOn(sessionsApi, "imageEvent").mockRejectedValue(new TypeError("Failed to fetch"));
    useStore.setState({ openProject: VALLEY_OPENED.project });
    render(<App />);
    await screen.findByRole("tablist");

    const s = useStore.getState();
    act(() => {
      s.startImageSessionTracking("a.jpg", Date.now() - 2000);
      s.closeSessionInterval();
      postHeldContributions();
    });

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        /Could not confirm these visits were recorded.*a\.jpg/,
      ),
    );
    expect(screen.queryByRole("heading", { name: "Notifications" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /a problem is ongoing/ })).toBeInTheDocument();
  });

  it("opens and closes over the project shell without remounting it", async () => {
    render(<App />);
    await screen.findByRole("tablist");
    const button = screen.getByRole("button", { name: "Notifications" });

    fireEvent.click(button);
    await PANE_FOCUS.heading();
    fireEvent.click(button);

    expect(button).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("heading", { name: "Notifications" })).not.toBeInTheDocument();
    expect(stateSocket.connect).toHaveBeenCalledTimes(1);
    expect(stateSocket.close).not.toHaveBeenCalled();
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
    expect(await screen.findByRole("button", { name: "Retry" })).toBeInTheDocument();
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

  it("lists a failed open that the selected tab hides the picker behind", async () => {
    vi.mocked(api.projects.open).mockRejectedValue(new Error("backend refused"));
    act(() => useStore.getState().setUser(""));
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);

    fireEvent.click(await screen.findByText(VALLEY_NAME));
    fireEvent.change(screen.getByLabelText("Your name"), { target: { value: "jordan" } });
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    expect(await screen.findByRole("tablist")).toBeInTheDocument();
    expect(screen.queryByText("Open a project")).not.toBeInTheDocument();
    await waitFor(() => expect(listedSentences()).toEqual(["backend refused"]));
  });

  it("lists a failed open whose project a snapshot adopted while the open was pending", async () => {
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

    expect(listedSentences()).toEqual(["selection failed"]);
  });
});

describe("App refused open", () => {
  beforeEach(() => vi.useFakeTimers({ shouldAdvanceTime: true }));
  afterEach(() => vi.useRealTimers());

  /** The chosen card's own line, the entry the picker shows in place. */
  const cardLine = () => screen.queryByTestId("entry-in-place");
  /** The notifications area, once it is open. */
  const area = () => within(screen.getByRole("region", { name: "Notifications" }));
  const toggleArea = () => fireEvent.click(screen.getByRole("button", { name: /^Notifications/ }));

  it("stays until dismissed, shows in place on its card, and Retry repeats the original open", async () => {
    const open = vi.mocked(api.projects.open);
    const select = vi.mocked(api.dataset.select);
    open.mockRejectedValueOnce(new Error("backend refused"));
    render(<App />);
    fireEvent.click(await screen.findByText(VALLEY_NAME));
    const chosen = useStore.getState().openChoice;
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    await waitFor(() => expect(cardLine()).toHaveTextContent("backend refused"));
    expect(useStore.getState().notifications.map((e) => e.context.project)).toEqual([
      VALLEY.record.id,
    ]);
    act(() => {
      vi.advanceTimersByTime(6001);
    });
    toggleArea();
    await PANE_FOCUS.heading();
    toggleArea();
    expect(cardLine()).toHaveTextContent("backend refused");
    expect(useStore.getState().notifications).toHaveLength(1);

    toggleArea();
    await PANE_FOCUS.heading();
    expect(area().getByRole("listitem")).toHaveTextContent(`Error, failure, ${VALLEY_NAME}`);
    fireEvent.click(area().getByRole("button", { name: /^Dismiss: backend refused/ }));
    expect(cardLine()).not.toBeInTheDocument();
    expect(useStore.getState().notifications).toEqual([]);

    open.mockRejectedValueOnce(new Error("backend refused again"));
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    await waitFor(() => expect(cardLine()).toHaveTextContent("backend refused again"));

    const otherDate = VALLEY.tree.captures.map((c) => c.name).find((d) => d !== chosen.date);
    fireEvent.change(screen.getByRole("combobox", { name: "Date" }), {
      target: { value: otherDate },
    });
    const retried = Promise.withResolvers<OpenAnswer>();
    open.mockReturnValueOnce(retried.promise);
    select.mockRejectedValueOnce(new Error("selection refused"));
    fireEvent.click(area().getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(open).toHaveBeenCalledTimes(3));
    expect(cardLine()).not.toBeInTheDocument();
    expect(useStore.getState().notifications).toHaveLength(1);
    expect(area().queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();

    await act(() => Promise.resolve(retried.resolve(VALLEY_OPENED)));
    await waitFor(() => expect(cardLine()).toHaveTextContent("selection refused"));
    expect(open.mock.calls[2]).toEqual(open.mock.calls[1]);
    expect(open.mock.calls[2][0]).toEqual({ id: VALLEY.record.id, user: "grower" });
    expect(select).toHaveBeenCalledWith({
      dataset_root: VALLEY_OPENED.project.path,
      date: chosen.date,
      subject: chosen.subject,
      bucket: chosen.bucket,
    });
    expect(useStore.getState().openChoice.date).toBe(otherDate);
    expect(listedSentences()).toEqual(["backend refused again", "selection refused"]);

    fireEvent.click(
      within(cardLine()!).getByRole("button", { name: /^Dismiss: selection refused/ }),
    );
    expect(cardLine()).not.toBeInTheDocument();
    expect(listedSentences()).toEqual(["backend refused again"]);
    expect(area().getAllByRole("listitem")).toHaveLength(1);
    expect(area().queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
  });

  it("counts a Retry refused the same way on the one entry, which the card shows in place", async () => {
    vi.mocked(api.projects.open).mockRejectedValue(new Error("backend refused"));
    render(<App />);
    fireEvent.click(await screen.findByText(VALLEY_NAME));
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    await waitFor(() => expect(cardLine()).toHaveTextContent("backend refused"));

    fireEvent.click(within(cardLine()!).getByRole("button", { name: "Retry" }));
    await waitFor(() => expect(cardLine()).toHaveTextContent("backend refused, 2 times"));

    toggleArea();
    await PANE_FOCUS.heading();
    expect(area().getByRole("listitem")).toHaveTextContent(/backend refused.*, 2 times/);
    expect(useStore.getState().notifications).toHaveLength(1);
  });

  it("leaves the card's line and the entry's Retry once the same card is chosen again, the entry staying listed", async () => {
    vi.mocked(api.projects.open).mockRejectedValue(new Error("backend refused"));
    render(<App />);
    fireEvent.click(await screen.findByText(VALLEY_NAME));
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    await waitFor(() => expect(cardLine()).toHaveTextContent("backend refused"));

    fireEvent.click(screen.getByText(VALLEY_NAME));

    expect(cardLine()).not.toBeInTheDocument();
    toggleArea();
    await PANE_FOCUS.heading();
    expect(area().getByRole("listitem")).toHaveTextContent("backend refused");
    expect(area().queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
  });
});

/** Answer what the Annotate tab reads on a capture's first image. */
function answerAnnotateReads() {
  vi.spyOn(api.annotate, "load").mockImplementation((path) => Promise.resolve(loadOf(path)));
  vi.spyOn(api.annotate, "save");
  vi.spyOn(api.images, "bands").mockResolvedValue(BAND_ANSWERS.plain);
  vi.spyOn(api.images, "viewReads").mockResolvedValue({ reads: [] });
  vi.spyOn(sessionsApi, "imageEvent").mockResolvedValue(SESSION_ANSWERS.landed);
}

describe("App picker on the project already open", () => {
  const NO_IMAGES = /has no images yet\. Ask the agent in the terminal to add images\./;

  it("offers no Open on the card of an open project holding no capture, and says what to do", async () => {
    vi.mocked(api.projects.list)
      .mockResolvedValueOnce(PROJECT_FIXTURES.picker)
      .mockResolvedValue(PROJECT_FIXTURES.picker_empty_open);
    vi.mocked(api.projects.open).mockResolvedValue(openedAs(EMPTY));
    render(<App />);

    fireEvent.click(await screen.findByText(EMPTY.record.display_name));
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    await waitFor(() => expect(useStore.getState().openProject?.id).toBe(EMPTY.record.id));

    // The picker's own poll is 2 s away, past waitFor's timeout: the card learns it is open
    // from the read the Open click chains on the open.
    const open = screen.getByRole("button", { name: "Open" });
    await waitFor(() => expect(open).toBeDisabled());
    expect(open).toHaveAccessibleDescription(NO_IMAGES);
    expect(screen.getByText(NO_IMAGES)).toBeInTheDocument();
  });

  it("keeps Open on the card of an open project holding captures after Switch project", async () => {
    answerAnnotateReads();
    vi.mocked(api.projects.list)
      .mockResolvedValueOnce(PROJECT_FIXTURES.picker)
      .mockResolvedValue(PROJECT_FIXTURES.picker_open);
    vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
    vi.mocked(api.dataset.select).mockResolvedValue(SUBJECT_SELECT_ANSWER);
    render(<App />);

    fireEvent.click(await screen.findByText(VALLEY_NAME));
    fireEvent.click(screen.getByRole("button", { name: "Open" }));
    await screen.findByTestId("canvas-stage");
    fireEvent.click(screen.getByRole("button", { name: "Switch project" }));
    await screen.findByText("Open a project");

    const open = await screen.findByRole("button", { name: "Open" });
    expect(await screen.findByText("open")).toBeInTheDocument();
    expect(open).toBeEnabled();
    expect(open).not.toHaveAccessibleDescription(NO_IMAGES);
    expect(screen.queryByText(NO_IMAGES)).not.toBeInTheDocument();
  });
});

describe("App visit attribution across a change of person", () => {
  const answer = SUBJECT_SELECT_ANSWER;
  const FIRST_IMAGE = answer.selection.images[0].name;
  it("attributes the outgoing visit to the first person and project, the next to the second", async () => {
    answerAnnotateReads();
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
