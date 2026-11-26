import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import { inferenceApi } from "@/api/inference";
import { sessionsApi } from "@/api/sessions";
import { subjectsApi } from "@/api/subjects";
import { trainingApi } from "@/api/training";
import type {
  InferenceJobs,
  OpenAnswer,
  Panel,
  PanelEnvelope,
  TrainingListing,
} from "@/api/types.generated";
import App from "@/App";
import { useActiveTabSync } from "@/hooks/useActiveTabSync";
import { INFERENCE_REFRESH_MS } from "@/hooks/useJobListings";
import { postHeldContributions } from "@/lib/sessionLifecycle";
import { useStore } from "@/store";
import { RUN_REFRESH_MS } from "@/tabs/trainingMetrics";
import { loadOf } from "@/test/annotate";
import { BAND_ANSWERS } from "@/test/bands";
import { measureWindow, PANE_FOCUS } from "@/test/dock";
import { INFERENCE_ANSWERS } from "@/test/inference";
import { LABEL_ANSWERS } from "@/test/labels";
import { announcements } from "@/test/notifications";
import { PANEL_EVENTS } from "@/test/panelEvents";
import { openedAs, PICKER, PROJECT_FIXTURES } from "@/test/projectRecords";
import { thrown } from "@/test/refusals";
import { SESSION_ANSWERS, sessionOf } from "@/test/sessions";
import { changeSelection, stateEnvelope, SUBJECT_SELECT_ANSWER } from "@/test/store";
import {
  CANCEL,
  LISTING_BEFORE_FAILURE,
  RUNS,
  STALLED_LISTING,
  SWEEPS,
  TRAINING_LISTING,
  TRAINING_REFUSALS,
} from "@/test/training";

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
// The training runs and the inference jobs are listed whether or not a project is open; each
// test sets what they answer.
vi.mock("@/api/training", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/training")>();
  return { ...actual, trainingApi: { ...actual.trainingApi, listRuns: vi.fn() } };
});
vi.mock("@/api/inference", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/api/inference")>();
  return { ...actual, inferenceApi: { ...actual.inferenceApi, listJobs: vi.fn() } };
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
  vi.mocked(trainingApi.listRuns).mockReset().mockResolvedValue({ runs: [], sweeps: [] });
  vi.mocked(inferenceApi.listJobs).mockReset().mockResolvedValue({ jobs: [] });
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

describe("App job listings", () => {
  const NO_ANSWER = () => new Promise<never>(() => {});
  const listings = [
    { tab: "training", read: () => vi.mocked(trainingApi.listRuns) },
    { tab: "inference", read: () => vi.mocked(inferenceApi.listJobs) },
  ] as const;

  afterEach(() => vi.useRealTimers());

  it("lists the training runs and the inference jobs with another tab active, which each tab then shows", async () => {
    vi.mocked(trainingApi.listRuns)
      .mockResolvedValueOnce(TRAINING_LISTING)
      .mockImplementation(NO_ANSWER);
    vi.mocked(inferenceApi.listJobs)
      .mockResolvedValueOnce(INFERENCE_ANSWERS.running)
      .mockImplementation(NO_ANSWER);
    useStore.setState({ openProject: VALLEY_OPENED.project });
    act(() => useStore.getState().setActiveTab("meta"));
    render(<App />);
    await screen.findByRole("tabpanel", { name: /meta/i });

    await waitFor(() => {
      expect(trainingApi.listRuns).toHaveBeenCalled();
      expect(inferenceApi.listJobs).toHaveBeenCalled();
    });

    act(() => useStore.getState().setActiveTab("training"));
    expect(await screen.findByText(RUNS.unlaunched.experiment_id)).toBeInTheDocument();
    act(() => useStore.getState().setActiveTab("inference"));
    expect(await screen.findByText(INFERENCE_ANSWERS.launched.job_id)).toBeInTheDocument();
  });

  it.each(listings)(
    "starts no second read of the $tab listing while its first is held open for a minute",
    async ({ tab, read }) => {
      read().mockImplementation(NO_ANSWER);
      useStore.setState({ openProject: VALLEY_OPENED.project });
      act(() => useStore.getState().setActiveTab(tab));
      vi.useFakeTimers();
      render(<App />);
      await vi.waitFor(() => expect(read()).toHaveBeenCalled());

      await act(async () => {
        await vi.advanceTimersByTimeAsync(60_000);
      });

      expect(read()).toHaveBeenCalledTimes(1);
    },
  );

  it.each([
    {
      tab: "training",
      read: () => vi.mocked(trainingApi.listRuns),
      refusal: TRAINING_REFUSALS.unopened,
      failed: "Could not load training runs",
    },
    {
      tab: "inference",
      read: () => vi.mocked(inferenceApi.listJobs),
      refusal: INFERENCE_ANSWERS.unopened,
      failed: "Could not load inference jobs",
    },
  ] as const)(
    "with no project open reads the $tab listing once across a minute, and once more on its Retry",
    async ({ tab, read, refusal, failed }) => {
      const refused = await thrown(refusal);
      read().mockRejectedValue(refused);
      const message = `${failed}: ${refusal.body.detail}`;
      act(() => useStore.getState().setActiveTab(tab));
      vi.useFakeTimers();
      render(<App />);
      await vi.waitFor(() => expect(screen.getByText(message)).toBeInTheDocument());

      await act(async () => {
        await vi.advanceTimersByTimeAsync(60_000);
      });
      expect(read()).toHaveBeenCalledTimes(1);

      fireEvent.click(within(screen.getByText(message)).getByRole("button", { name: "Retry" }));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(60_000);
      });
      expect(read()).toHaveBeenCalledTimes(2);
      expect(screen.getByText(message)).toBeInTheDocument();
    },
  );

  it("reads the runs once more as soon as the read in flight when a Cancel lands answers", async () => {
    const held = Promise.withResolvers<TrainingListing>();
    vi.mocked(trainingApi.listRuns)
      .mockResolvedValueOnce(TRAINING_LISTING)
      .mockReturnValueOnce(held.promise)
      .mockResolvedValue({ runs: [], sweeps: [] });
    vi.spyOn(trainingApi, "cancel").mockResolvedValue(CANCEL);
    const id = RUNS.unlaunched.experiment_id;
    useStore.setState({ openProject: VALLEY_OPENED.project });
    act(() => useStore.getState().setActiveTab("training"));
    vi.useFakeTimers();
    render(<App />);
    await vi.waitFor(() => expect(screen.getByText(id)).toBeInTheDocument());
    await act(async () => {
      await vi.advanceTimersByTimeAsync(RUN_REFRESH_MS);
    });
    expect(trainingApi.listRuns).toHaveBeenCalledTimes(2);

    await act(async () => {
      fireEvent.click(screen.getByRole("button", { name: `Cancel ${id}` }));
      await Promise.resolve();
    });
    expect(trainingApi.cancel).toHaveBeenCalledTimes(1);
    expect(trainingApi.listRuns).toHaveBeenCalledTimes(2);

    await act(() => Promise.resolve(held.resolve(TRAINING_LISTING)));

    expect(trainingApi.listRuns).toHaveBeenCalledTimes(3);
    expect(screen.getByText("No runs.")).toBeInTheDocument();
  });

  it("never shows the runs a read answers for the project open before another opened", async () => {
    const held = Promise.withResolvers<TrainingListing>();
    vi.mocked(trainingApi.listRuns)
      .mockReturnValueOnce(held.promise)
      .mockResolvedValue({ runs: [], sweeps: [] });
    useStore.setState({ openProject: VALLEY_OPENED.project });
    act(() => useStore.getState().setActiveTab("training"));
    render(<App />);
    await waitFor(() => expect(trainingApi.listRuns).toHaveBeenCalledTimes(1));

    act(() =>
      useStore
        .getState()
        .mergeSnapshot(stateEnvelope({ version: 1, project: openedAs(RIDGE).project })),
    );
    act(() => useStore.getState().setActiveTab("training"));
    await waitFor(() => expect(trainingApi.listRuns).toHaveBeenCalledTimes(2));
    expect(await screen.findByText("No runs.")).toBeInTheDocument();
    await act(() => Promise.resolve(held.resolve(TRAINING_LISTING)));

    expect(screen.getByText("No runs.")).toBeInTheDocument();
    expect(screen.queryByText(RUNS.unlaunched.experiment_id)).not.toBeInTheDocument();
  });

  it("keeps the runs a read answered beside the failure of the read after it", async () => {
    vi.mocked(trainingApi.listRuns)
      .mockResolvedValueOnce(TRAINING_LISTING)
      .mockRejectedValue(new TypeError("Failed to fetch"));
    useStore.setState({ openProject: VALLEY_OPENED.project });
    act(() => useStore.getState().setActiveTab("training"));
    vi.useFakeTimers();
    render(<App />);
    await vi.waitFor(() =>
      expect(screen.getByText(RUNS.unlaunched.experiment_id)).toBeInTheDocument(),
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(RUN_REFRESH_MS);
    });

    expect(screen.getByText("Could not load training runs: Failed to fetch")).toBeInTheDocument();
    expect(screen.getByText(RUNS.unlaunched.experiment_id)).toBeInTheDocument();
  });

  it("shows the listing a read answers after a failed one, without a Retry", async () => {
    vi.mocked(trainingApi.listRuns)
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValue(TRAINING_LISTING);
    useStore.setState({ openProject: VALLEY_OPENED.project });
    act(() => useStore.getState().setActiveTab("training"));
    vi.useFakeTimers();
    render(<App />);
    await vi.waitFor(() =>
      expect(screen.getByText("Could not load training runs: Failed to fetch")).toBeInTheDocument(),
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(60_000);
    });

    expect(screen.getByText(RUNS.unlaunched.experiment_id)).toBeInTheDocument();
    expect(screen.queryByText(/Could not load training runs/)).not.toBeInTheDocument();
  });

  it("shows none of one project's runs or jobs once another project opens", async () => {
    vi.mocked(trainingApi.listRuns)
      .mockResolvedValueOnce(TRAINING_LISTING)
      .mockImplementation(NO_ANSWER);
    vi.mocked(inferenceApi.listJobs)
      .mockResolvedValueOnce(INFERENCE_ANSWERS.running)
      .mockImplementation(NO_ANSWER);
    useStore.setState({ openProject: VALLEY_OPENED.project });
    act(() => useStore.getState().setActiveTab("training"));
    render(<App />);
    expect(await screen.findByText(RUNS.unlaunched.experiment_id)).toBeInTheDocument();
    act(() => useStore.getState().setActiveTab("inference"));
    expect(await screen.findByText(INFERENCE_ANSWERS.launched.job_id)).toBeInTheDocument();

    act(() =>
      useStore
        .getState()
        .mergeSnapshot(stateEnvelope({ version: 1, project: openedAs(RIDGE).project })),
    );
    act(() => useStore.getState().setActiveTab("training"));
    expect(await screen.findByText("No runs.")).toBeInTheDocument();
    expect(screen.queryByText(RUNS.unlaunched.experiment_id)).not.toBeInTheDocument();
    act(() => useStore.getState().setActiveTab("inference"));
    expect(await screen.findByText("No jobs.")).toBeInTheDocument();
    expect(screen.queryByText(INFERENCE_ANSWERS.launched.job_id)).not.toBeInTheDocument();
  });
});

describe("App job entries", () => {
  const NO_ANSWER = () => new Promise<never>(() => {});
  const streamed = RUNS.streamed.experiment_id;
  const streamedBefore = LISTING_BEFORE_FAILURE.runs.find((r) => r.experiment_id === streamed);
  const RUN_RUNNING = `Training run ${streamed}: running, epoch ${streamedBefore?.current_epoch}`;
  const RUN_FAILED = `Training run ${streamed}: failed. ${RUNS.streamed.status_error}`;
  const [failing] = INFERENCE_ANSWERS.failing_started.jobs;
  const failingFailed = INFERENCE_ANSWERS.held_started.jobs.find(
    (j) => j.job_id === failing.job_id,
  );
  const FAILING_RUNNING = `Inference job ${failing.job_id}: running`;
  const FAILING_FAILED = `Inference job ${failing.job_id}: failed. ${
    failingFailed?.state.status === "failed" ? failingFailed.state.error : ""
  }`;
  const held = INFERENCE_ANSWERS.launched.job_id;
  const HELD_RUNNING = `Inference job ${held}: running`;

  afterEach(() => vi.useRealTimers());

  /** Open Valley farm on the Meta tab, its listings answering `training` and then `inference` in
   *  turn and nothing after, and wait for each listing's first answer. */
  async function watchFromMeta(training: TrainingListing[], inference: InferenceJobs[]) {
    const answers = <T,>(read: { mockResolvedValueOnce: (v: T) => unknown }, each: T[]) =>
      each.forEach((answer) => read.mockResolvedValueOnce(answer));
    if (training.length)
      answers(vi.mocked(trainingApi.listRuns).mockImplementation(NO_ANSWER), training);
    if (inference.length)
      answers(vi.mocked(inferenceApi.listJobs).mockImplementation(NO_ANSWER), inference);
    useStore.setState({ openProject: VALLEY_OPENED.project });
    act(() => useStore.getState().setActiveTab("meta"));
    vi.useFakeTimers();
    render(<App />);
    await vi.waitFor(() => {
      expect(trainingApi.listRuns).toHaveBeenCalledTimes(1);
      expect(inferenceApi.listJobs).toHaveBeenCalledTimes(1);
    });
    await settle();
  }

  /** Let every answer already given land. */
  const settle = () => act(() => vi.advanceTimersByTimeAsync(0));
  /** Let the next read of a listing read every `ms` answer. */
  const next = (ms: number) => act(() => vi.advanceTimersByTimeAsync(ms));
  const alert = () => screen.getByRole("alert");
  const notificationsButton = () => screen.getByRole("button", { name: /^Notifications/ });
  /** Open the notifications area, which counts every entry listed as seen, and close it. */
  function lookAtArea() {
    fireEvent.click(notificationsButton());
    fireEvent.click(notificationsButton());
  }
  const entryOf = (sentence: string) =>
    useStore.getState().notifications.find((e) => e.sentence === sentence);

  it("tells a run the page did not launch from its first listing as running, and its end in the same entry, spoken", async () => {
    await watchFromMeta([LISTING_BEFORE_FAILURE, TRAINING_LISTING], []);

    expect(listedSentences()).toEqual(
      expect.arrayContaining([
        RUN_RUNNING,
        `Training run ${RUNS.agent.experiment_id}: running`,
        `Sweep ${SWEEPS.searched.sweep_id}: running`,
        `Sweep ${SWEEPS.relaunched.sweep_id}: running`,
      ]),
    );
    const listed = listedSentences().length;

    await next(RUN_REFRESH_MS);

    expect(listedSentences()).toHaveLength(listed);
    expect(listedSentences()).not.toContain(RUN_RUNNING);
    expect(entryOf(RUN_FAILED)).toEqual(
      expect.objectContaining({
        kind: "job",
        level: "error",
        context: { project: VALLEY.record.id, target: streamed },
      }),
    );
    expect(alert()).toHaveTextContent(RUN_FAILED);
  });

  it("tells an inference job from first sight to its end in one entry, its progress unannounced and its end counted as new", async () => {
    const { held_started, running, canceled } = INFERENCE_ANSWERS;
    await watchFromMeta([], [INFERENCE_ANSWERS.failing_started, held_started, running, canceled]);
    expect(listedSentences()).toEqual([FAILING_RUNNING]);
    const alerts = announcements(alert());
    const status = within(alert().parentElement!).getByRole("status");
    const statuses = announcements(status);

    await next(INFERENCE_REFRESH_MS);
    expect(listedSentences()).toEqual([FAILING_FAILED, HELD_RUNNING]);
    expect(alert()).toHaveTextContent(FAILING_FAILED);
    expect(alerts).toEqual([FAILING_FAILED]);
    expect(statuses).toEqual([HELD_RUNNING]);

    await next(INFERENCE_REFRESH_MS);
    const progressed = `${HELD_RUNNING}, 1 of 9 images`;
    expect(listedSentences()).toEqual([FAILING_FAILED, progressed]);
    expect(alert()).toHaveTextContent(FAILING_FAILED);
    expect(alerts).toEqual([FAILING_FAILED]);
    expect(statuses).toEqual([HELD_RUNNING]);

    lookAtArea();
    expect(notificationsButton()).toHaveAccessibleName("Notifications");
    await next(INFERENCE_REFRESH_MS);
    const ended = `Inference job ${held}: canceled`;
    expect(listedSentences()).toEqual([FAILING_FAILED, ended]);
    expect(statuses).toEqual([HELD_RUNNING, ended]);
    expect(status).toHaveTextContent(ended);
    expect(status).not.toHaveTextContent(HELD_RUNNING);
    expect(notificationsButton()).toHaveAccessibleName("Notifications, 1 new");
  });

  it("adds nothing when the same end is answered again", async () => {
    await watchFromMeta(
      [LISTING_BEFORE_FAILURE, TRAINING_LISTING, TRAINING_LISTING, TRAINING_LISTING],
      [],
    );
    await next(RUN_REFRESH_MS);
    const told = useStore.getState().notifications;
    expect(told.map((e) => e.sentence)).toContain(RUN_FAILED);

    await next(RUN_REFRESH_MS);
    await next(RUN_REFRESH_MS);

    expect(useStore.getState().notifications).toEqual(told);
    expect(alert()).toHaveTextContent(RUN_FAILED);
  });

  it("tells the end of a job that started and ended between two reads", async () => {
    await watchFromMeta([], [{ jobs: [] }, INFERENCE_ANSWERS.held_started]);
    expect(listedSentences()).toEqual([]);

    await next(INFERENCE_REFRESH_MS);

    expect(listedSentences()).toContain(FAILING_FAILED);
  });

  it("raises nothing for a dismissed running job until its end", async () => {
    const { held_started, running, canceled } = INFERENCE_ANSWERS;
    await watchFromMeta([], [held_started, running, canceled]);
    expect(listedSentences()).toEqual([HELD_RUNNING]);

    fireEvent.click(notificationsButton());
    const area = within(screen.getByRole("region", { name: "Notifications" }));
    fireEvent.click(
      area.getByRole("button", { name: new RegExp(`^Dismiss: Inference job ${held}`) }),
    );
    expect(listedSentences()).toEqual([]);

    await next(INFERENCE_REFRESH_MS);
    expect(listedSentences()).toEqual([]);

    await next(INFERENCE_REFRESH_MS);
    expect(listedSentences()).toEqual([`Inference job ${held}: canceled`]);
  });

  it("tells a watched job an answer no longer lists as a warning claiming no cause", async () => {
    await watchFromMeta([], [INFERENCE_ANSWERS.running, { jobs: [] }]);
    expect(listedSentences()).toEqual([`${HELD_RUNNING}, 1 of 9 images`]);

    await next(INFERENCE_REFRESH_MS);

    const gone = `Inference job ${held}: no longer listed; its end was not seen here`;
    expect(listedSentences()).toEqual([gone]);
    fireEvent.click(notificationsButton());
    const area = within(screen.getByRole("region", { name: "Notifications" }));
    expect(area.getByRole("listitem")).toHaveTextContent(`${gone}Warning, job, ${held}`);
  });

  it("tells a run whose sign of life went stale as interrupted, counted as new, and its later end", async () => {
    await watchFromMeta([LISTING_BEFORE_FAILURE, STALLED_LISTING, TRAINING_LISTING], []);
    expect(listedSentences()).toContain(RUN_RUNNING);

    lookAtArea();
    await next(RUN_REFRESH_MS);
    const stalled = `Training run ${streamed}: interrupted`;
    expect(entryOf(stalled)).toEqual(expect.objectContaining({ level: "warning" }));
    expect(listedSentences()).not.toContain(RUN_RUNNING);
    expect(notificationsButton()).toHaveAccessibleName("Notifications, 1 new");

    await next(RUN_REFRESH_MS);
    expect(listedSentences()).not.toContain(stalled);
    expect(alert()).toHaveTextContent(RUN_FAILED);
  });

  it("tells a run first seen interrupted nothing, and tells it once a later answer lists it running", async () => {
    await watchFromMeta([STALLED_LISTING, LISTING_BEFORE_FAILURE], []);
    expect(useStore.getState().notifications.map((e) => e.context.target)).not.toContain(streamed);

    await next(RUN_REFRESH_MS);

    expect(listedSentences()).toContain(RUN_RUNNING);
  });

  it("tells a run a later answer first lists interrupted nothing, nor its end after", async () => {
    await watchFromMeta([{ runs: [], sweeps: [] }, STALLED_LISTING, TRAINING_LISTING], []);
    const aboutStreamed = () =>
      useStore.getState().notifications.filter((e) => e.context.target === streamed);

    await next(RUN_REFRESH_MS);
    expect(listedSentences()).toContain(`Training run ${RUNS.agent.experiment_id}: running`);
    expect(aboutStreamed()).toEqual([]);

    await next(RUN_REFRESH_MS);
    expect(aboutStreamed()).toEqual([]);
  });

  it("tells nothing when another project opens and its listings answer empty, and what changed once the first opens again", async () => {
    const valley = [LISTING_BEFORE_FAILURE, TRAINING_LISTING];
    const inValley = () => useStore.getState().openProject?.id === VALLEY.record.id;
    vi.mocked(trainingApi.listRuns).mockImplementation(() => {
      if (!inValley()) return Promise.resolve({ runs: [], sweeps: [] });
      const answer = valley.shift();
      return answer ? Promise.resolve(answer) : NO_ANSWER();
    });
    vi.mocked(inferenceApi.listJobs).mockResolvedValue({ jobs: [] });
    await watchFromMeta([], []);
    const told = useStore.getState().notifications;
    expect(told.map((e) => e.sentence)).toContain(RUN_RUNNING);

    act(() =>
      useStore
        .getState()
        .mergeSnapshot(stateEnvelope({ version: 1, project: openedAs(RIDGE).project })),
    );
    await settle();
    expect(trainingApi.listRuns).toHaveBeenCalledTimes(2);
    expect(useStore.getState().notifications).toEqual(told);

    act(() =>
      useStore
        .getState()
        .mergeSnapshot(stateEnvelope({ version: 2, project: VALLEY_OPENED.project })),
    );
    await settle();
    expect(listedSentences()).not.toContain(RUN_RUNNING);
    expect(alert()).toHaveTextContent(RUN_FAILED);
  });

  it("tells nothing from an answer read for a project no longer open", async () => {
    const held = Promise.withResolvers<TrainingListing>();
    vi.mocked(trainingApi.listRuns).mockReturnValueOnce(held.promise);
    await watchFromMeta([], []);

    act(() =>
      useStore
        .getState()
        .mergeSnapshot(stateEnvelope({ version: 1, project: openedAs(RIDGE).project })),
    );
    await settle();
    await act(() => Promise.resolve(held.resolve(LISTING_BEFORE_FAILURE)));
    await settle();

    expect(useStore.getState().notifications).toEqual([]);
  });

  it("tells nothing from a failed read, the jobs an answer left watched reading as they did", async () => {
    vi.mocked(trainingApi.listRuns)
      .mockResolvedValueOnce(LISTING_BEFORE_FAILURE)
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockImplementation(NO_ANSWER);
    await watchFromMeta([], []);
    const told = useStore.getState().notifications;
    expect(told.map((e) => e.sentence)).toContain(RUN_RUNNING);

    await next(RUN_REFRESH_MS);

    expect(trainingApi.listRuns).toHaveBeenCalledTimes(2);
    expect(useStore.getState().notifications).toEqual(told);
  });

  it("counts a job told again as seen while the area is open over it", async () => {
    await watchFromMeta([LISTING_BEFORE_FAILURE, TRAINING_LISTING], []);
    fireEvent.click(notificationsButton());
    await settle();
    expect(notificationsButton()).toHaveAccessibleName("Notifications");

    await next(RUN_REFRESH_MS);

    expect(listedSentences()).toContain(RUN_FAILED);
    expect(notificationsButton()).toHaveAccessibleName("Notifications");
  });

  it("tells nothing of a run a first listing already lists ended", async () => {
    await watchFromMeta([TRAINING_LISTING], []);

    expect(useStore.getState().notifications.map((e) => e.context.target)).not.toContain(streamed);
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
