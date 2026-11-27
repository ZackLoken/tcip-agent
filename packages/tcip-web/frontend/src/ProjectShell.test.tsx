import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

import type { LoadedLabels } from "@/api/client";
import { ROUTES } from "@/api/routes";
import type { PanelEnvelope } from "@/api/types.generated";
import { dismissLabel } from "@/lib/dismissLabel";
import { ANNOTATE_ANSWERS, LANDED, loadOf } from "@/test/annotate";
import { LABEL_ANSWERS } from "@/test/labels";
import {
  DISMISSALS,
  PANEL_EVENTS,
  PANEL_EVENTS_AFTER_DISMISSAL,
  PANEL_EVENTS_AT_CONNECT,
} from "@/test/panelEvents";
import { openedAs, PICKER } from "@/test/projectRecords";
import { answeredWith, thrown } from "@/test/refusals";
import {
  NAV_ANSWER,
  RAW_SELECTION,
  SELECTED_SUBJECT,
  SUBJECT_SELECT_ANSWER,
  TEST_PROJECT,
} from "@/test/store";
import { FakeWebSocket } from "@/test/webSocket";

// The shell mounts the Annotate tab, which pulls in Konva, and jsdom has no canvas backend.
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

type Page = {
  ProjectShell: typeof import("@/ProjectShell").ProjectShell;
  NotificationsPane: typeof import("@/components/Notifications").NotificationsPane;
  useStore: typeof import("@/store").useStore;
  load: MockInstance<typeof import("@/api/client").api.annotate.load>;
  save: MockInstance<typeof import("@/api/client").api.annotate.save>;
  select: MockInstance<typeof import("@/api/client").api.dataset.select>;
};

/** The page each test runs in. */
let page: Page;

/** A request the backend never answers. */
const pending = () => new Promise<never>(() => {});

/** Load every module anew, as a reload of the page does, and answer what the shell and its
 *  Annotate tab read on the produced capture's first image, and a save with the produced save of a
 *  box on it; the Meta tab's reads are left unanswered. */
async function loadPage(): Promise<Page> {
  vi.resetModules();
  const [
    { ProjectShell },
    { NotificationsPane },
    { useStore },
    { api },
    { answerAnnotateMount, LANDED: landed },
    subjects,
    training,
    inference,
    { metaApi },
    { sessionsApi },
  ] = await Promise.all([
    import("@/ProjectShell"),
    import("@/components/Notifications"),
    import("@/store"),
    import("@/api/client"),
    import("@/test/annotate"),
    import("@/api/subjects"),
    import("@/api/training"),
    import("@/api/inference"),
    import("@/api/meta"),
    import("@/api/sessions"),
  ]);
  const load = answerAnnotateMount();
  const save = vi.spyOn(api.annotate, "save").mockResolvedValue(landed.box);
  const select = vi.spyOn(api.dataset, "select").mockRejectedValue(new Error("selection refused"));
  vi.spyOn(api.dataset, "nav").mockResolvedValue(NAV_ANSWER);
  vi.spyOn(subjects.subjectsApi, "load").mockResolvedValue(LABEL_ANSWERS.registry_loaded);
  vi.spyOn(training.trainingApi, "listRuns").mockResolvedValue({ runs: [], sweeps: [] });
  vi.spyOn(inference.inferenceApi, "listJobs").mockResolvedValue({ jobs: [] });
  vi.spyOn(metaApi, "reports").mockImplementation(pending);
  vi.spyOn(metaApi, "retrospectives").mockImplementation(pending);
  vi.spyOn(sessionsApi, "load").mockImplementation(pending);
  return { ProjectShell, NotificationsPane, useStore, load, save, select };
}

/** The last socket the page opened for `panel`'s events. */
function panelSocket(panel: string): FakeWebSocket {
  const socket = FakeWebSocket.instances.findLast((s) => s.url.endsWith(`/ws/panel/${panel}`));
  if (!socket) throw new Error(`the page opened no socket for the ${panel} panel`);
  return socket;
}

/** Deliver `envelope` on its panel's socket, as the backend sends it. */
function deliver(envelope: PanelEnvelope) {
  act(() => panelSocket(envelope.panel).message(JSON.stringify(envelope)));
}

/** Drop `panel`'s socket and let the page connect it again once its backoff has run. */
function reconnect(panel: string) {
  const before = FakeWebSocket.instances.length;
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  try {
    act(() => panelSocket(panel).drop());
    act(() => {
      vi.advanceTimersToNextTimer();
    });
  } finally {
    vi.useRealTimers();
  }
  expect(FakeWebSocket.instances).toHaveLength(before + 1);
  act(() => panelSocket(panel).open());
}

/** The notice entries the notifications area lists. */
const notices = () => page.useStore.getState().notifications.filter((e) => e.kind === "notice");

/** Mount the shell beside the open notifications area with `project` open on `selection`, the
 *  produced capture by default, the Annotate tab active, and wait for the labels of the image in
 *  view to load. */
async function mountOnCapture(project = TEST_PROJECT, selection = RAW_SELECTION) {
  const { ProjectShell, NotificationsPane, useStore, load } = page;
  useStore.getState().setUser("grower");
  useStore.setState((s) => ({
    openProject: project,
    gui: { ...s.gui, active_tab: "annotate", dataset: selection },
  }));
  render(
    <>
      <ProjectShell />
      <NotificationsPane />
    </>,
  );
  await waitFor(() => expect(load).toHaveBeenCalledTimes(1));
  await act(async () => {});
  for (const panel of ["annotate", "results", "app"]) act(() => panelSocket(panel).open());
}

/** The open notifications area. */
const area = () => within(screen.getByRole("region", { name: "Notifications" }));

/** The entries the open area lists, newest first. */
const listItems = () => area().queryAllByRole("listitem");

/** Expect `item` to state no count of times raised. */
const expectRaisedOnce = (item: HTMLElement) => expect(item).not.toHaveTextContent(/ times/);

/** A request as the page sent it: its path and its body. */
type Sent = { path: string; body: unknown };

/** Answer each POST the page sends to a path ending in `route` with the next of `answers`, and
 *  every other request as the page's own fetch does; return those POSTs, in order. */
function answerPosts(route: string, ...answers: (() => Promise<Response>)[]): Sent[] {
  const pageFetch = globalThis.fetch;
  const sent: Sent[] = [];
  vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
    const path = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (!path.endsWith(route) || init?.method !== "POST") return pageFetch(input, init);
    const body = init.body;
    sent.push({ path, body: typeof body === "string" ? (JSON.parse(body) as unknown) : body });
    const answer = answers.shift();
    if (!answer) throw new Error(`no answer left for ${path}`);
    return answer();
  });
  return sent;
}

/** Answer each request the page sends to a dismissal route with the next of `answers`, and
 *  every other request as the page's own fetch does; return the dismissal requests, in order. */
const answerDismissals = (...answers: (() => Promise<Response>)[]) =>
  answerPosts("/dismissals", ...answers);

/** Send the page's label saves to the fetch layer, answering each with the next of `answers`;
 *  return the save requests, in order. */
function answerSaves(...answers: (() => Promise<Response>)[]): Sent[] {
  page.save.mockRestore();
  return answerPosts(ROUTES.postAnnotateLabels, ...answers);
}

/** An answer of `status` sending `body`. */
const answering = (status: number, body: unknown) => () =>
  Promise.resolve(answeredWith(status, body));

/** No answer at all, as a request the network drops rejects. */
const unanswered = () => Promise.reject(new TypeError("Failed to fetch"));

/** Press the dismiss control of the listed entry saying `text`. */
function dismissEntry(text: string) {
  fireEvent.click(
    area().getByRole("button", { name: (name) => name.startsWith(dismissLabel(text)) }),
  );
}

/** Let every answer already given land. */
const settle = () => act(async () => {});

/** Make `active` the active tab, as the tab bar does, and let what it reads land. */
async function switchTo(active: "meta" | "annotate") {
  act(() => page.useStore.getState().setActiveTab(active));
  await settle();
}

/** Add a box to the canvas, leaving it with unsaved edits. */
function addBox() {
  act(() =>
    page.useStore.getState().add("boxes", {
      x1: 10,
      y1: 10,
      x2: 50,
      y2: 50,
      subject: SELECTED_SUBJECT,
      attributes: {},
    }),
  );
  expect(page.useStore.getState().canvas.dirty).toBe(true);
}

/** Hold the page's next load of an image unanswered; returns the answer to give it later. */
function holdNextLoad(): (labels: LoadedLabels) => void {
  let answer: ((labels: LoadedLabels) => void) | undefined;
  page.load.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        answer = resolve;
      }),
  );
  return (labels) => {
    if (!answer) throw new Error("the page sent no load the test held");
    const given = answer;
    act(() => given(labels));
  };
}

/** The status bar's mark of unsaved work on the canvas. */
const unsaved = () => screen.queryByText("Unsaved changes");

/** The image stepper's readout of where the image in view stands among the capture's images. */
const imagePosition = () => screen.getByRole("textbox", { name: "Image position" });

/** The entries the open area lists whose sentence holds `text`. */
const listing = (text: string) =>
  listItems().filter((item) => within(item).queryByText(text, { exact: false }));

/** The entries the open area lists as failures, by the kind each names. */
const failureEntries = () =>
  listItems().filter((item) => within(item).queryByText(/^\w+, failure, /));

/** The line the canvas shows in place for the image in view. */
const inPlace = () => screen.getByTestId("entry-in-place");

/** The save route's refusal of a document changed since it was loaded, as the route answers it. */
const refusedAsChanged = answering(
  LABEL_ANSWERS.version_conflict.status,
  LABEL_ANSWERS.version_conflict.body,
);

/** The sentence that refusal states. */
const CONFLICT_SENTENCE = LABEL_ANSWERS.version_conflict.body.detail.message;

/** The line a refused save of a document changed since it was loaded shows under the canvas. */
const RELOAD_CHOICE = `${CONFLICT_SENTENCE}. Reload to load the latest (discards your unsaved edits), or keep editing.`;

/** The line a save that failed for any other reason shows under the canvas. */
const SAVE_AGAIN =
  /^Could not save annotations \(.+\)\. Your edits are kept in the editor; press Save to retry\.$/;

beforeEach(async () => {
  FakeWebSocket.instances = [];
  vi.stubGlobal("WebSocket", FakeWebSocket);
  page = await loadPage();
});

afterEach(cleanup);

describe("ProjectShell agent activity", () => {
  const { labels_written: written, labels_written_second: second } = PANEL_EVENTS;

  it("lists two live saves of different images as one notice of the agent's activity, counted twice", async () => {
    await mountOnCapture();

    deliver(written);
    deliver(second);

    expect(notices()).toEqual([
      expect.objectContaining({
        level: "info",
        sentence: "A process: labels written",
        count: 2,
        context: { project: written.project_id, target: "Annotate" },
      }),
    ]);
  });

  it("names the client a save declared, without its version", async () => {
    const declared = PANEL_EVENTS.labels_written_declared;
    const { agent_client_name: name, agent_client_version: version } = declared.identity;
    expect(version).toBeTruthy();
    await mountOnCapture();

    deliver(declared);

    expect(notices().map((e) => e.sentence)).toEqual([`${name}: labels written`]);
  });

  it("reloads a clean canvas for a save sent at connect, raising no notice, and does neither when it is sent again", async () => {
    const atConnect = PANEL_EVENTS_AT_CONNECT.labels_written;
    expect(atConnect.event.image_path).toBe(RAW_SELECTION.images[0].path);
    await mountOnCapture();

    deliver(atConnect);
    await waitFor(() => expect(page.load).toHaveBeenCalledTimes(2));
    expect(page.load).toHaveBeenLastCalledWith(atConnect.event.image_path);
    expect(notices()).toEqual([]);

    reconnect("annotate");
    deliver(atConnect);
    await act(async () => {});

    expect(page.load).toHaveBeenCalledTimes(2);
    expect(notices()).toEqual([]);
  });

  it("does not push a live save to the activity again when it is sent at connect", async () => {
    await mountOnCapture();
    deliver(written);
    const taken = page.useStore.getState().agentActivity;
    expect(taken?.envelope).toEqual(written);

    reconnect("annotate");
    deliver(PANEL_EVENTS_AT_CONNECT.labels_written);

    expect(page.useStore.getState().agentActivity).toBe(taken);
  });

  it("lists a save under the project it was posted for, not the one open", async () => {
    const ridge = openedAs(PICKER.ridge).project;
    expect(ridge.id).not.toBe(written.project_id);
    await mountOnCapture(ridge);

    deliver(written);

    expect(notices()).toEqual([
      expect.objectContaining({ context: { project: written.project_id, target: "Annotate" } }),
    ]);
  });

  it("shows no agent line in the status bar after a live save", async () => {
    await mountOnCapture();

    deliver(written);

    expect(screen.queryByText(new RegExp(`labels written on ${written.event.name}`))).toBeNull();
    expect(screen.getByText(/^Zoom: /)).toBeInTheDocument();
  });
});

describe("ProjectShell agent saves on the Annotate tab", () => {
  const { labels_written: written, labels_written_second: second } = PANEL_EVENTS;
  const [first, next] = RAW_SELECTION.images;
  const reloadWarning =
    "A process just updated this image's labels. Reload to load them (discards your unsaved edits), or keep editing.";

  /** The entries the open area lists saying the reload warning. */
  const warnings = () => listItems().filter((item) => within(item).queryByText(reloadWarning));

  /** The status bar's readout of the size of the image whose document the canvas holds. */
  const sizeReadout = () => screen.queryByText(/^Image: /);

  it("loads the image in view once at a return to the tab after a save over a clean canvas", async () => {
    expect(written.event.image_path).toBe(first.path);
    await mountOnCapture();
    deliver(written);
    await waitFor(() => expect(page.load).toHaveBeenCalledTimes(2));
    await settle();

    await switchTo("meta");
    await switchTo("annotate");

    expect(page.load).toHaveBeenCalledTimes(3);
    expect(page.load).toHaveBeenLastCalledWith(first.path);
  });

  it("raises the reload warning once, not again at a return to the tab, over unsaved edits it keeps", async () => {
    answerSaves(refusedAsChanged);
    await mountOnCapture();
    addBox();
    deliver(written);
    expect(warnings()).toHaveLength(1);
    expectRaisedOnce(warnings()[0]);

    await switchTo("meta");
    await switchTo("annotate");

    expect(warnings()).toHaveLength(1);
    expectRaisedOnce(warnings()[0]);
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
    expect(screen.getByText("1 box")).toBeInTheDocument();
  });

  it("lists a save delivered while another tab is active, and loads once at the return", async () => {
    await mountOnCapture();
    await switchTo("meta");

    deliver(written);
    expect(listItems()).toHaveLength(1);
    expect(listItems()[0]).toHaveTextContent("A process: labels written");
    await switchTo("annotate");

    expect(page.load).toHaveBeenCalledTimes(2);
  });

  it("loads the image in view again for a save delivered before its first load lands, keeping the later answer", async () => {
    const before = loadOf(first.path);
    const after = LANDED.box.document;
    expect(after.image_path).toBe(first.path);
    expect([before.boxes.length, after.boxes.length]).toEqual([0, 1]);
    expect([after.polygons, after.points]).toEqual([[], []]);
    const answerFirstLoad = holdNextLoad();
    page.load.mockResolvedValueOnce(after);
    await mountOnCapture();
    expect(screen.queryByText("1 box")).toBeNull();

    deliver(written);
    await waitFor(() => expect(page.load).toHaveBeenCalledTimes(2));
    await settle();
    answerFirstLoad(before);
    await settle();

    expect(page.load).toHaveBeenLastCalledWith(first.path);
    expect(screen.getByText("1 box")).toBeInTheDocument();
  });

  it("loads the image in view again for a save delivered after its first load was refused", async () => {
    page.load.mockRejectedValueOnce(await thrown(ANNOTATE_ANSWERS.unreadable));
    const { img_width: width, img_height: height } = loadOf(first.path);
    await mountOnCapture();
    expect(sizeReadout()).toBeNull();

    deliver(written);
    await waitFor(() => expect(page.load).toHaveBeenCalledTimes(2));
    await settle();

    expect(page.load).toHaveBeenLastCalledWith(first.path);
    expect(sizeReadout()).toHaveTextContent(`Image: ${width}×${height}`);
  });

  it("loads the image just moved to again for a save of it delivered while it loads, leaving the previous image once", async () => {
    expect(second.event.image_path).toBe(next.path);
    await mountOnCapture();
    addBox();
    holdNextLoad();

    fireEvent.click(screen.getByRole("button", { name: "Next image" }));
    await settle();
    expect(page.save).toHaveBeenCalledTimes(1);
    deliver(second);
    await settle();

    expect(page.load.mock.calls.filter(([path]) => path === next.path)).toHaveLength(2);
    expect(page.save).toHaveBeenCalledTimes(1);
    expect(warnings()).toEqual([]);
    expect(imagePosition()).toHaveValue("2");
    expect(screen.queryByText("1 box")).toBeNull();
    expect(unsaved()).toBeNull();
  });
});

describe("ProjectShell a move to another image over unsaved edits", () => {
  const [first, second, third] = RAW_SELECTION.images;
  const saved = ANNOTATE_ANSWERS.saves.box;

  /** Press the image stepper's control that moves `way`, as a person does. */
  const step = (way: "Next" | "Previous") =>
    fireEvent.click(screen.getByRole("button", { name: `${way} image` }));

  /** The loads of the image at `path` the page has sent. */
  const loadsOf = (path: string) => page.load.mock.calls.filter(([p]) => p === path);

  /** Answer a held leaving save with `answer` and let what follows land. */
  const answerHeld = (held: PromiseWithResolvers<Response>, answer: () => Promise<Response>) =>
    act(async () => held.resolve(await answer()));

  /** Expect the one box added to the first image unsaved, with that image in view. */
  function expectBackOnFirst() {
    expect(unsaved()).toBeInTheDocument();
    expect(imagePosition()).toHaveValue("1");
    expect(screen.getByText("1 box")).toBeInTheDocument();
  }

  it("returns to the image left with its edits and offers Reload under the canvas when the leaving save is refused as changed", async () => {
    const sent = answerSaves(refusedAsChanged);
    await mountOnCapture();
    addBox();

    step("Next");
    await settle();

    expectBackOnFirst();
    expect(sent).toHaveLength(1);
    expect(sent[0].body).toMatchObject({ image_path: first.path });
    expect(within(inPlace()).getByText(RELOAD_CHOICE)).toBeInTheDocument();
    expect(within(inPlace()).getByRole("button", { name: "Reload" })).toBeInTheDocument();
    const failures = listing(CONFLICT_SENTENCE);
    expect(failures).toHaveLength(1);
    expectRaisedOnce(failures[0]);
  });

  it("loads the latest once Reload is pressed after a refused leaving save, and then moves", async () => {
    const latest = LANDED.box.document;
    expect(latest.image_path).toBe(first.path);
    const sent = answerSaves(refusedAsChanged);
    await mountOnCapture();
    addBox();
    addBox();
    step("Next");
    await settle();
    page.load.mockResolvedValueOnce(latest);

    fireEvent.click(within(inPlace()).getByRole("button", { name: "Reload" }));
    await settle();

    expect(page.load).toHaveBeenLastCalledWith(first.path);
    expect(screen.getByText("1 box")).toBeInTheDocument();
    expect(unsaved()).toBeNull();
    expect(screen.queryByTestId("entry-in-place")).toBeNull();

    step("Next");
    await settle();

    expect(imagePosition()).toHaveValue("2");
    expect(page.load).toHaveBeenLastCalledWith(second.path);
    expect(unsaved()).toBeNull();
    expect(sent).toHaveLength(1);
  });

  it("returns to the image left with its edits when the leaving save got no answer, and moves once a pressed Save lands", async () => {
    const sent = answerSaves(unanswered, answering(200, saved));
    await mountOnCapture();
    addBox();

    step("Next");
    await settle();

    expectBackOnFirst();
    expect(within(inPlace()).getByText(SAVE_AGAIN)).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await settle();
    expect(sent).toHaveLength(2);
    expect(unsaved()).toBeNull();
    step("Next");
    await settle();

    expect(imagePosition()).toHaveValue("2");
    expect(page.load).toHaveBeenLastCalledWith(second.path);
    expect(sent).toHaveLength(2);
  });

  it("shows the next image's document only once a leaving save answered after it lands", async () => {
    const leaving = Promise.withResolvers<Response>();
    const sent = answerSaves(() => leaving.promise);
    await mountOnCapture();
    addBox();

    step("Next");
    await settle();

    expect(loadsOf(second.path)).toHaveLength(1);
    expect(screen.getByText("1 box")).toBeInTheDocument();
    expect(sent).toHaveLength(1);

    await answerHeld(leaving, answering(200, saved));

    expect(imagePosition()).toHaveValue("2");
    expect(screen.queryByText("1 box")).toBeNull();
    expect(unsaved()).toBeNull();
    expect(listItems()).toEqual([]);
  });

  it("returns to the image left with its edits when the save of two moves made before it answers is refused", async () => {
    const leaving = Promise.withResolvers<Response>();
    const sent = answerSaves(() => leaving.promise);
    await mountOnCapture();
    addBox();
    step("Next");
    await settle();
    step("Next");
    await settle();
    expect(imagePosition()).toHaveValue("3");
    expect(loadsOf(third.path)).toHaveLength(1);

    await answerHeld(leaving, refusedAsChanged);

    expectBackOnFirst();
    expect(sent).toHaveLength(1);
    expect(within(inPlace()).getByText(RELOAD_CHOICE)).toBeInTheDocument();
  });

  it("returns to the image left with its edits when a move by the next-image key meets a refused leaving save", async () => {
    const sent = answerSaves(refusedAsChanged);
    await mountOnCapture();
    addBox();

    fireEvent.keyDown(window, { key: "ArrowRight" });
    await settle();

    expect(loadsOf(second.path)).toHaveLength(1);
    expectBackOnFirst();
    expect(sent).toHaveLength(1);
    expect(within(inPlace()).getByText(RELOAD_CHOICE)).toBeInTheDocument();
  });

  it("keeps the edits of a step back made before the leaving save answers, loading nothing over them", async () => {
    const leaving = Promise.withResolvers<Response>();
    answerSaves(() => leaving.promise);
    await mountOnCapture();
    addBox();
    holdNextLoad();
    step("Next");
    await settle();

    step("Previous");
    await settle();

    expect(loadsOf(first.path)).toHaveLength(1);
    expect(imagePosition()).toHaveValue("1");
    expect(screen.getByText("1 box")).toBeInTheDocument();

    await answerHeld(leaving, answering(200, saved));

    expect(imagePosition()).toHaveValue("1");
    expect(unsaved()).toBeNull();
  });

  it("keeps the edits of a step back made after the next image's labels answered and before the leaving save did", async () => {
    const leaving = Promise.withResolvers<Response>();
    answerSaves(() => leaving.promise);
    await mountOnCapture();
    addBox();
    step("Next");
    await settle();
    expect(loadsOf(second.path)).toHaveLength(1);

    step("Previous");
    await settle();

    expect(loadsOf(first.path)).toHaveLength(1);
    expectBackOnFirst();

    await answerHeld(leaving, answering(200, saved));

    expect(imagePosition()).toHaveValue("1");
    expect(unsaved()).toBeNull();
  });

  it("sends a leaving save that landed before the next image's labels once, whatever move follows", async () => {
    const sent = answerSaves(answering(200, saved));
    await mountOnCapture();
    addBox();
    holdNextLoad();
    step("Next");
    await settle();

    step("Next");
    await settle();

    expect(sent).toHaveLength(1);
    expect(imagePosition()).toHaveValue("3");
    expect(page.load).toHaveBeenLastCalledWith(third.path);
    expect(unsaved()).toBeNull();
    expect(failureEntries()).toEqual([]);
  });
});

describe("ProjectShell a return to the Annotate tab over unsaved edits", () => {
  const [first] = RAW_SELECTION.images;
  const saved = ANNOTATE_ANSWERS.saves.box;

  /** The image filter's choices, each with how many images it keeps. */
  const filterChoices = () =>
    within(screen.getByRole("combobox", { name: "Images" }))
      .getAllByRole("option")
      .map((option) => option.textContent);

  /** Mount on the capture with the page's saves answered by `answers`, add a box, then switch to
   *  the Meta tab and back; returns the save requests and how many loads the page sent before. */
  async function leaveAndReturn(...answers: (() => Promise<Response>)[]) {
    const sent = answerSaves(...answers);
    await mountOnCapture();
    addBox();
    const loadsBefore = page.load.mock.calls.length;
    await switchTo("meta");
    await switchTo("annotate");
    return { sent, loadsBefore };
  }

  it("leaves the canvas saved after a leaving save that landed, sending it once and listing no failure", async () => {
    const { sent } = await leaveAndReturn(answering(200, saved));

    expect(sent).toHaveLength(1);
    expect(failureEntries()).toEqual([]);
    expect(unsaved()).toBeNull();
  });

  it("keeps the edits of a leaving save refused as changed, offering Reload in place and listing the refusal once", async () => {
    const { sent, loadsBefore } = await leaveAndReturn(refusedAsChanged);

    expect(unsaved()).toBeInTheDocument();
    expect(screen.getByText("1 box")).toBeInTheDocument();
    expect(page.load).toHaveBeenCalledTimes(loadsBefore);
    expect(sent).toHaveLength(1);
    expect(within(inPlace()).getByText(RELOAD_CHOICE)).toBeInTheDocument();
    expect(within(inPlace()).getByRole("button", { name: "Reload" })).toBeInTheDocument();
    const failures = listing(CONFLICT_SENTENCE);
    expect(failures).toHaveLength(1);
    expect(failureEntries()).toEqual(failures);
    expectRaisedOnce(failures[0]);
  });

  it("loads the latest over the kept edits once Reload is pressed after a refused leaving save", async () => {
    const latest = LANDED.box.document;
    expect(latest.image_path).toBe(first.path);
    expect(latest.boxes).toHaveLength(1);
    const sent = answerSaves(refusedAsChanged);
    await mountOnCapture();
    addBox();
    addBox();
    await switchTo("meta");
    await switchTo("annotate");
    const reload = within(inPlace()).getByRole("button", { name: "Reload" });
    expect(screen.getByText("2 boxes")).toBeInTheDocument();
    const loadsBefore = page.load.mock.calls.length;
    page.load.mockResolvedValueOnce(latest);

    fireEvent.click(reload);
    await settle();

    expect(page.load).toHaveBeenCalledTimes(loadsBefore + 1);
    expect(page.load).toHaveBeenLastCalledWith(first.path);
    expect(screen.getByText("1 box")).toBeInTheDocument();
    expect(unsaved()).toBeNull();
    expect(screen.queryByTestId("entry-in-place")).toBeNull();
    expect(sent).toHaveLength(1);
  });

  it("keeps the canvas a leaving save still holds at the return, and adopts its answer when it lands", async () => {
    const leaving = Promise.withResolvers<Response>();
    const { sent, loadsBefore } = await leaveAndReturn(
      () => leaving.promise,
      answering(200, saved),
    );

    expect(page.load).toHaveBeenCalledTimes(loadsBefore);
    expect(sent).toHaveLength(1);
    await act(() => Promise.resolve(leaving.resolve(answeredWith(200, saved))));
    expect(unsaved()).toBeNull();

    addBox();
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await settle();

    expect(sent).toHaveLength(2);
    expect(sent[1].body).toMatchObject({
      image_path: first.path,
      base_mtime: saved.document.base_mtime,
    });
  });

  it("counts the image a leaving save landed on in the image filter, not the image the tab mounted on", async () => {
    const onSecond = { ...SUBJECT_SELECT_ANSWER.selection, current_image_index: 1 };
    const leaving = Promise.withResolvers<Response>();
    answerSaves(() => leaving.promise);
    await mountOnCapture(TEST_PROJECT, onSecond);
    fireEvent.click(screen.getByRole("button", { name: "Previous image" }));
    await waitFor(() => expect(page.load).toHaveBeenLastCalledWith(first.path));
    await settle();
    addBox();
    await switchTo("meta");
    await switchTo("annotate");
    expect(filterChoices()).toEqual(expect.arrayContaining(["unannotated (4)", "in progress (1)"]));

    await act(() => Promise.resolve(leaving.resolve(answeredWith(200, saved))));

    expect(filterChoices()).toEqual(expect.arrayContaining(["unannotated (3)", "in progress (2)"]));
  });

  it("keeps the edits of a leaving save that got no answer, saying so in place and listing its failure once", async () => {
    const { sent, loadsBefore } = await leaveAndReturn(unanswered);

    expect(unsaved()).toBeInTheDocument();
    expect(page.load).toHaveBeenCalledTimes(loadsBefore);
    expect(sent).toHaveLength(1);
    expect(within(inPlace()).getByText(SAVE_AGAIN)).toBeInTheDocument();
    const failures = failureEntries();
    expect(failures).toHaveLength(1);
    expect(within(failures[0]).getByText(SAVE_AGAIN)).toBeInTheDocument();
    expectRaisedOnce(failures[0]);
  });
});

describe("ProjectShell agent focus", () => {
  const focus = PANEL_EVENTS.annotate_focus;

  it("lists a focus it could not apply under the project the focus was posted for", async () => {
    const ridge = openedAs(PICKER.ridge).project;
    expect(ridge.id).not.toBe(focus.project_id);
    await mountOnCapture(ridge);

    deliver(focus);

    await waitFor(() =>
      expect(page.useStore.getState().notifications).toContainEqual(
        expect.objectContaining({
          kind: "failure",
          sentence: "Agent tried to focus the Annotate tab, but it couldn't be applied.",
          context: { project: focus.project_id },
        }),
      ),
    );
  });

  it("applies a focus once when it is sent again at a connect", async () => {
    await mountOnCapture();

    deliver(focus);
    await waitFor(() => expect(page.select).toHaveBeenCalledTimes(1));
    reconnect("app");
    deliver(PANEL_EVENTS_AT_CONNECT.annotate_focus);
    await act(async () => {});

    expect(page.select).toHaveBeenCalledTimes(1);
  });
});

describe("ProjectShell agent notes", () => {
  const { banner, banner_second: second, banner_blank: blank } = PANEL_EVENTS;
  /** The request a dismissal of `envelope`'s banner sends: its panel's route, and the committed
   *  person, the project the banner was posted for and its event. */
  const dismissalOf = (envelope: PanelEnvelope): Sent => ({
    path: `/api/events/${envelope.panel}/dismissals`,
    body: { user: "grower", project_id: envelope.project_id, event_id: envelope.event_id },
  });
  /** The projects of the entries listed, which the area does not show. */
  const listedProjects = () => page.useStore.getState().notifications.map((e) => e.context.project);
  /** The one entry the open area lists. */
  function onlyItem(): HTMLElement {
    const items = listItems();
    expect(items).toHaveLength(1);
    return items[0];
  }
  /** Expect `item`'s line to name its level's word, its kind and where the note was pushed. */
  const expectLine = (item: HTMLElement, line: string) =>
    expect(within(item).getByText(new RegExp(`^${line},`))).toBeInTheDocument();
  /** Expect the open area to list nothing. */
  function expectListsNothing() {
    expect(listItems()).toEqual([]);
    expect(area().getByText(/^Nothing to report\./)).toBeInTheDocument();
  }

  it("lists two banners one panel's socket delivers as two entries, each its whole text, naming the agent's note on that tab", async () => {
    await mountOnCapture();

    deliver(banner);
    deliver(second);

    const items = listItems();
    expect(items).toHaveLength(2);
    for (const { event } of [banner, second]) {
      const showing = items.filter((item) => within(item).queryByText(event.text));
      expect(showing).toHaveLength(1);
      expectLine(showing[0], "Information, notice, Agent's note for Results");
    }
    expect(listedProjects()).toEqual([banner.project_id, banner.project_id]);
  });

  it("lists the application panel's banner sent at connect", async () => {
    const appBanner = PANEL_EVENTS_AT_CONNECT.app_banner;
    await mountOnCapture();

    deliver(appBanner);

    const item = onlyItem();
    expect(within(item).getByText(appBanner.event.text)).toBeInTheDocument();
    expectLine(item, "Information, notice, Agent's note for the app");
    expect(listedProjects()).toEqual([appBanner.project_id]);
  });

  it("adds no entry and no count for a banner sent again at a reconnect", async () => {
    await mountOnCapture();
    deliver(banner);
    expect(within(onlyItem()).getByText(banner.event.text)).toBeInTheDocument();
    expectRaisedOnce(onlyItem());

    reconnect("results");
    deliver(PANEL_EVENTS_AT_CONNECT.banner);

    expect(within(onlyItem()).getByText(banner.event.text)).toBeInTheDocument();
    expectRaisedOnce(onlyItem());
  });

  it("lists nothing for a banner whose text is blank", async () => {
    expect(blank.event.text.trim()).toBe("");
    await mountOnCapture();

    deliver(blank);
    deliver(second);

    expect(within(onlyItem()).getByText(second.event.text)).toBeInTheDocument();
  });

  it("sends a dismissed banner's dismissal to its panel's route by the person, for its project and event, leaving the other listed", async () => {
    const sent = answerDismissals(answering(200, DISMISSALS.dismissed));
    await mountOnCapture();
    deliver(banner);
    deliver(second);

    dismissEntry(banner.event.text);
    await settle();

    expect(sent).toEqual([dismissalOf(banner)]);
    expect(within(onlyItem()).getByText(second.event.text)).toBeInTheDocument();
  });

  it("lists, on a page loaded anew and sent what the backend holds after a dismissal, the banners not dismissed and not the dismissed one", async () => {
    await mountOnCapture();

    for (const envelope of Object.values(PANEL_EVENTS_AFTER_DISMISSAL)) deliver(envelope);
    await settle();

    const notes = listItems().filter((item) =>
      within(item).queryByText(/^Information, notice, Agent's note for /),
    );
    expect(notes).toHaveLength(2);
    expect(within(notes[0]).getByText(PANEL_EVENTS.app_banner.event.text)).toBeInTheDocument();
    expect(within(notes[1]).getByText(second.event.text)).toBeInTheDocument();
    expect(area().queryByText(banner.event.text)).not.toBeInTheDocument();
  });

  it("tells a dismissal whose line was not written as a warning carrying its sentence, offering no Retry", async () => {
    const gap = DISMISSALS.audit_gap;
    const sent = answerDismissals(answering(gap.status, gap.body));
    await mountOnCapture();
    deliver(second);

    dismissEntry(second.event.text);
    await settle();

    expect(sent).toEqual([dismissalOf(second)]);
    const item = onlyItem();
    expect(within(item).getByText(gap.body.detail.message)).toBeInTheDocument();
    expectLine(item, "Warning, notice, Agent's note for Results");
    expect(listedProjects()).toEqual([second.project_id]);
    expect(area().queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
  });

  it.each([
    ["refused naming no person", answering(DISMISSALS.no_one.status, DISMISSALS.no_one.body)],
    ["given no answer", unanswered],
  ])(
    "tells a dismissal %s as one failure whose Retry sends it again, leaving the failure without its Retry, or any word of one, once taken",
    async (_case, failed) => {
      const sent = answerDismissals(failed, answering(200, DISMISSALS.dismissed));
      await mountOnCapture();
      deliver(banner);

      dismissEntry(banner.event.text);
      await settle();
      expectLine(onlyItem(), "Error, failure, Agent's note for Results");

      fireEvent.click(area().getByRole("button", { name: "Retry" }));
      await settle();

      expect(sent).toEqual([dismissalOf(banner), dismissalOf(banner)]);
      const item = onlyItem();
      expectLine(item, "Error, failure, Agent's note for Results");
      expectRaisedOnce(item);
      expect(item).not.toHaveTextContent(/Retry/);
    },
  );

  it("sends nothing for a banner dismissed once another project is open, and lists nothing", async () => {
    const sent = answerDismissals();
    await mountOnCapture();
    deliver(banner);
    expect(within(onlyItem()).getByText(banner.event.text)).toBeInTheDocument();

    act(() => page.useStore.setState({ openProject: openedAs(PICKER.ridge).project }));
    dismissEntry(banner.event.text);
    await settle();

    expect(sent).toEqual([]);
    expectListsNothing();
  });

  it("withdraws a failed dismissal's Retry, and any word of one, sending nothing, once it is pressed after another project opened", async () => {
    const sent = answerDismissals(unanswered);
    await mountOnCapture();
    deliver(banner);
    dismissEntry(banner.event.text);
    await settle();
    expect(area().getByRole("button", { name: "Retry" })).toBeInTheDocument();

    act(() => page.useStore.setState({ openProject: openedAs(PICKER.ridge).project }));
    fireEvent.click(area().getByRole("button", { name: "Retry" }));
    await settle();

    expect(sent).toEqual([dismissalOf(banner)]);
    const item = onlyItem();
    expectLine(item, "Error, failure, Agent's note for Results");
    expectRaisedOnce(item);
    expect(item).not.toHaveTextContent(/Retry/);
  });

  it("lists nothing for a dismissal refused because the backend has another project open", async () => {
    const refused = DISMISSALS.another_project;
    const sent = answerDismissals(answering(refused.status, refused.body));
    await mountOnCapture();
    deliver(banner);

    dismissEntry(banner.event.text);
    await settle();

    expect(sent).toEqual([dismissalOf(banner)]);
    expectListsNothing();
  });
});
