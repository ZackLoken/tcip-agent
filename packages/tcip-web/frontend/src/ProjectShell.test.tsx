import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

import type { LoadedLabels } from "@/api/client";
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
import { NAV_ANSWER, RAW_SELECTION, SELECTED_SUBJECT, TEST_PROJECT } from "@/test/store";
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

/** Mount the shell beside the open notifications area with `project` open on the produced
 *  capture, the Annotate tab active, and wait for its first image's labels to load. */
async function mountOnCapture(project = TEST_PROJECT) {
  const { ProjectShell, NotificationsPane, useStore, load } = page;
  useStore.getState().setUser("grower");
  useStore.setState((s) => ({
    openProject: project,
    gui: { ...s.gui, active_tab: "annotate", dataset: RAW_SELECTION },
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

/** A dismissal route's request as the page sent it: its path and its body. */
type Sent = { path: string; body: unknown };

/** Answer each request the page sends to a dismissal route with the next of `answers`, and
 *  every other request as the page's own fetch does; return the dismissal requests, in order. */
function answerDismissals(...answers: (() => Promise<Response>)[]): Sent[] {
  const pageFetch = globalThis.fetch;
  const sent: Sent[] = [];
  vi.stubGlobal("fetch", (input: RequestInfo | URL, init?: RequestInit) => {
    const path = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    if (!path.endsWith("/dismissals")) return pageFetch(input, init);
    const body = init?.body;
    sent.push({ path, body: typeof body === "string" ? (JSON.parse(body) as unknown) : body });
    const answer = answers.shift();
    if (!answer) throw new Error(`no answer left for ${path}`);
    return answer();
  });
  return sent;
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

  /** The listed entries saying the reload warning. */
  const warnings = () =>
    page.useStore.getState().notifications.filter((e) => e.sentence === reloadWarning);

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

  it("raises the reload warning once, not again at a return to the tab, over unsaved edits", async () => {
    await mountOnCapture();
    addBox();
    deliver(written);
    expect(warnings()).toEqual([expect.objectContaining({ count: 1 })]);

    await switchTo("meta");
    await switchTo("annotate");

    const listing = area()
      .getAllByRole("listitem")
      .filter((item) => within(item).queryByText(reloadWarning));
    expect(listing).toHaveLength(1);
    expect(listing[0]).not.toHaveTextContent(/ times/);
  });

  it("lists a save delivered while another tab is active, and loads once at the return", async () => {
    await mountOnCapture();
    await switchTo("meta");

    deliver(written);
    expect(notices().map((e) => e.sentence)).toEqual(["A process: labels written"]);
    await switchTo("annotate");

    expect(page.load).toHaveBeenCalledTimes(2);
  });

  it("loads the image in view again for a save delivered before its first load lands, keeping the later answer", async () => {
    const after = LANDED.box.document;
    expect(after.image_path).toBe(first.path);
    const answerFirstLoad = holdNextLoad();
    page.load.mockResolvedValueOnce(after);
    await mountOnCapture();

    deliver(written);
    await waitFor(() => expect(page.load).toHaveBeenCalledTimes(2));
    await settle();
    answerFirstLoad(loadOf(first.path));
    await settle();

    expect(page.load).toHaveBeenLastCalledWith(first.path);
    expect(page.useStore.getState().canvas.loaded).toBe(after);
  });

  it("loads the image in view again for a save delivered after its first load was refused", async () => {
    page.load.mockRejectedValueOnce(await thrown(ANNOTATE_ANSWERS.unreadable));
    await mountOnCapture();
    expect(page.useStore.getState().canvas.loaded).toBeNull();

    deliver(written);
    await waitFor(() => expect(page.load).toHaveBeenCalledTimes(2));
    await settle();

    expect(page.load).toHaveBeenLastCalledWith(first.path);
    expect(page.useStore.getState().canvas.loaded?.image_path).toBe(first.path);
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
  /** The entries the open area lists, newest first. */
  const listItems = () => area().queryAllByRole("listitem");
  /** The one entry the open area lists. */
  function onlyItem(): HTMLElement {
    const items = listItems();
    expect(items).toHaveLength(1);
    return items[0];
  }
  /** Expect `item`'s line to name its level's word, its kind and where the note was pushed. */
  const expectLine = (item: HTMLElement, line: string) =>
    expect(within(item).getByText(new RegExp(`^${line},`))).toBeInTheDocument();
  /** Expect `item` to state no count of times raised. */
  const expectRaisedOnce = (item: HTMLElement) => expect(item).not.toHaveTextContent(/ times/);
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
