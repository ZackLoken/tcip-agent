import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

import type { PanelEnvelope } from "@/api/types.generated";
import { dismissLabel } from "@/lib/dismissLabel";
import { LABEL_ANSWERS } from "@/test/labels";
import {
  DISMISSALS,
  PANEL_EVENTS,
  PANEL_EVENTS_AFTER_DISMISSAL,
  PANEL_EVENTS_AT_CONNECT,
} from "@/test/panelEvents";
import { openedAs, PICKER } from "@/test/projectRecords";
import { answeredWith } from "@/test/refusals";
import { NAV_ANSWER, RAW_SELECTION, TEST_PROJECT } from "@/test/store";
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
  select: MockInstance<typeof import("@/api/client").api.dataset.select>;
};

/** The page each test runs in. */
let page: Page;

/** Load every module anew, as a reload of the page does, and answer what the shell and its
 *  Annotate tab read on the produced capture's first image. */
async function loadPage(): Promise<Page> {
  vi.resetModules();
  const [
    { ProjectShell },
    { NotificationsPane },
    { useStore },
    { api },
    { answerAnnotateMount },
    subjects,
    training,
    inference,
  ] = await Promise.all([
    import("@/ProjectShell"),
    import("@/components/Notifications"),
    import("@/store"),
    import("@/api/client"),
    import("@/test/annotate"),
    import("@/api/subjects"),
    import("@/api/training"),
    import("@/api/inference"),
  ]);
  const load = answerAnnotateMount();
  const select = vi.spyOn(api.dataset, "select").mockRejectedValue(new Error("selection refused"));
  vi.spyOn(api.dataset, "nav").mockResolvedValue(NAV_ANSWER);
  vi.spyOn(subjects.subjectsApi, "load").mockResolvedValue(LABEL_ANSWERS.registry_loaded);
  vi.spyOn(training.trainingApi, "listRuns").mockResolvedValue({ runs: [], sweeps: [] });
  vi.spyOn(inference.inferenceApi, "listJobs").mockResolvedValue({ jobs: [] });
  return { ProjectShell, NotificationsPane, useStore, load, select };
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
  const listed = () => page.useStore.getState().notifications;

  it("lists two banners one panel's socket delivers as two entries, each its whole text, naming the agent's note on that tab", async () => {
    await mountOnCapture();

    deliver(banner);
    deliver(second);

    const items = area().getAllByRole("listitem");
    expect(items).toHaveLength(2);
    for (const { event } of [banner, second]) {
      expect(items.filter((item) => within(item).queryByText(event.text))).toHaveLength(1);
    }
    for (const { context } of listed()) {
      expect(context.project).toBe(banner.project_id);
      expect(context.target).toMatch(/Results/);
      expect(context.target).not.toBe("Results");
    }
  });

  it("lists the application panel's banner sent at connect", async () => {
    const appBanner = PANEL_EVENTS_AT_CONNECT.app_banner;
    await mountOnCapture();

    deliver(appBanner);

    expect(area().getByText(appBanner.event.text)).toBeInTheDocument();
    expect(listed().map((e) => [e.kind, e.context.project])).toEqual([
      ["notice", appBanner.project_id],
    ]);
  });

  it("adds no entry and no count for a banner sent again at a reconnect", async () => {
    await mountOnCapture();
    deliver(banner);
    expect(listed()).toEqual([expect.objectContaining({ sentence: banner.event.text, count: 1 })]);

    reconnect("results");
    deliver(PANEL_EVENTS_AT_CONNECT.banner);

    expect(listed()).toEqual([expect.objectContaining({ sentence: banner.event.text, count: 1 })]);
  });

  it("lists nothing for a banner whose text is blank", async () => {
    expect(blank.event.text.trim()).toBe("");
    await mountOnCapture();

    deliver(blank);
    deliver(second);

    expect(listed().map((e) => e.sentence)).toEqual([second.event.text]);
  });

  it("sends a dismissed banner's dismissal to its panel's route by the person, for its project and event, leaving the other listed", async () => {
    const sent = answerDismissals(answering(200, DISMISSALS.dismissed));
    await mountOnCapture();
    deliver(banner);
    deliver(second);

    dismissEntry(banner.event.text);
    await settle();

    expect(sent).toEqual([dismissalOf(banner)]);
    expect(listed().map((e) => e.sentence)).toEqual([second.event.text]);
    expect(area().getAllByRole("listitem")).toHaveLength(1);
  });

  it("lists, on a page loaded anew and sent what the backend holds after a dismissal, the banners not dismissed and not the dismissed one", async () => {
    await mountOnCapture();

    for (const envelope of Object.values(PANEL_EVENTS_AFTER_DISMISSAL)) deliver(envelope);
    await settle();

    const notes = listed().filter((e) => e.kind === "notice");
    expect(notes.map((e) => e.sentence)).toEqual([
      second.event.text,
      PANEL_EVENTS.app_banner.event.text,
    ]);
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
    expect(listed()).toEqual([
      expect.objectContaining({
        kind: "notice",
        level: "warning",
        sentence: gap.body.detail.message,
      }),
    ]);
    expect(listed()[0].context.project).toBe(second.project_id);
    expect(area().queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
  });

  it.each([
    ["refused naming no person", answering(DISMISSALS.no_one.status, DISMISSALS.no_one.body)],
    ["given no answer", unanswered],
  ])(
    "tells a dismissal %s as one failure whose Retry sends it again, leaving the failure without its Retry once taken",
    async (_case, failed) => {
      const sent = answerDismissals(failed, answering(200, DISMISSALS.dismissed));
      await mountOnCapture();
      deliver(banner);

      dismissEntry(banner.event.text);
      await settle();
      expect(listed()).toEqual([expect.objectContaining({ kind: "failure", level: "error" })]);

      fireEvent.click(area().getByRole("button", { name: "Retry" }));
      await settle();

      expect(sent).toEqual([dismissalOf(banner), dismissalOf(banner)]);
      expect(listed()).toEqual([expect.objectContaining({ kind: "failure", count: 1 })]);
      expect(area().queryByRole("button", { name: "Retry" })).not.toBeInTheDocument();
    },
  );

  it("sends nothing for a banner dismissed once another project is open, and lists nothing", async () => {
    const sent = answerDismissals();
    await mountOnCapture();
    deliver(banner);
    expect(listed()).toHaveLength(1);

    act(() => page.useStore.setState({ openProject: openedAs(PICKER.ridge).project }));
    dismissEntry(banner.event.text);
    await settle();

    expect(sent).toEqual([]);
    expect(listed()).toEqual([]);
  });

  it("lists nothing for a dismissal refused because the backend has another project open", async () => {
    const refused = DISMISSALS.another_project;
    const sent = answerDismissals(answering(refused.status, refused.body));
    await mountOnCapture();
    deliver(banner);

    dismissEntry(banner.event.text);
    await settle();

    expect(sent).toEqual([dismissalOf(banner)]);
    expect(listed()).toEqual([]);
  });
});
