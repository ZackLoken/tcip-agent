import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

import type { PanelEnvelope } from "@/api/types.generated";
import { LABEL_ANSWERS } from "@/test/labels";
import { PANEL_EVENTS, PANEL_EVENTS_AT_CONNECT } from "@/test/panelEvents";
import { openedAs, PICKER } from "@/test/projectRecords";
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
    { useStore },
    { api },
    { answerAnnotateMount },
    subjects,
    training,
    inference,
  ] = await Promise.all([
    import("@/ProjectShell"),
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
  return { ProjectShell, useStore, load, select };
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

/** Drop `panel`'s socket and let the page connect it again. */
async function reconnect(panel: string) {
  const before = FakeWebSocket.instances.length;
  act(() => panelSocket(panel).drop());
  await waitFor(() => expect(FakeWebSocket.instances.length).toBeGreaterThan(before), {
    timeout: 2000,
  });
  act(() => panelSocket(panel).open());
}

/** The notice entries the notifications area lists. */
const notices = () => page.useStore.getState().notifications.filter((e) => e.kind === "notice");

/** Mount the shell with `project` open on the produced capture, the Annotate tab active, and wait
 *  for its first image's labels to load. */
async function mountOnCapture(project = TEST_PROJECT) {
  const { ProjectShell, useStore, load } = page;
  useStore.getState().setUser("grower");
  useStore.setState((s) => ({
    openProject: project,
    gui: { ...s.gui, active_tab: "annotate", dataset: RAW_SELECTION },
  }));
  render(<ProjectShell />);
  await waitFor(() => expect(load).toHaveBeenCalledTimes(1));
  await act(async () => {});
  for (const panel of ["annotate", "app"]) act(() => panelSocket(panel).open());
}

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

    await reconnect("annotate");
    deliver(atConnect);
    await act(async () => {});

    expect(page.load).toHaveBeenCalledTimes(2);
    expect(notices()).toEqual([]);
  });

  it("counts a live save sent again at connect once", async () => {
    await mountOnCapture();

    deliver(written);
    await reconnect("annotate");
    deliver(PANEL_EVENTS_AT_CONNECT.labels_written);

    expect(notices()).toEqual([expect.objectContaining({ count: 1 })]);
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
    await reconnect("app");
    deliver(PANEL_EVENTS_AT_CONNECT.annotate_focus);
    await act(async () => {});

    expect(page.select).toHaveBeenCalledTimes(1);
  });
});
