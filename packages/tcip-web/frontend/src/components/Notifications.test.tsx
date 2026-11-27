import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { sessionsApi } from "@/api/sessions";
import { CONNECTION_LOST, StateSocket } from "@/api/ws";
import {
  NotificationItem,
  NotificationsButton,
  NotificationsLiveRegion,
  NotificationsPane,
} from "@/components/Notifications";
import { dismissLabel } from "@/lib/dismissLabel";
import { postHeldContributions } from "@/lib/sessionLifecycle";
import { useStore } from "@/store";
import type { EntryAction } from "@/store/slices/notifications";
import { announcements, areaEntries, metaLine, onlyAreaEntry } from "@/test/notifications";
import { thrown } from "@/test/refusals";
import { SESSION_ANSWERS } from "@/test/sessions";
import { PRODUCED_ENVELOPE, openTestProject } from "@/test/store";
import { FakeWebSocket, lastSocket } from "@/test/webSocket";

const initialStoreState = useStore.getState();

/** The notifications host as App mounts it while the area is closed, the button and the live
 *  region, under strict mode as `main.tsx` renders it. */
function renderClosedHost() {
  return render(
    <StrictMode>
      <NotificationsButton open={false} onToggle={() => {}} />
      <NotificationsLiveRegion />
    </StrictMode>,
  );
}

/** The notifications host as App mounts it while the area is open: the button, the pane the
 *  dock holds and the live region, under strict mode. */
function renderOpenHost() {
  return render(
    <StrictMode>
      <NotificationsButton open onToggle={() => {}} />
      <NotificationsPane />
      <NotificationsLiveRegion />
    </StrictMode>,
  );
}

async function flush() {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
});

afterEach(cleanup);

describe("the lost connection", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    FakeWebSocket.instances = [];
    vi.stubGlobal("WebSocket", FakeWebSocket);
  });

  afterEach(() => vi.useRealTimers());

  it("with the area closed raises one entry, announced once through every retry, and its mark clears once it reconnects", async () => {
    renderClosedHost();
    const told = announcements(screen.getByRole("alert"));
    const socket = new StateSocket();
    socket.connect();
    act(() => lastSocket().open());

    for (const delay of [500, 1000, 2000]) {
      act(() => lastSocket().drop());
      act(() => {
        vi.advanceTimersByTime(delay);
      });
    }
    await act(flush);

    expect(useStore.getState().notifications).toHaveLength(1);
    expect(told).toEqual([CONNECTION_LOST]);
    expect(screen.getByRole("button", { name: /a problem is ongoing/ })).toBeInTheDocument();

    act(() => lastSocket().open());
    expect(screen.queryByTestId("condition-mark")).not.toBeInTheDocument();
    expect(useStore.getState().holding.connection).toBeUndefined();
    socket.close();
  });
});

describe("the stale bundle", () => {
  const SENTENCE = "The frontend bundle was not built from the source; run the build.";

  it("is an entry carrying the backend's sentence, and ends once a frame states none", () => {
    vi.spyOn(useStore.getState(), "mergeSnapshot").mockImplementation(() => {});
    vi.stubGlobal("WebSocket", FakeWebSocket);
    renderOpenHost();
    const socket = new StateSocket();
    socket.connect();
    act(() => lastSocket().open());

    act(() => lastSocket().message(JSON.stringify({ ...PRODUCED_ENVELOPE, bundle: SENTENCE })));
    const entry = onlyAreaEntry();
    expect(within(entry).getByText(SENTENCE)).toBeInTheDocument();
    expect(metaLine(entry)).toMatch(/^Information, condition, .*, ongoing$/);
    expect(screen.getByRole("status")).toHaveTextContent(SENTENCE);
    expect(screen.getByTestId("condition-mark")).toBeInTheDocument();

    act(() => lastSocket().message(JSON.stringify(PRODUCED_ENVELOPE)));
    expect(screen.queryByTestId("condition-mark")).not.toBeInTheDocument();
    expect(metaLine(onlyAreaEntry())).toMatch(/^Information, condition, .*, ended$/);
    socket.close();
  });
});

describe("the unconfirmed visits", () => {
  it("dismissed, stay away on the next failed send while visits stay unconfirmed, and the mark still shows", async () => {
    vi.spyOn(sessionsApi, "imageEvent").mockRejectedValue(new TypeError("Failed to fetch"));
    useStore.getState().setUser("jordan");
    openTestProject();
    renderOpenHost();
    const s = useStore.getState();
    s.startImageSessionTracking("a.jpg", Date.now() - 2000);
    s.closeSessionInterval();
    postHeldContributions();
    await act(flush);
    const entry = onlyAreaEntry();
    expect(entry).toHaveTextContent(/Could not confirm these visits were recorded.*a\.jpg/);
    expect(metaLine(entry)).toMatch(/^Error, condition, .*, ongoing$/);

    fireEvent.click(within(entry).getByRole("button", { name: /^Dismiss: / }));
    s.startImageSessionTracking("b.jpg", Date.now() - 2000);
    s.closeSessionInterval();
    postHeldContributions();
    await act(flush);

    expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(3);
    expect(areaEntries()).toHaveLength(0);
    expect(screen.getByText(/Nothing to report/)).toBeInTheDocument();
    expect(screen.getByTestId("condition-mark")).toBeInTheDocument();
  });
});

describe("the refused visits", () => {
  it("are one entry naming each image under its reason, dismissed by a control named for that text, announced as first refused", async () => {
    const refused = SESSION_ANSWERS.ended_named;
    vi.spyOn(sessionsApi, "imageEvent").mockRejectedValue(await thrown(refused));
    useStore.getState().setUser("jordan");
    openTestProject();
    renderOpenHost();
    const told = announcements(screen.getByRole("alert"));
    const s = useStore.getState();
    for (const image of ["a.jpg", "b.jpg"]) {
      s.startImageSessionTracking(image, Date.now() - 2000);
      s.closeSessionInterval();
    }
    postHeldContributions();
    for (let i = 0; i < 3; i++) await act(flush);

    const said = `Not recorded (${refused.body.detail}): a.jpg, b.jpg`;
    const entry = onlyAreaEntry();
    expect(within(entry).getByText(said)).toBeInTheDocument();
    expect(told).toEqual([`Not recorded (${refused.body.detail}): a.jpg`]);
    fireEvent.click(
      within(entry).getByRole("button", { name: (name) => name.startsWith(dismissLabel(said)) }),
    );
    expect(areaEntries()).toHaveLength(0);
  });
});

describe("the notifications area", () => {
  it("counts an entry new until the area is seen", () => {
    renderClosedHost();
    act(() => useStore.getState().holdCondition("bundle", "rebuild the bundle"));
    expect(screen.getByRole("button", { name: /^Notifications, 1 new/ })).toBeInTheDocument();

    act(() => useStore.getState().markNotificationsSeen());
    expect(screen.getByRole("button", { name: /^Notifications, a problem/ })).toBeInTheDocument();
  });
});

describe("the notifications button's glyph", () => {
  const states: [string, () => void, string][] = [
    ["no entry and no condition", () => {}, ""],
    [
      "a new entry",
      () =>
        useStore.getState().raiseNotification({
          kind: "failure",
          level: "error",
          text: { sentence: "Could not load the registry." },
          context: { project: "valley" },
        }),
      "1",
    ],
    [
      "a holding condition already seen",
      () => {
        useStore.getState().holdCondition("bundle", "rebuild the bundle");
        useStore.getState().markNotificationsSeen();
      },
      "",
    ],
    [
      "a new entry and a holding condition",
      () => {
        useStore.getState().holdCondition("connection", CONNECTION_LOST);
        useStore.getState().raiseNotification({
          kind: "failure",
          level: "error",
          text: { sentence: "Could not load the registry." },
          context: { project: "valley" },
        });
      },
      "2",
    ],
  ];

  it.each(states)(
    "stands for the word with %s: a hidden icon, only the count as text, and a title equal to its name",
    (_state, setUp, count) => {
      renderClosedHost();
      act(setUp);
      const button = screen.getByRole("button", { name: /^Notifications/ });

      expect(button.textContent).toBe(count);
      const icon = button.querySelector("svg");
      expect(icon).toHaveAttribute("aria-hidden", "true");
      expect(button.title).not.toBe("");
      expect(button).toHaveAccessibleName(button.title);
    },
  );
});

/** Raise a failure stating `sentence` for the project `project`, standing on no key. */
function raiseFailure(sentence: string, project: string, action?: EntryAction) {
  act(() =>
    useStore.getState().raiseNotification({
      kind: "failure",
      level: "error",
      text: { sentence },
      context: { project },
      action,
    }),
  );
}

describe("a failure", () => {
  it("raised again for the same project is one entry counted twice, another project's is its own, and none is dropped", () => {
    raiseFailure("Could not load the registry.", "valley");
    raiseFailure("Could not load the registry.", "valley");
    raiseFailure("Could not load the registry.", "ridge");

    const listed = useStore.getState().notifications;
    expect(listed.map((e) => [e.context.project, e.count])).toEqual([
      ["valley", 2],
      ["ridge", 1],
    ]);

    for (const n of [1, 2, 3, 4, 5, 6]) raiseFailure(`Failure ${n}.`, "valley");
    expect(useStore.getState().notifications).toHaveLength(8);
  });

  it("raised again on its released key with the same text is the same entry counted twice, standing on the key again", () => {
    const raise = () =>
      act(() =>
        useStore.getState().raiseNotification({
          kind: "failure",
          level: "error",
          text: { sentence: "Failed to fetch" },
          context: { project: "valley" },
          key: "open:valley",
        }),
      );
    raise();
    act(() => useStore.getState().releaseNotification("open:valley"));
    raise();

    expect(useStore.getState().notifications.map((e) => [e.key, e.count])).toEqual([
      ["open:valley", 2],
    ]);
  });

  it("standing with no condition leaves the button unmarked, and a condition marks it", () => {
    renderClosedHost();
    raiseFailure("Could not load the registry.", "valley");

    expect(screen.queryByTestId("condition-mark")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^Notifications/ })).not.toHaveAccessibleName(
      /a problem is ongoing/,
    );

    act(() => useStore.getState().holdCondition("connection", CONNECTION_LOST));
    expect(screen.getByTestId("condition-mark")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /a problem is ongoing/ })).toBeInTheDocument();
  });

  it("is announced once assertively with the area closed, and a notice once politely", async () => {
    renderClosedHost();
    const alerts = announcements(screen.getByRole("alert"));
    const statuses = announcements(screen.getByRole("status"));

    raiseFailure("Could not load the registry.", "valley");
    raiseFailure("Could not load the registry.", "valley");
    act(() =>
      useStore.getState().raiseNotification({
        kind: "notice",
        level: "info",
        text: { sentence: "Renamed Valley farm to Valley east." },
        context: { project: "valley" },
      }),
    );
    await act(flush);

    expect(alerts).toEqual(["Could not load the registry."]);
    expect(statuses).toEqual(["Renamed Valley farm to Valley east."]);
  });

  it("shows its kind, level, count and action in the area, and says neither ongoing nor ended, where a condition still does", () => {
    const run = vi.fn();
    renderOpenHost();
    raiseFailure("Could not open the project.", "valley", { label: "Retry", run });
    raiseFailure("Could not open the project.", "valley", { label: "Retry", run });

    const entry = onlyAreaEntry();
    expect(metaLine(entry)).toMatch(/^Error, failure, .*, 2 times$/);
    expect(metaLine(entry)).not.toMatch(/ongoing|ended/);
    fireEvent.click(within(entry).getByRole("button", { name: "Retry" }));
    expect(run).toHaveBeenCalledTimes(1);

    act(() => useStore.getState().holdCondition("connection", CONNECTION_LOST));
    expect(metaLine(areaEntries()[0])).toMatch(/^Error, condition, .*, ongoing$/);
  });

  /** Raise a failure stating `sentence` on `key`, offering a Retry. */
  const raiseOn = (key: string, sentence: string) =>
    act(() =>
      useStore.getState().raiseNotification({
        kind: "failure",
        level: "error",
        text: { sentence },
        context: { project: "valley" },
        key,
        action: { label: "Retry", run: vi.fn() },
      }),
    );

  it("whose cause ended is listed saying resolved after its time, offering no action, and one freed for another reason says nothing of it", () => {
    renderOpenHost();
    raiseOn("open:valley", "Could not open Valley farm.");
    raiseOn("open:ridge", "Could not open Ridge farm.");

    act(() => useStore.getState().resolveNotification("open:valley"));
    act(() => useStore.getState().releaseNotification("open:ridge"));

    const [ridge, valley] = areaEntries();
    expect(within(valley).getByText("Could not open Valley farm.")).toBeInTheDocument();
    expect(metaLine(valley)).toMatch(/^Error, failure, .+, resolved$/);
    expect(within(valley).queryByRole("button", { name: "Retry" })).toBeNull();
    expect(within(ridge).getByText("Could not open Ridge farm.")).toBeInTheDocument();
    expect(metaLine(ridge)).not.toMatch(/resolved/);
    expect(within(ridge).queryByRole("button", { name: "Retry" })).toBeNull();
  });

  it("resolved and raised again with the same text is one entry counted twice, no longer saying resolved", () => {
    renderOpenHost();
    raiseOn("open:valley", "Could not open Valley farm.");
    act(() => useStore.getState().resolveNotification("open:valley"));
    expect(metaLine(onlyAreaEntry())).toMatch(/, resolved$/);

    raiseOn("open:valley", "Could not open Valley farm.");

    const entry = onlyAreaEntry();
    expect(metaLine(entry)).toMatch(/^Error, failure, .+, 2 times$/);
    expect(within(entry).getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("offers an action stating what it acts on under its label and that, its label alone shown", () => {
    const run = vi.fn();
    renderOpenHost();
    raiseFailure("Could not open the project.", "valley", {
      label: "Retry",
      actsOn: "opening Valley farm",
      run,
    });

    const retry = within(onlyAreaEntry()).getByRole("button", {
      name: "Retry opening Valley farm",
    });
    expect(retry).toHaveTextContent(/^Retry$/);
    fireEvent.click(retry);
    expect(run).toHaveBeenCalledTimes(1);
  });
});

describe("a job", () => {
  const raiseJob = (sentence: string, retell: boolean) =>
    act(() =>
      useStore.getState().raiseNotification({
        kind: "job",
        level: "info",
        text: { sentence },
        context: { project: "valley", target: "run-1" },
        key: "training-job:valley:run-1",
        retell,
      }),
    );
  /** The job's one entry standing, and a later failure, both seen; every polite announcement after
   *  that is returned as it is made. */
  async function seenBesideAFailure(): Promise<string[]> {
    renderOpenHost();
    raiseJob("Training run run-1: running", false);
    raiseFailure("Could not load the registry.", "valley");
    act(() => useStore.getState().markNotificationsSeen());
    await act(flush);
    return announcements(screen.getByRole("status"));
  }

  it("told again on its standing entry is one entry with a new id, announced, counted as new and listed first", async () => {
    const statuses = await seenBesideAFailure();
    const [standing] = useStore.getState().notifications;

    raiseJob("Training run run-1: completed", true);
    await act(flush);

    expect(statuses).toEqual(["Training run run-1: completed"]);
    const jobs = useStore.getState().notifications.filter((e) => e.kind === "job");
    expect(jobs).toEqual([
      expect.objectContaining({ key: standing.key, sentence: "Training run run-1: completed" }),
    ]);
    expect(jobs[0].id).toBeGreaterThan(standing.id);
    expect(screen.getByRole("button", { name: "Notifications, 1 new" })).toBeInTheDocument();
    expect(areaEntries()[0]).toHaveTextContent("Training run run-1: completed");
  });

  it("raised on its standing entry without being told again updates that entry in place, unannounced", async () => {
    const statuses = await seenBesideAFailure();
    const [standing] = useStore.getState().notifications;

    raiseJob("Training run run-1: running, epoch 2", false);
    await act(flush);

    expect(statuses).toEqual([]);
    expect(useStore.getState().notifications[0]).toEqual(
      expect.objectContaining({
        id: standing.id,
        sentence: "Training run run-1: running, epoch 2",
      }),
    );
    expect(screen.getByRole("button", { name: "Notifications" })).toBeInTheDocument();
  });
});

describe("the live region", () => {
  const FIRST = "Renamed Valley farm to Valley east.";
  const SECOND = "Removed Ridge farm.";
  const raiseNotice = (sentence: string) =>
    useStore.getState().raiseNotification({
      kind: "notice",
      level: "info",
      text: { sentence },
      context: { project: "valley" },
    });

  it("marks both of its regions as presenting only what is added, as markup", () => {
    renderClosedHost();

    expect(screen.getByRole("status")).toHaveAttribute("aria-atomic", "false");
    expect(screen.getByRole("alert")).toHaveAttribute("aria-atomic", "false");
  });

  it("is given each of two notices raised in one act, in the order raised", async () => {
    renderClosedHost();
    const statuses = announcements(screen.getByRole("status"));

    act(() => {
      raiseNotice(FIRST);
      raiseNotice(SECOND);
    });
    await act(flush);

    expect(statuses).toEqual([FIRST, SECOND]);
  });

  it("holds the text of an announced entry only while the entry is listed", () => {
    renderClosedHost();
    const status = screen.getByRole("status");
    act(() => raiseNotice(FIRST));
    expect(status).toHaveTextContent(FIRST);

    act(() => useStore.getState().dismissNotification(useStore.getState().notifications[0].id));

    expect(status).toBeEmptyDOMElement();
  });

  it("is given nothing new by the dismissal of one of two announced entries, and keeps the other's text", async () => {
    renderClosedHost();
    const status = screen.getByRole("status");
    const statuses = announcements(status);
    act(() => {
      raiseNotice(FIRST);
      raiseNotice(SECOND);
    });
    const [first] = useStore.getState().notifications;

    act(() => useStore.getState().dismissNotification(first.id));
    await act(flush);

    expect(statuses).toEqual([FIRST, SECOND]);
    expect(status).not.toHaveTextContent(FIRST);
    expect(status).toHaveTextContent(SECOND);
  });
});

describe("an entry in place", () => {
  it("offers the action its placing control supplies, which the same entry in the area does not", () => {
    const reload = vi.fn();
    render(
      <>
        <NotificationItem
          placement="in place"
          entryKey="annotate-reload:a.jpg"
          afterDismiss={() => {}}
          action={{ label: "Reload", run: reload }}
        />
        <NotificationsPane />
      </>,
    );
    act(() =>
      useStore.getState().raiseNotification({
        kind: "failure",
        level: "error",
        text: { sentence: "The labels changed since they loaded." },
        context: { project: "valley", target: "a.jpg" },
        key: "annotate-reload:a.jpg",
      }),
    );

    fireEvent.click(
      within(screen.getByTestId("entry-in-place")).getByRole("button", { name: "Reload" }),
    );
    expect(reload).toHaveBeenCalledTimes(1);
    expect(within(onlyAreaEntry()).queryByRole("button", { name: "Reload" })).toBeNull();
  });
});
