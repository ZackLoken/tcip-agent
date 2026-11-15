import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { sessionsApi } from "@/api/sessions";
import { CONNECTION_LOST, StateSocket } from "@/api/ws";
import {
  NotificationsButton,
  NotificationsLiveRegion,
  NotificationsPane,
} from "@/components/Notifications";
import { postHeldContributions } from "@/lib/sessionLifecycle";
import { useStore } from "@/store";
import { PRODUCED_ENVELOPE, openTestProject } from "@/test/store";
import { FakeWebSocket, lastSocket } from "@/test/webSocket";

const initialStoreState = useStore.getState();

/** The notifications host as App mounts it, under strict mode as `main.tsx` renders it. */
function renderHost(onClose = () => {}) {
  return render(
    <StrictMode>
      <NotificationsButton open={false} onToggle={() => {}} />
      <NotificationsPane onClose={onClose} />
      <NotificationsLiveRegion />
    </StrictMode>,
  );
}

/** Every text the assertive live region has been given since `region` began being watched. */
function announcements(region: HTMLElement): string[] {
  const told: string[] = [];
  new MutationObserver((records) => {
    for (const record of records) {
      record.addedNodes.forEach((node) => told.push(node.textContent ?? ""));
    }
  }).observe(region, { childList: true, subtree: true });
  return told;
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
    renderHost();
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
    expect(within(screen.getByRole("listitem")).getByText(/ended/)).toBeInTheDocument();
    socket.close();
  });
});

describe("the stale bundle", () => {
  const SENTENCE = "The frontend bundle was not built from the source; run the build.";

  it("is an entry carrying the backend's sentence, and ends once a frame states none", () => {
    vi.spyOn(useStore.getState(), "mergeSnapshot").mockImplementation(() => {});
    vi.stubGlobal("WebSocket", FakeWebSocket);
    renderHost();
    const socket = new StateSocket();
    socket.connect();
    act(() => lastSocket().open());

    act(() => lastSocket().message(JSON.stringify({ ...PRODUCED_ENVELOPE, bundle: SENTENCE })));
    expect(within(screen.getByRole("listitem")).getByText(SENTENCE)).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent(SENTENCE);
    expect(screen.getByTestId("condition-mark")).toBeInTheDocument();

    act(() => lastSocket().message(JSON.stringify(PRODUCED_ENVELOPE)));
    expect(screen.queryByTestId("condition-mark")).not.toBeInTheDocument();
    expect(screen.getByText(/ended/)).toBeInTheDocument();
    socket.close();
  });
});

describe("the unconfirmed visits", () => {
  it("dismissed, stay away on the next failed send while visits stay unconfirmed, and the mark still shows", async () => {
    vi.spyOn(sessionsApi, "imageEvent").mockRejectedValue(new TypeError("Failed to fetch"));
    useStore.getState().setUser("jordan");
    openTestProject();
    renderHost();
    const s = useStore.getState();
    s.startImageSessionTracking("a.jpg", Date.now() - 2000);
    s.closeSessionInterval();
    postHeldContributions();
    await act(flush);
    const entry = screen.getByRole("listitem");
    expect(entry).toHaveTextContent(/Could not confirm these visits were recorded.*a\.jpg/);

    fireEvent.click(within(entry).getByRole("button", { name: /^Dismiss: / }));
    s.startImageSessionTracking("b.jpg", Date.now() - 2000);
    s.closeSessionInterval();
    postHeldContributions();
    await act(flush);

    expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(3);
    expect(screen.queryByRole("listitem")).not.toBeInTheDocument();
    expect(screen.getByText(/Nothing to report/)).toBeInTheDocument();
    expect(screen.getByTestId("condition-mark")).toBeInTheDocument();
  });
});

describe("the notifications area", () => {
  it("counts an entry new until the area is seen, and closes on Escape", () => {
    const onClose = vi.fn();
    renderHost(onClose);
    act(() => useStore.getState().holdCondition("bundle", "rebuild the bundle"));
    expect(screen.getByRole("button", { name: /^Notifications, 1 new/ })).toBeInTheDocument();

    act(() => useStore.getState().markNotificationsSeen());
    expect(screen.getByRole("button", { name: /^Notifications, a problem/ })).toBeInTheDocument();

    fireEvent.keyDown(screen.getByRole("heading", { name: "Notifications" }), { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
