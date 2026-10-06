import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

// jsdom lacks xterm's canvas and measurement APIs, so the emulator and its addons are mocked;
// vi.hoisted makes the class exist when the hoisted vi.mock factories run.
const { termInstances, MockTerminal } = vi.hoisted(() => {
  class MockTerminal {
    static instances: MockTerminal[] = [];
    rows = 30;
    cols = 100;
    unicode = { activeVersion: "6" };
    open = vi.fn();
    loadAddon = vi.fn();
    write = vi.fn();
    reset = vi.fn();
    dispose = vi.fn();
    focus = vi.fn();
    paste = vi.fn();
    getSelection = vi.fn(() => "");
    hasSelection = vi.fn(() => false);
    clearSelection = vi.fn();
    attachCustomKeyEventHandler = vi.fn();
    onData = vi.fn((_cb: (data: string) => void) => ({ dispose: vi.fn() }));
    onResize = vi.fn(() => ({ dispose: vi.fn() }));
    constructor() {
      MockTerminal.instances.push(this);
    }
  }
  return { termInstances: MockTerminal.instances, MockTerminal };
});
vi.mock("@xterm/xterm", () => ({ Terminal: MockTerminal }));
vi.mock("@xterm/addon-fit", () => ({
  FitAddon: class {
    fit = vi.fn();
  },
}));
vi.mock("@xterm/addon-unicode11", () => ({ Unicode11Addon: class {} }));
vi.mock("@xterm/addon-web-links", () => ({ WebLinksAddon: class {} }));
vi.mock("@xterm/xterm/css/xterm.css", () => ({}));

vi.mock("@/api/terminal", () => ({
  terminalApi: {
    status: vi.fn(),
    createSession: vi.fn(),
    restart: vi.fn(),
    submit: vi.fn(),
  },
  terminalWsUrl: (id: string) => `ws://test/api/terminal/ws/${id}`,
}));

import { terminalApi } from "@/api/terminal";
import { TerminalRail } from "@/components/TerminalRail";
import { useStore } from "@/store";

// jsdom lacks WebSocket and ResizeObserver.
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static OPEN = 1;
  static CLOSED = 3;
  readyState = 0;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: ((ev: { code: number }) => void) | null = null;
  send = vi.fn();
  close = vi.fn();
  constructor(public url: string) {
    MockWebSocket.instances.push(this);
  }

  /** Drop the socket (server-side or network); code 1008 mirrors the "unknown session" close. */
  drop(code = 1006) {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ code });
  }
}
vi.stubGlobal("WebSocket", MockWebSocket);
vi.stubGlobal(
  "ResizeObserver",
  class {
    observe = vi.fn();
    disconnect = vi.fn();
  },
);

const RITUAL = "[TCIP session-start ritual] Project: Demo. Run the ritual first.";
const PROVIDER = { id: "harness", name: "A harness", unavailable_reason: null };
const LAUNCHED = { provider: "harness", executable: "/bin/harness", version: null };
const LAUNCH = { session_id: "t1", existing: false, launched: LAUNCHED, ritual: RITUAL };

afterEach(cleanup);
beforeEach(() => {
  localStorage.clear();
  termInstances.length = 0;
  MockWebSocket.instances.length = 0;
  useStore.setState({ pendingTerminalMessages: [] });
  useStore.getState().setTerminalOpen(true);
  vi.mocked(terminalApi.status).mockResolvedValue({ providers: [PROVIDER] });
  vi.mocked(terminalApi.createSession).mockResolvedValue(LAUNCH);
  vi.mocked(terminalApi.restart).mockResolvedValue(LAUNCH);
  vi.mocked(terminalApi.submit).mockReset().mockResolvedValue({});
});

function submitted(): [string, string][] {
  return vi.mocked(terminalApi.submit).mock.calls.map(([id, { text }]) => [id, text]);
}

/** A promise and the function that settles it, for an answer the test releases itself. */
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve: (value: T) => void = () => {};
  const promise = new Promise<T>((settle) => (resolve = settle));
  return { promise, resolve };
}

/** Wait for the rail's `index`th socket and open it the way a live connection opens. */
async function openSocket(index = 0): Promise<MockWebSocket> {
  await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThan(index));
  const ws = MockWebSocket.instances[index];
  ws.readyState = MockWebSocket.OPEN;
  act(() => ws.onopen?.());
  return ws;
}

describe("TerminalRail", () => {
  it("renders nothing when the rail is closed", () => {
    useStore.getState().setTerminalOpen(false);
    const { container } = render(<TerminalRail />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the row's own reason when its harness is unavailable", async () => {
    vi.mocked(terminalApi.status).mockResolvedValue({
      providers: [{ ...PROVIDER, unavailable_reason: "A harness is not available: no `h`." }],
    });
    render(<TerminalRail />);
    expect(await screen.findByText(/A harness is not available/)).toBeInTheDocument();
    expect(terminalApi.createSession).not.toHaveBeenCalled();
  });

  it("a transient status failure is retryable, not latched", async () => {
    vi.mocked(terminalApi.status).mockRejectedValueOnce(new Error("backend down"));
    render(<TerminalRail />);
    expect(await screen.findByText(/Couldn't reach the TCIP backend/)).toBeInTheDocument();
    // Backend comes back; Retry re-probes and the terminal mounts.
    vi.mocked(terminalApi.status).mockResolvedValue({ providers: [PROVIDER] });
    fireEvent.click(screen.getByText("Retry"));
    expect(await screen.findByTestId("terminal-host")).toBeInTheDocument();
  });

  it("creates a session by the named person and attaches the terminal when available", async () => {
    useStore.setState({ user: "jordan" });
    render(<TerminalRail />);
    expect(await screen.findByTestId("terminal-host")).toBeInTheDocument();
    await waitFor(() =>
      expect(terminalApi.createSession).toHaveBeenCalledWith({
        provider: "harness",
        rows: 30,
        cols: 100,
        user: "jordan",
      }),
    );
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));
    expect(MockWebSocket.instances[0].url).toContain("/api/terminal/ws/t1");
    // The emulator was mounted into the host.
    expect(termInstances[0].open).toHaveBeenCalled();
  });

  it("minimize hides the rail (session survives server-side)", async () => {
    render(<TerminalRail />);
    await screen.findByTestId("terminal-host");
    fireEvent.click(screen.getByLabelText("Minimize agent terminal"));
    expect(useStore.getState().terminalOpen).toBe(false);
  });

  it("focuses the terminal and routes paste through term.paste (bracketed)", async () => {
    render(<TerminalRail />);
    const host = await screen.findByTestId("terminal-host");
    await waitFor(() => expect(termInstances[0].focus).toHaveBeenCalled());
    const evt = new Event("paste", {
      bubbles: true,
      cancelable: true,
    }) as unknown as ClipboardEvent;
    Object.defineProperty(evt, "clipboardData", { value: { getData: () => "pasted text" } });
    host.dispatchEvent(evt);
    expect(termInstances[0].paste).toHaveBeenCalledWith("pasted text");
  });

  it("copies the terminal selection to the clipboard when a drag-select ends", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true });
    render(<TerminalRail />);
    const host = await screen.findByTestId("terminal-host");
    await waitFor(() => expect(termInstances[0].focus).toHaveBeenCalled());
    const term = termInstances[0];
    term.hasSelection.mockReturnValue(true);
    term.getSelection.mockReturnValue("selected transcript text");
    host.dispatchEvent(new Event("mouseup", { bubbles: true }));
    expect(writeText).toHaveBeenCalledWith("selected transcript text");
  });

  it("exposes a resize separator", async () => {
    render(<TerminalRail />);
    await screen.findByTestId("terminal-host");
    expect(screen.getByLabelText("Resize agent terminal")).toBeInTheDocument();
  });

  describe("the harness picker", () => {
    const SECOND = { id: "other", name: "Another harness", unavailable_reason: null };
    const ABSENT = {
      id: "absent",
      name: "An absent harness",
      unavailable_reason: "An absent harness is not available: no `a` executable is on PATH.",
    };

    it("lists every row, an unavailable one disabled with its reason", async () => {
      vi.mocked(terminalApi.status).mockResolvedValue({ providers: [PROVIDER, SECOND, ABSENT] });
      render(<TerminalRail />);
      const picker = (await screen.findByLabelText("Agent harness")) as HTMLSelectElement;
      const options = Array.from(picker.options);
      expect(options.map((o) => [o.value, o.disabled])).toEqual([
        ["harness", false],
        ["other", false],
        ["absent", true],
      ]);
      expect(options[2].title).toBe(ABSENT.unavailable_reason);
      expect(picker.value).toBe("harness");
    });

    it("choosing another row restarts the live agent on it and remembers the choice", async () => {
      useStore.setState({ user: "jordan" });
      vi.mocked(terminalApi.status).mockResolvedValue({ providers: [PROVIDER, SECOND] });
      render(<TerminalRail />);
      await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalledTimes(1));
      fireEvent.change(await screen.findByLabelText("Agent harness"), {
        target: { value: "other" },
      });
      await waitFor(() =>
        expect(terminalApi.restart).toHaveBeenCalledWith("t1", {
          provider: "other",
          rows: 30,
          cols: 100,
          user: "jordan",
        }),
      );
      expect(terminalApi.createSession).toHaveBeenCalledTimes(1);
      expect(localStorage.getItem("tcip.terminal_provider")).toBe("other");
    });

    it("launches the remembered row when the table still lists it", async () => {
      localStorage.setItem("tcip.terminal_provider", "other");
      vi.mocked(terminalApi.status).mockResolvedValue({ providers: [PROVIDER, SECOND] });
      render(<TerminalRail />);
      await waitFor(() =>
        expect(terminalApi.createSession).toHaveBeenCalledWith(
          expect.objectContaining({ provider: "other" }),
        ),
      );
    });

    it("launches the first row that can when the remembered one cannot", async () => {
      localStorage.setItem("tcip.terminal_provider", "absent");
      vi.mocked(terminalApi.status).mockResolvedValue({ providers: [ABSENT, SECOND] });
      render(<TerminalRail />);
      await waitFor(() =>
        expect(terminalApi.createSession).toHaveBeenCalledWith(
          expect.objectContaining({ provider: "other" }),
        ),
      );
      expect(screen.queryByText(/is not available/)).not.toBeInTheDocument();
    });
  });

  it("restart calls the API with the terminal's dimensions and resets the emulator", async () => {
    useStore.setState({ user: "jordan" });
    render(<TerminalRail />);
    await screen.findByTestId("terminal-host");
    await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalled());
    termInstances[0].rows = 41;
    termInstances[0].cols = 133;
    fireEvent.click(screen.getByLabelText("Restart the agent"));
    await waitFor(() =>
      expect(terminalApi.restart).toHaveBeenCalledWith("t1", {
        provider: "harness",
        rows: 41,
        cols: 133,
        user: "jordan",
      }),
    );
    expect(termInstances[0].reset).toHaveBeenCalled();
  });

  describe("control frames sent to the PTY", () => {
    it("reports the emulator's rows and columns on attach, in that order", async () => {
      render(<TerminalRail />);
      const ws = await openSocket();
      // Rows and columns differ, so a frame that transposes them cannot still read correct.
      expect(ws.send).toHaveBeenCalledWith(JSON.stringify({ type: "resize", rows: 30, cols: 100 }));
    });

    it("forwards a keystroke as an input frame carrying the typed characters", async () => {
      render(<TerminalRail />);
      const ws = await openSocket();
      termInstances[0].onData.mock.calls[0][0]("ls\r");
      expect(ws.send).toHaveBeenCalledWith(JSON.stringify({ type: "input", data: "ls\r" }));
    });
  });

  describe("staged requests", () => {
    it("submits a staged request to the session, then clears it", async () => {
      render(<TerminalRail />);
      await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalled());

      act(() => useStore.getState().sendToAgentTerminal("run a sweep over lr and batch size"));
      await waitFor(() =>
        expect(submitted()).toEqual([["t1", "run a sweep over lr and batch size"]]),
      );
      expect(useStore.getState().pendingTerminalMessages).toEqual([]);
    });

    it("opens a closed rail and submits the request once its session exists", async () => {
      useStore.getState().setTerminalOpen(false);
      render(<TerminalRail />);
      expect(screen.queryByTestId("terminal-host")).not.toBeInTheDocument();

      act(() => useStore.getState().sendToAgentTerminal("run a sweep"));
      expect(useStore.getState().terminalOpen).toBe(true);
      await waitFor(() => expect(submitted()).toEqual([["t1", "run a sweep"]]));
      expect(useStore.getState().pendingTerminalMessages).toEqual([]);
    });

    it("keeps a request staged until the backend takes it", async () => {
      const answer = deferred<Record<string, never>>();
      vi.mocked(terminalApi.submit).mockReturnValueOnce(answer.promise);
      render(<TerminalRail />);
      act(() => useStore.getState().sendToAgentTerminal("after reopen"));
      await waitFor(() => expect(submitted()).toHaveLength(1));
      expect(useStore.getState().pendingTerminalMessages).toEqual(["after reopen"]);

      await act(async () => answer.resolve({}));
      expect(useStore.getState().pendingTerminalMessages).toEqual([]);
    });

    it("holds a request staged during a restart until the restart answers", async () => {
      const answer = deferred<typeof LAUNCH>();
      vi.mocked(terminalApi.restart).mockReturnValueOnce(answer.promise);
      render(<TerminalRail />);
      await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalled());

      fireEvent.click(screen.getByLabelText("Restart the agent"));
      act(() => useStore.getState().sendToAgentTerminal("after restart"));
      await act(async () => {});
      expect(submitted()).toEqual([]);

      await act(async () => answer.resolve(LAUNCH));
      await waitFor(() => expect(submitted()).toEqual([["t1", "after restart"]]));
    });

    it("submits a request once to the session two overlapping creates across a reopen share", async () => {
      const first = deferred<typeof LAUNCH>();
      const second = deferred<typeof LAUNCH>();
      vi.mocked(terminalApi.createSession)
        .mockReturnValueOnce(first.promise)
        .mockReturnValueOnce(second.promise);
      render(<TerminalRail />);
      await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalledTimes(1));
      act(() => useStore.getState().setTerminalOpen(false));
      act(() => useStore.getState().sendToAgentTerminal("tab request"));
      await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalledTimes(2));

      await act(async () => first.resolve(LAUNCH));
      expect(submitted()).toEqual([]);
      await act(async () => second.resolve({ ...LAUNCH, existing: true }));
      await waitFor(() => expect(submitted()).toEqual([["t1", "tab request"]]));
    });
  });

  describe("session-start ritual", () => {
    it("prints the launch's ritual and hides it once the breeder types", async () => {
      render(<TerminalRail />);
      expect(await screen.findByTestId("terminal-ritual")).toHaveTextContent(RITUAL);
      await waitFor(() => expect(termInstances[0].onData).toHaveBeenCalled());

      act(() => termInstances[0].onData.mock.calls[0][0]("h"));
      await waitFor(() => expect(screen.queryByTestId("terminal-ritual")).not.toBeInTheDocument());
    });

    it("prints a restart's ritual", async () => {
      vi.mocked(terminalApi.restart).mockResolvedValue({ ...LAUNCH, ritual: "restarted ritual" });
      render(<TerminalRail />);
      await screen.findByTestId("terminal-ritual");

      fireEvent.click(screen.getByLabelText("Restart the agent"));
      await waitFor(() =>
        expect(screen.getByTestId("terminal-ritual")).toHaveTextContent("restarted ritual"),
      );
    });
  });

  describe("reconnect after a drop", () => {
    it("reconnects with backoff, keeping the same session for an ordinary drop", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      try {
        render(<TerminalRail />);
        const first = await openSocket();
        first.drop(1006);
        await act(async () => {
          await vi.advanceTimersByTimeAsync(500);
        });
        expect(MockWebSocket.instances).toHaveLength(2);
        expect(terminalApi.createSession).toHaveBeenCalledTimes(1);
      } finally {
        vi.useRealTimers();
      }
    });

    it("invalidates the session on a close-before-open or code 1008, so the next attempt re-creates one", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      try {
        render(<TerminalRail />);
        await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));
        MockWebSocket.instances[0].drop(1008); // never opened

        await act(async () => {
          await vi.advanceTimersByTimeAsync(500);
        });
        expect(MockWebSocket.instances).toHaveLength(2);
        expect(terminalApi.createSession).toHaveBeenCalledTimes(2);
      } finally {
        vi.useRealTimers();
      }
    });
  });
});
