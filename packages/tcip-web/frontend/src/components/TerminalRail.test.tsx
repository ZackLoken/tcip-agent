import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

// xterm renders into canvas/DOM measurement APIs jsdom doesn't have; mock the
// emulator and its addons; the rail's own logic (status gate, session wiring,
// open/close) is what these tests pin. vi.hoisted so the class exists when the
// hoisted vi.mock factories run.
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
    scrollLines = vi.fn();
    input = vi.fn();
    parser = { registerCsiHandler: vi.fn(() => ({ dispose: vi.fn() })) };
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
    restart: vi.fn().mockResolvedValue({ session_id: "t1", alive: true }),
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

afterEach(cleanup);
beforeEach(() => {
  termInstances.length = 0;
  MockWebSocket.instances.length = 0;
  useStore.getState().setTerminalOpen(true);
  vi.mocked(terminalApi.status).mockResolvedValue({ available: true });
  vi.mocked(terminalApi.createSession).mockResolvedValue({ session_id: "t1", existing: false });
});

describe("TerminalRail", () => {
  it("renders nothing when the rail is closed", () => {
    useStore.getState().setTerminalOpen(false);
    const { container } = render(<TerminalRail />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the unconfigured state when Claude Code is unavailable", async () => {
    vi.mocked(terminalApi.status).mockResolvedValue({
      available: false,
      reason: "Claude Code is not available.",
    });
    render(<TerminalRail />);
    expect(await screen.findByText(/Claude Code is not available/)).toBeInTheDocument();
    expect(terminalApi.createSession).not.toHaveBeenCalled();
  });

  it("a transient status failure is retryable, not latched", async () => {
    vi.mocked(terminalApi.status).mockRejectedValueOnce(new Error("backend down"));
    render(<TerminalRail />);
    expect(await screen.findByText(/Couldn't reach the TCIP backend/)).toBeInTheDocument();
    // Backend comes back; Retry re-probes and the terminal mounts.
    vi.mocked(terminalApi.status).mockResolvedValue({ available: true });
    fireEvent.click(screen.getByText("Retry"));
    expect(await screen.findByTestId("terminal-host")).toBeInTheDocument();
  });

  it("creates a session and attaches the terminal when available", async () => {
    render(<TerminalRail />);
    expect(await screen.findByTestId("terminal-host")).toBeInTheDocument();
    await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalled());
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

  it("forwards the wheel as SGR mouse-scroll events so Claude scrolls its conversation", async () => {
    render(<TerminalRail />);
    const host = await screen.findByTestId("terminal-host");
    await waitFor(() => expect(termInstances[0].focus).toHaveBeenCalled());
    host.dispatchEvent(new WheelEvent("wheel", { deltaY: -120, bubbles: true, cancelable: true }));
    // wheel-up encodes SGR button 64; Claude receives it as a scroll, not a cursor key
    expect(termInstances[0].input).toHaveBeenCalledWith(expect.stringContaining("<64;"));
  });

  it("exposes a resize separator", async () => {
    render(<TerminalRail />);
    await screen.findByTestId("terminal-host");
    expect(screen.getByLabelText("Resize agent terminal")).toBeInTheDocument();
  });

  it("restart calls the API and resets the emulator", async () => {
    render(<TerminalRail />);
    await screen.findByTestId("terminal-host");
    await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalled());
    fireEvent.click(screen.getByLabelText("Restart the agent"));
    await waitFor(() => expect(terminalApi.restart).toHaveBeenCalledWith("t1", 30, 100));
    expect(termInstances[0].reset).toHaveBeenCalled();
  });

  describe("control frames sent to the PTY", () => {
    async function openSocket() {
      render(<TerminalRail />);
      await screen.findByTestId("terminal-host");
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));
      const ws = MockWebSocket.instances[0];
      ws.readyState = MockWebSocket.OPEN;
      ws.onopen?.();
      return ws;
    }

    it("reports the emulator's rows and columns on attach, in that order", async () => {
      const ws = await openSocket();
      // Rows and columns differ, so a frame that transposes them cannot still read correct.
      expect(ws.send).toHaveBeenCalledWith(JSON.stringify({ type: "resize", rows: 30, cols: 100 }));
    });

    it("forwards a keystroke as an input frame carrying the typed characters", async () => {
      const ws = await openSocket();
      termInstances[0].onData.mock.calls[0][0]("ls\r");
      expect(ws.send).toHaveBeenCalledWith(JSON.stringify({ type: "input", data: "ls\r" }));
    });
  });

  describe("sendToAgentTerminal hand-off", () => {
    it("sends a staged message as terminal input once the socket is open, then clears it", async () => {
      render(<TerminalRail />);
      await screen.findByTestId("terminal-host");
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));
      const ws = MockWebSocket.instances[0];
      ws.readyState = MockWebSocket.OPEN;
      ws.onopen?.();

      useStore.getState().sendToAgentTerminal("run a sweep over lr and batch size");
      await waitFor(() =>
        expect(ws.send).toHaveBeenCalledWith(
          JSON.stringify({ type: "input", data: "run a sweep over lr and batch size\r" }),
        ),
      );
      expect(useStore.getState().pendingTerminalMessage).toBeNull();
    });

    it("opens a closed rail and delivers the message once it connects, instead of dropping it", async () => {
      useStore.getState().setTerminalOpen(false);
      render(<TerminalRail />);
      expect(screen.queryByTestId("terminal-host")).not.toBeInTheDocument();

      useStore.getState().sendToAgentTerminal("run a sweep");
      expect(useStore.getState().terminalOpen).toBe(true);

      await screen.findByTestId("terminal-host");
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));
      const ws = MockWebSocket.instances[0];
      // Not open yet: the message must still be staged, not silently lost.
      expect(useStore.getState().pendingTerminalMessage).toBe("run a sweep");

      ws.readyState = MockWebSocket.OPEN;
      ws.onopen?.();
      await waitFor(() =>
        expect(ws.send).toHaveBeenCalledWith(
          JSON.stringify({ type: "input", data: "run a sweep\r" }),
        ),
      );
      expect(useStore.getState().pendingTerminalMessage).toBeNull();
    });
  });

  describe("starter hint", () => {
    it("shows a starter hint when no project is open", async () => {
      render(<TerminalRail />);
      await screen.findByTestId("terminal-host");
      expect(await screen.findByTestId("terminal-starter-hint")).toBeInTheDocument();
    });

    it("hides the starter hint once the breeder sends input", async () => {
      render(<TerminalRail />);
      await screen.findByTestId("terminal-host");
      await screen.findByTestId("terminal-starter-hint");
      termInstances[0].onData.mock.calls[0][0]("h");
      await waitFor(() =>
        expect(screen.queryByTestId("terminal-starter-hint")).not.toBeInTheDocument(),
      );
    });

    it("does not show the starter hint when a project is already open", async () => {
      const dataset = useStore.getState().gui.dataset;
      useStore.getState().patchGui({
        dataset: { ...dataset, dataset_root: "/workspace/demo_trait_site", date: "2026-05-01" },
      });
      try {
        render(<TerminalRail />);
        await screen.findByTestId("terminal-host");
        expect(screen.queryByTestId("terminal-starter-hint")).not.toBeInTheDocument();
      } finally {
        // Restore, so a later test in this file can't inherit "a project is open".
        useStore.getState().patchGui({ dataset });
      }
    });

    it("does not show the starter hint for a project open with no dated images yet", async () => {
      const dataset = useStore.getState().gui.dataset;
      useStore.getState().patchGui({
        dataset: { ...dataset, dataset_root: "/workspace/demo_trait_site", date: null },
      });
      try {
        render(<TerminalRail />);
        await screen.findByTestId("terminal-host");
        expect(screen.queryByTestId("terminal-starter-hint")).not.toBeInTheDocument();
      } finally {
        useStore.getState().patchGui({ dataset });
      }
    });
  });

  describe("reconnect after a drop", () => {
    it("reconnects with backoff, keeping the same session for an ordinary drop", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      try {
        render(<TerminalRail />);
        await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));
        const first = MockWebSocket.instances[0];
        first.readyState = MockWebSocket.OPEN;
        first.onopen?.();

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
