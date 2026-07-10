import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

// xterm renders into canvas/DOM measurement APIs jsdom doesn't have — mock the
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
    attachCustomKeyEventHandler = vi.fn();
    onData = vi.fn(() => ({ dispose: vi.fn() }));
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
  readyState = 0;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: unknown }) => void) | null = null;
  onclose: (() => void) | null = null;
  send = vi.fn();
  close = vi.fn();
  constructor(public url: string) {
    MockWebSocket.instances.push(this);
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

  it("exposes a resize separator", async () => {
    render(<TerminalRail />);
    await screen.findByTestId("terminal-host");
    expect(screen.getByLabelText("Resize agent terminal")).toBeInTheDocument();
  });

  it("restart calls the API and resets the emulator", async () => {
    render(<TerminalRail />);
    await screen.findByTestId("terminal-host");
    await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalled());
    fireEvent.click(screen.getByText("Restart"));
    await waitFor(() => expect(terminalApi.restart).toHaveBeenCalledWith("t1", 30, 100));
    expect(termInstances[0].reset).toHaveBeenCalled();
  });
});
