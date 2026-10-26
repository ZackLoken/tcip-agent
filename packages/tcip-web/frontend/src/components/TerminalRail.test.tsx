import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, onTestFinished, vi } from "vitest";

// jsdom lacks xterm's canvas and measurement APIs, so the emulator and its addons are mocked unless
// a test sets `emulator.real`; vi.hoisted makes these exist when the hoisted vi.mock factories run.
const { termInstances, MockTerminal, painted, emulator } = vi.hoisted(() => {
  // The pixel width the mock paints its screen at: 8 px per column, or 0 for an unpainted one.
  const painted = { columnPx: 8 };
  class MockTerminal {
    static instances: MockTerminal[] = [];
    rows = 30;
    cols = 100;
    unicode = { activeVersion: "6" };
    open = vi.fn((host: HTMLElement) => {
      const screen = document.createElement("div");
      screen.className = "xterm-screen";
      screen.getBoundingClientRect = () =>
        ({ width: painted.columnPx * this.cols, height: 0 }) as DOMRect;
      host.appendChild(screen);
    });
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
  // Whether the rail builds a real xterm and its real addons, and the real terminal it built.
  const emulator: { real: boolean; term: import("@xterm/xterm").Terminal | null } = {
    real: false,
    term: null,
  };
  return { termInstances: MockTerminal.instances, MockTerminal, painted, emulator };
});
vi.mock("@xterm/xterm", async (importOriginal) => {
  const xterm = await importOriginal<typeof import("@xterm/xterm")>();
  return {
    Terminal: function (options: import("@xterm/xterm").ITerminalOptions) {
      if (!emulator.real) return new MockTerminal();
      emulator.term = new xterm.Terminal(options);
      return emulator.term;
    },
  };
});
vi.mock("@xterm/addon-fit", async (importOriginal) => {
  const { FitAddon } = await importOriginal<typeof import("@xterm/addon-fit")>();
  return {
    FitAddon: function () {
      return emulator.real ? new FitAddon() : { fit: vi.fn() };
    },
  };
});
vi.mock("@xterm/addon-unicode11", async (importOriginal) => {
  const { Unicode11Addon } = await importOriginal<typeof import("@xterm/addon-unicode11")>();
  return {
    Unicode11Addon: function () {
      return emulator.real ? new Unicode11Addon() : {};
    },
  };
});
vi.mock("@xterm/addon-web-links", async (importOriginal) => {
  const { WebLinksAddon } = await importOriginal<typeof import("@xterm/addon-web-links")>();
  return {
    WebLinksAddon: function () {
      return emulator.real ? new WebLinksAddon() : {};
    },
  };
});
vi.mock("@xterm/xterm/css/xterm.css", () => ({}));

vi.mock("@/api/terminal", () => ({
  terminalApi: {
    status: vi.fn(),
    createSession: vi.fn(),
    end: vi.fn(),
    handOff: vi.fn(),
    submit: vi.fn(),
  },
  terminalWsUrl: (id: string) => `ws://test/api/terminal/ws/${id}`,
}));

import { StructuredRefusalError } from "@/api/http";
import { terminalApi } from "@/api/terminal";
import { TERMINAL_ENDED_CLOSE_CODE } from "@/api/types.generated";
import { TerminalRail } from "@/components/TerminalRail";
import { MAX_BACKOFF_MS } from "@/lib/reconnectingSocket";
import { useStore } from "@/store";

/** The browser's WebSocket as the rail's tests stand it in, so no test opens a connection. */
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

  /** Drop the socket (server-side or network) with close `code`. */
  drop(code = 1006) {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.({ code });
  }
}

const PROVIDER = { id: "harness", name: "A harness", unavailable_reason: null };
const LAUNCHED = {
  provider: "harness",
  argv: ["/bin/harness"],
  version: null,
  confinement: null,
  prepared: [],
  resumed: null,
};
const LAUNCH = { session_id: "t1", launched: LAUNCHED };

type Globals = { IS_REACT_ACT_ENVIRONMENT?: boolean };

// Read before any test overrides them.
const JSDOM_PLATFORM = navigator.platform;
const ACT_ENVIRONMENT = (globalThis as Globals).IS_REACT_ACT_ENVIRONMENT;

afterEach(cleanup);
beforeEach(() => {
  localStorage.clear();
  termInstances.length = 0;
  emulator.real = false;
  emulator.term = null;
  MockWebSocket.instances.length = 0;
  vi.stubGlobal("WebSocket", MockWebSocket);
  // The matchMedia a real xterm's open reads, which jsdom lacks.
  vi.stubGlobal("matchMedia", (media: string) => ({
    matches: false,
    media,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
  }));
  useStore.setState({
    pendingTerminalMessages: [],
    terminalPhase: "live",
    terminalProvider: null,
  });
  useStore.getState().setTerminalOpen(true);
  vi.mocked(terminalApi.status).mockResolvedValue({ providers: [PROVIDER] });
  vi.mocked(terminalApi.createSession).mockResolvedValue(LAUNCH);
  vi.mocked(terminalApi.end).mockResolvedValue({ undelivered: [] });
  vi.mocked(terminalApi.handOff).mockReset().mockResolvedValue(null);
  vi.mocked(terminalApi.submit).mockReset().mockResolvedValue({});
});

/** Ask to end the agent from the header and confirm it. */
function exitAgent() {
  fireEvent.click(screen.getByLabelText("End the agent"));
  fireEvent.click(screen.getByRole("button", { name: "End" }));
}

function submitted(): [string, string][] {
  return vi.mocked(terminalApi.submit).mock.calls.map(([id, { text }]) => [id, text]);
}

/** Give the browser `clipboard` for the running test, removed when the test finishes. */
function withClipboard(clipboard: Partial<Clipboard>) {
  Object.defineProperty(navigator, "clipboard", { value: clipboard, configurable: true });
  onTestFinished(() => void Reflect.deleteProperty(navigator, "clipboard"));
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

  describe("pasting into a real xterm", () => {
    const KEYBOARD_PASTES: KeyboardEventInit[] = [
      { key: "v", keyCode: 86, ctrlKey: true },
      { key: "V", keyCode: 86, ctrlKey: true, shiftKey: true },
      { key: "Insert", keyCode: 45, shiftKey: true },
    ];

    /** Have the browser report `platform` as the platform the rail runs on. */
    function onPlatform(platform: string) {
      vi.spyOn(navigator, "platform", "get").mockReturnValue(platform);
    }

    beforeEach(() => {
      emulator.real = true;
      onPlatform("Win32");
    });

    /** Render the rail live, with xterm's input textarea and the input frames sent the program. */
    async function mounted() {
      render(<TerminalRail />);
      const host = await screen.findByTestId("terminal-host");
      const ws = await openSocket();
      const input = () =>
        ws.send.mock.calls
          .map(([frame]) => JSON.parse(frame as string) as { type: string; data?: string })
          .filter((frame) => frame.type === "input")
          .map((frame) => frame.data);
      return { host, ws, textarea: host.querySelector("textarea")!, input };
    }

    /** Have the program set a DEC private mode, and wait until xterm has applied it. */
    async function programSets(ws: MockWebSocket, sequence: string, applied: () => boolean) {
      act(() => ws.onmessage?.({ data: sequence }));
      await waitFor(() => expect(applied()).toBe(true));
    }

    /** Press the key `init` describes in xterm's textarea; the keydown event is returned. */
    function press(textarea: HTMLTextAreaElement, init: KeyboardEventInit): KeyboardEvent {
      const key = new KeyboardEvent("keydown", { ...init, bubbles: true, cancelable: true });
      textarea.dispatchEvent(key);
      return key;
    }

    it("hands the keyboard pastes to the browser, whose paste event xterm sends once", async () => {
      const { ws, textarea, input } = await mounted();
      for (const init of KEYBOARD_PASTES)
        expect(press(textarea, init).defaultPrevented).toBe(false);
      expect(input()).toEqual([]);
      await programSets(ws, "\x1b[?2004h", () => emulator.term!.modes.bracketedPasteMode);
      const paste = new Event("paste", { bubbles: true, cancelable: true }) as ClipboardEvent;
      Object.defineProperty(paste, "clipboardData", { value: { getData: () => "docs/run.md" } });
      textarea.dispatchEvent(paste);
      expect(paste.defaultPrevented).toBe(false);
      expect(input()).toEqual(["\x1b[200~docs/run.md\x1b[201~"]);
    });

    it("on a Mac hands Cmd+V to the browser and sends Ctrl+V to the program", async () => {
      onPlatform("MacIntel");
      const { textarea, input } = await mounted();
      expect(press(textarea, { key: "v", keyCode: 86, metaKey: true }).defaultPrevented).toBe(
        false,
      );
      expect(input()).toEqual([]);
      press(textarea, { key: "v", keyCode: 86, ctrlKey: true });
      expect(input()).toEqual(["\x16"]);
    });

    it("reads no clipboard for a keyboard paste", async () => {
      const readText = vi.fn().mockResolvedValue("clipboard text");
      withClipboard({ readText });
      const { textarea } = await mounted();
      for (const init of KEYBOARD_PASTES) press(textarea, init);
      expect(readText).not.toHaveBeenCalled();
    });

    it("a right click pastes the clipboard only while the program has not taken the mouse", async () => {
      const readText = vi.fn().mockResolvedValue("clipboard text");
      withClipboard({ readText });
      const { host, ws, input } = await mounted();
      fireEvent.contextMenu(host);
      await waitFor(() => expect(input()).toEqual(["clipboard text"]));
      await programSets(ws, "\x1b[?1000h", () => emulator.term!.modes.mouseTrackingMode !== "none");
      fireEvent.contextMenu(host);
      expect(readText).toHaveBeenCalledTimes(1);
      expect(input()).toEqual(["clipboard text"]);
    });
  });

  it("copies the terminal selection to the clipboard when a drag-select ends", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    withClipboard({ writeText });
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

  it("resizes by one painted terminal column per arrow key and persists the width", async () => {
    painted.columnPx = 8;
    render(<TerminalRail />);
    await screen.findByTestId("terminal-host");
    await waitFor(() => expect(termInstances[0].open).toHaveBeenCalled());
    const handle = screen.getByLabelText("Resize agent terminal");
    const rail = screen.getByLabelText("Agent terminal");
    expect(rail.style.width).toBe("480px");
    fireEvent.keyDown(handle, { key: "ArrowLeft" });
    expect(rail.style.width).toBe("488px");
    expect(localStorage.getItem("tcip.terminal_width")).toBe("488");
    fireEvent.keyDown(handle, { key: "ArrowRight" });
    fireEvent.keyDown(handle, { key: "ArrowRight" });
    expect(rail.style.width).toBe("472px");
    expect(localStorage.getItem("tcip.terminal_width")).toBe("472");
  });

  it("a drag's release persists the width displayed then, a key resize during it included", async () => {
    painted.columnPx = 8;
    render(<TerminalRail />);
    await screen.findByTestId("terminal-host");
    await waitFor(() => expect(termInstances[0].open).toHaveBeenCalled());
    const handle = screen.getByLabelText("Resize agent terminal");
    const rail = screen.getByLabelText("Agent terminal");
    fireEvent.mouseDown(handle);
    fireEvent.mouseMove(window, { clientX: window.innerWidth - 560 });
    fireEvent.mouseMove(window, { clientX: window.innerWidth - 600 });
    expect(rail.style.width).toBe("600px");
    expect(localStorage.getItem("tcip.terminal_width")).toBeNull();
    fireEvent.mouseUp(window);
    expect(localStorage.getItem("tcip.terminal_width")).toBe("600");
    fireEvent.keyDown(handle, { key: "ArrowLeft" });
    expect(rail.style.width).toBe("608px");
    expect(localStorage.getItem("tcip.terminal_width")).toBe("608");

    fireEvent.mouseDown(handle);
    fireEvent.mouseMove(window, { clientX: window.innerWidth - 640 });
    expect(localStorage.getItem("tcip.terminal_width")).toBe("608");
    fireEvent.keyDown(handle, { key: "ArrowLeft" });
    expect(rail.style.width).toBe("648px");
    expect(localStorage.getItem("tcip.terminal_width")).toBe("648");
    fireEvent.mouseUp(window);
    expect(rail.style.width).toBe("648px");
    expect(localStorage.getItem("tcip.terminal_width")).toBe("648");
  });

  it("a release delivered as soon as the dragged width is committed persists that width", async () => {
    render(<TerminalRail />);
    await screen.findByTestId("terminal-host");
    const rail = screen.getByLabelText("Agent terminal");
    fireEvent.mouseDown(screen.getByLabelText("Resize agent terminal"));
    // A browser reports the move as the current event, which React schedules as continuous work;
    // jsdom still reports the mousedown, which React would render and flush synchronously.
    const move = new MouseEvent("mousemove", { clientX: window.innerWidth - 600 });
    vi.spyOn(window, "event", "get").mockReturnValue(move);
    // Each clock read passes React's 5 ms frame, so the scheduler yields after the commit.
    let now = performance.now();
    vi.spyOn(performance, "now").mockImplementation(() => (now += 6));
    vi.stubGlobal("IS_REACT_ACT_ENVIRONMENT", false);
    const released = new Promise<void>((resolve) => {
      const observer = new MutationObserver(() => {
        observer.disconnect();
        window.dispatchEvent(new MouseEvent("mouseup"));
        resolve();
      });
      observer.observe(rail, { attributeFilter: ["style"] });
    });
    window.dispatchEvent(move);
    await released;
    expect(rail.style.width).toBe("600px");
    expect(localStorage.getItem("tcip.terminal_width")).toBe("600");
  });

  // Runs after the tests above in file order and observes what they overrode: a stubbed global
  // (unstubGlobals), a spied platform (restoreMocks) and the test-scoped clipboard.
  it("finds every global the tests before it overrode back at its value from before them", () => {
    expect((globalThis as Globals).IS_REACT_ACT_ENVIRONMENT).toBe(ACT_ENVIRONMENT);
    expect(navigator.platform).toBe(JSDOM_PLATFORM);
    expect("clipboard" in navigator).toBe(false);
  });

  it("leaves the width alone when no column has been painted to measure", async () => {
    painted.columnPx = 0;
    render(<TerminalRail />);
    await screen.findByTestId("terminal-host");
    await waitFor(() => expect(termInstances[0].open).toHaveBeenCalled());
    fireEvent.keyDown(screen.getByLabelText("Resize agent terminal"), { key: "ArrowLeft" });
    expect(screen.getByLabelText("Agent terminal").style.width).toBe("480px");
    expect(localStorage.getItem("tcip.terminal_width")).toBeNull();
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

    /** Render the rail on two rows with its agent live, and choose the second while it runs. */
    async function switchedWhileRunning(): Promise<{
      picker: HTMLSelectElement;
      first: MockWebSocket;
    }> {
      vi.mocked(terminalApi.status).mockResolvedValue({ providers: [PROVIDER, SECOND] });
      render(<TerminalRail />);
      const first = await openSocket();
      const picker = (await screen.findByLabelText("Agent harness")) as HTMLSelectElement;
      expect(picker.title).toMatch(/asks the agent for its hand-off/);
      fireEvent.change(picker, { target: { value: "other" } });
      await waitFor(() => expect(terminalApi.handOff).toHaveBeenCalledTimes(1));
      return { picker, first };
    }

    it("choosing another row while an agent runs shows its hand-off, then starts the chosen row", async () => {
      useStore.setState({ user: "jordan" });
      const { picker, first } = await switchedWhileRunning();
      expect(terminalApi.handOff).toHaveBeenCalledWith("t1", { user: "jordan" });
      expect(picker.disabled).toBe(true);
      expect(picker.title).toMatch(/writing its hand-off/);
      expect(screen.getByLabelText("End the agent")).toBeInTheDocument();
      expect(terminalApi.end).not.toHaveBeenCalled();
      expect(terminalApi.createSession).toHaveBeenCalledTimes(1);

      act(() => first.drop(TERMINAL_ENDED_CLOSE_CODE));
      await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalledTimes(2));
      expect(vi.mocked(terminalApi.createSession).mock.calls[1][0]).toMatchObject({
        provider: "other",
      });
      expect(terminalApi.end).not.toHaveBeenCalled();
      expect(screen.queryByText("Start")).not.toBeInTheDocument();
      expect(localStorage.getItem("tcip.terminal_provider")).toBe("other");
    });

    it("End during a hand-off ends the agent at once and starts the chosen row", async () => {
      useStore.setState({ user: "jordan" });
      await switchedWhileRunning();
      exitAgent();
      await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalledTimes(2));
      expect(terminalApi.end).toHaveBeenCalledWith("t1", { user: "jordan" });
      expect(vi.mocked(terminalApi.createSession).mock.calls[1][0]).toMatchObject({
        provider: "other",
      });
      expect(screen.queryByText("Start")).not.toBeInTheDocument();
    });

    it("a minimized rail's switch starts the chosen row once it is open again", async () => {
      await switchedWhileRunning();
      act(() => useStore.getState().setTerminalOpen(false));
      act(() => useStore.getState().setTerminalOpen(true));
      const reopened = await openSocket(1);
      expect(reopened.url).toContain("/api/terminal/ws/t1");
      act(() => reopened.drop(TERMINAL_ENDED_CLOSE_CODE));
      await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalledTimes(2));
      expect(vi.mocked(terminalApi.createSession).mock.calls[1][0]).toMatchObject({
        provider: "other",
      });
    });

    it("a switch from a session the backend ended at once starts the chosen row with its requests", async () => {
      vi.mocked(terminalApi.handOff).mockResolvedValue({ undelivered: ["never given"] });
      vi.mocked(terminalApi.createSession)
        .mockResolvedValueOnce(LAUNCH)
        .mockResolvedValueOnce({ ...LAUNCH, session_id: "t2" });
      await switchedWhileRunning();
      await waitFor(() => expect(submitted()).toEqual([["t2", "never given"]]));
      expect(vi.mocked(terminalApi.createSession).mock.calls[1][0]).toMatchObject({
        provider: "other",
      });
    });

    it("a switch whose hand-off the backend refused stays on the row still running", async () => {
      vi.mocked(terminalApi.handOff).mockRejectedValue(
        new StructuredRefusalError({ message: "no such session" }, 404, "no such session"),
      );
      const { picker } = await switchedWhileRunning();
      expect(await screen.findByText(/no such session/)).toBeInTheDocument();
      expect(picker.value).toBe("harness");
      expect(picker.disabled).toBe(false);
      expect(useStore.getState().terminalProvider).toBe("harness");
      expect(terminalApi.createSession).toHaveBeenCalledTimes(1);
      expect(terminalApi.end).not.toHaveBeenCalled();
    });

    it("while an end is pending the picker is disabled and says the agent is ending", async () => {
      const answer = Promise.withResolvers<{ undelivered: string[] }>();
      vi.mocked(terminalApi.end).mockReturnValueOnce(answer.promise);
      vi.mocked(terminalApi.status).mockResolvedValue({ providers: [PROVIDER, SECOND] });
      render(<TerminalRail />);
      await openSocket();
      const picker = (await screen.findByLabelText("Agent harness")) as HTMLSelectElement;
      exitAgent();
      await act(async () => {});
      expect(picker.disabled).toBe(true);
      expect(picker.title).toMatch(/agent is ending/);

      fireEvent.change(picker, { target: { value: "other" } });
      await act(async () => {});
      expect(useStore.getState().terminalProvider).toBeNull();
      expect(localStorage.getItem("tcip.terminal_provider")).toBeNull();
      expect(terminalApi.end).toHaveBeenCalledTimes(1);
      expect(terminalApi.createSession).toHaveBeenCalledTimes(1);

      await act(() => Promise.resolve(answer.resolve({ undelivered: [] })));
      expect(picker.disabled).toBe(false);
      expect(picker.value).toBe("harness");
      expect(terminalApi.createSession).toHaveBeenCalledTimes(1);
    });

    it("choosing a row after an end only chooses what Start launches", async () => {
      vi.mocked(terminalApi.status).mockResolvedValue({ providers: [PROVIDER, SECOND] });
      render(<TerminalRail />);
      await openSocket();
      exitAgent();
      await screen.findByText("Start");
      fireEvent.change(screen.getByLabelText("Agent harness"), { target: { value: "other" } });
      await act(async () => {});
      expect(terminalApi.end).toHaveBeenCalledTimes(1);
      expect(terminalApi.createSession).toHaveBeenCalledTimes(1);

      fireEvent.click(screen.getByText("Start"));
      await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalledTimes(2));
      expect(vi.mocked(terminalApi.createSession).mock.calls[1][0]).toMatchObject({
        provider: "other",
      });
    });

    it("launches the remembered row when the table still lists it", async () => {
      useStore.setState({ terminalProvider: "other" });
      vi.mocked(terminalApi.status).mockResolvedValue({ providers: [PROVIDER, SECOND] });
      render(<TerminalRail />);
      await waitFor(() =>
        expect(terminalApi.createSession).toHaveBeenCalledWith(
          expect.objectContaining({ provider: "other" }),
        ),
      );
    });

    it("launches the first row that can when the remembered one cannot", async () => {
      useStore.setState({ terminalProvider: "absent" });
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

  describe("ending the agent", () => {
    it("the header holds the picker, the light, Minimize and Exit in that order", async () => {
      render(<TerminalRail />);
      await openSocket();
      const labelled = Array.from(
        screen.getByLabelText("Agent terminal").querySelectorAll("[aria-label]"),
      ).map((element) => element.getAttribute("aria-label"));
      expect(labelled).toEqual([
        "Resize agent terminal",
        "Agent harness",
        "Agent connected",
        "Minimize agent terminal",
        "End the agent",
      ]);
      expect(screen.queryByText(/restart/i)).not.toBeInTheDocument();
    });

    it("asks first, and Cancel leaves the agent running", async () => {
      render(<TerminalRail />);
      const ws = await openSocket();
      fireEvent.click(screen.getByLabelText("End the agent"));
      expect(screen.getByText("End the agent?")).toBeInTheDocument();
      expect(terminalApi.end).not.toHaveBeenCalled();

      fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
      expect(screen.getByLabelText("End the agent")).toBeInTheDocument();
      expect(terminalApi.end).not.toHaveBeenCalled();
      expect(ws.close).not.toHaveBeenCalled();
    });

    it("End ends the session by the person, stops the socket and offers Start", async () => {
      useStore.setState({ user: "jordan" });
      render(<TerminalRail />);
      const ws = await openSocket();
      exitAgent();
      expect(await screen.findByText("Start")).toBeInTheDocument();
      expect(terminalApi.end).toHaveBeenCalledWith("t1", { user: "jordan" });
      expect(ws.close).toHaveBeenCalled();
      expect(screen.queryByLabelText("End the agent")).not.toBeInTheDocument();
    });

    it("reopening the rail after an end creates no session", async () => {
      render(<TerminalRail />);
      await openSocket();
      exitAgent();
      await screen.findByText("Start");

      act(() => useStore.getState().setTerminalOpen(false));
      act(() => useStore.getState().setTerminalOpen(true));
      expect(await screen.findByText("Start")).toBeInTheDocument();
      await act(async () => {});
      expect(terminalApi.createSession).toHaveBeenCalledTimes(1);
      expect(MockWebSocket.instances).toHaveLength(1);
    });

    it("Start launches one fresh session through the socket's create", async () => {
      vi.mocked(terminalApi.createSession)
        .mockResolvedValueOnce(LAUNCH)
        .mockResolvedValueOnce({ ...LAUNCH, session_id: "t2" });
      render(<TerminalRail />);
      await openSocket();
      exitAgent();
      fireEvent.click(await screen.findByText("Start"));

      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2));
      expect(terminalApi.createSession).toHaveBeenCalledTimes(2);
      expect(MockWebSocket.instances[1].url).toContain("/api/terminal/ws/t2");
      expect(screen.queryByText("Start")).not.toBeInTheDocument();
    });

    it("an end whose line failed still stages what it answered and shows the gap", async () => {
      vi.mocked(terminalApi.end).mockRejectedValue(
        new StructuredRefusalError(
          {
            error: "audit_entry_not_written",
            message: "the end went unrecorded",
            committed: { undelivered: ["never given"] },
          },
          409,
          "the end went unrecorded",
        ),
      );
      render(<TerminalRail />);
      await openSocket();
      exitAgent();
      expect(await screen.findByText(/the end went unrecorded/)).toBeInTheDocument();
      expect(screen.getByText("Start")).toBeInTheDocument();
      expect(useStore.getState().pendingTerminalMessages).toEqual(["never given"]);
    });

    it("an end the backend could not carry out reattaches to the session it left running", async () => {
      vi.mocked(terminalApi.end).mockRejectedValue(
        new StructuredRefusalError(
          { message: "The agent process could not be stopped." },
          503,
          "The agent process could not be stopped.",
        ),
      );
      render(<TerminalRail />);
      await openSocket();
      exitAgent();
      expect(await screen.findByText(/could not be stopped/)).toBeInTheDocument();
      await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2));
      expect(MockWebSocket.instances[1].url).toContain("/api/terminal/ws/t1");
      expect(screen.queryByText("Start")).not.toBeInTheDocument();
      expect(terminalApi.createSession).toHaveBeenCalledTimes(1);
    });

    it("shows the agent ended and offers Start only once the end answers", async () => {
      const answer = Promise.withResolvers<{ undelivered: string[] }>();
      vi.mocked(terminalApi.end).mockReturnValueOnce(answer.promise);
      render(<TerminalRail />);
      await openSocket();
      exitAgent();
      await act(async () => {});
      expect(screen.queryByText("Start")).not.toBeInTheDocument();

      await act(() => Promise.resolve(answer.resolve({ undelivered: [] })));
      expect(screen.getByText("Start")).toBeInTheDocument();
      expect(terminalApi.createSession).toHaveBeenCalledTimes(1);
    });

    it("ends only once the submit in flight settles, keeping equal requests in their order", async () => {
      const taken = Promise.withResolvers<Record<string, never>>();
      vi.mocked(terminalApi.submit)
        .mockResolvedValueOnce({})
        .mockResolvedValueOnce({})
        .mockReturnValueOnce(taken.promise);
      vi.mocked(terminalApi.end).mockResolvedValue({
        undelivered: ["repeat", "middle", "repeat"],
      });
      render(<TerminalRail />);
      await openSocket();
      for (const [count, text] of ["repeat", "middle", "repeat"].entries()) {
        act(() => useStore.getState().sendToAgentTerminal(text));
        await waitFor(() => expect(submitted()).toHaveLength(count + 1));
      }

      exitAgent();
      await act(async () => {});
      expect(terminalApi.end).not.toHaveBeenCalled();
      await act(() => Promise.resolve(taken.resolve({})));
      await screen.findByText("Start");
      expect(useStore.getState().pendingTerminalMessages).toEqual(["repeat", "middle", "repeat"]);
    });

    it("an end the backend could not carry out resumes the staged requests too", async () => {
      const taken = Promise.withResolvers<Record<string, never>>();
      vi.mocked(terminalApi.submit).mockReturnValueOnce(taken.promise);
      vi.mocked(terminalApi.end).mockRejectedValue(
        new StructuredRefusalError({ message: "survived" }, 503, "survived"),
      );
      render(<TerminalRail />);
      await openSocket();
      act(() => useStore.getState().sendToAgentTerminal("first"));
      await waitFor(() => expect(submitted()).toEqual([["t1", "first"]]));
      act(() => useStore.getState().sendToAgentTerminal("second"));

      exitAgent();
      await act(() => Promise.resolve(taken.resolve({})));
      await waitFor(() =>
        expect(submitted()).toEqual([
          ["t1", "first"],
          ["t1", "second"],
        ]),
      );
      expect(useStore.getState().pendingTerminalMessages).toEqual([]);
    });

    it("reopening the rail while an End is unanswered creates no session", async () => {
      const answer = Promise.withResolvers<{ undelivered: string[] }>();
      vi.mocked(terminalApi.end).mockReturnValueOnce(answer.promise);
      render(<TerminalRail />);
      await openSocket();
      exitAgent();
      await act(async () => {});

      act(() => useStore.getState().setTerminalOpen(false));
      act(() => useStore.getState().setTerminalOpen(true));
      await screen.findByTestId("terminal-host");
      await act(async () => {});
      expect(terminalApi.createSession).toHaveBeenCalledTimes(1);
      expect(MockWebSocket.instances).toHaveLength(1);

      await act(() => Promise.resolve(answer.resolve({ undelivered: [] })));
      expect(screen.getByText("Start")).toBeInTheDocument();
      expect(terminalApi.createSession).toHaveBeenCalledTimes(1);
    });

    it("an end answered 404 shows the agent ended without reattaching", async () => {
      vi.mocked(terminalApi.end).mockRejectedValue(
        new StructuredRefusalError(
          { message: "no such terminal session: t1" },
          404,
          "no such terminal session: t1",
        ),
      );
      render(<TerminalRail />);
      await openSocket();
      exitAgent();
      expect(await screen.findByText("Start")).toBeInTheDocument();
      await act(async () => {});
      expect(screen.queryByText(/no such terminal session/)).not.toBeInTheDocument();
      expect(MockWebSocket.instances).toHaveLength(1);
    });

    it("a socket closed as ended shows Start, and no reconnect creates a session", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      try {
        render(<TerminalRail />);
        const ws = await openSocket();
        act(() => ws.drop(TERMINAL_ENDED_CLOSE_CODE));
        expect(await screen.findByText("Start")).toBeInTheDocument();
        await act(async () => {
          await vi.advanceTimersByTimeAsync(MAX_BACKOFF_MS);
        });
        expect(MockWebSocket.instances).toHaveLength(1);
        expect(terminalApi.createSession).toHaveBeenCalledTimes(1);
      } finally {
        vi.useRealTimers();
      }
    });
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
      const answer = Promise.withResolvers<Record<string, never>>();
      vi.mocked(terminalApi.submit).mockReturnValueOnce(answer.promise);
      render(<TerminalRail />);
      act(() => useStore.getState().sendToAgentTerminal("after reopen"));
      await waitFor(() => expect(submitted()).toHaveLength(1));
      expect(useStore.getState().pendingTerminalMessages).toEqual(["after reopen"]);

      await act(() => Promise.resolve(answer.resolve({})));
      expect(useStore.getState().pendingTerminalMessages).toEqual([]);
    });

    it("submits the requests an end answered, then those staged after it, once Start launches", async () => {
      vi.mocked(terminalApi.end).mockResolvedValue({ undelivered: ["never given"] });
      vi.mocked(terminalApi.createSession)
        .mockResolvedValueOnce(LAUNCH)
        .mockResolvedValueOnce({ ...LAUNCH, session_id: "t2" });
      render(<TerminalRail />);
      await openSocket();
      exitAgent();
      await screen.findByText("Start");
      act(() => useStore.getState().sendToAgentTerminal("staged after the end"));
      await act(async () => {});
      expect(submitted()).toEqual([]);
      expect(useStore.getState().pendingTerminalMessages).toEqual([
        "never given",
        "staged after the end",
      ]);

      fireEvent.click(screen.getByText("Start"));
      await waitFor(() =>
        expect(submitted()).toEqual([
          ["t2", "never given"],
          ["t2", "staged after the end"],
        ]),
      );
    });

    it("submits a request once to the session two overlapping creates across a reopen share", async () => {
      const first = Promise.withResolvers<typeof LAUNCH>();
      const second = Promise.withResolvers<typeof LAUNCH>();
      vi.mocked(terminalApi.createSession)
        .mockReturnValueOnce(first.promise)
        .mockReturnValueOnce(second.promise);
      render(<TerminalRail />);
      await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalledTimes(1));
      act(() => useStore.getState().setTerminalOpen(false));
      act(() => useStore.getState().sendToAgentTerminal("tab request"));
      await waitFor(() => expect(terminalApi.createSession).toHaveBeenCalledTimes(2));

      await act(() => Promise.resolve(first.resolve(LAUNCH)));
      expect(submitted()).toEqual([]);
      await act(() => Promise.resolve(second.resolve(LAUNCH)));
      await waitFor(() => expect(submitted()).toEqual([["t1", "tab request"]]));
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

    it("retries the same session after a close before open, creating none", async () => {
      vi.useFakeTimers({ shouldAdvanceTime: true });
      try {
        render(<TerminalRail />);
        await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));
        MockWebSocket.instances[0].drop(1006); // never opened: the backend was unreachable

        await act(async () => {
          await vi.advanceTimersByTimeAsync(500);
        });
        expect(MockWebSocket.instances).toHaveLength(2);
        expect(MockWebSocket.instances[1].url).toContain("/api/terminal/ws/t1");
        expect(terminalApi.createSession).toHaveBeenCalledTimes(1);
      } finally {
        vi.useRealTimers();
      }
    });
  });
});
