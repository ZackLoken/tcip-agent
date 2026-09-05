/**
 * The agent rail: the real Claude Code CLI, embedded. An xterm.js terminal attached
 * over WebSocket to a server-side PTY running `claude` in the repo root, so the breeder
 * talks to the same agent (CLAUDE.md, skills, MCP tools, permission prompts) the
 * platform is designed around: full fidelity, no translation layer. The terminal is
 * recolored to the field-station palette via the ANSI theme (Claude Code paints its TUI
 * with ANSI colors, so it renders in our palette natively); its internal layout is
 * untouched, that's the exact-experience point.
 */

import { useEffect, useRef, useState } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { Unicode11Addon } from "@xterm/addon-unicode11";
import { WebLinksAddon } from "@xterm/addon-web-links";
import "@xterm/xterm/css/xterm.css";

import { terminalApi, terminalWsUrl } from "@/api/terminal";
import type { TerminalInputFrame, TerminalResizeFrame } from "@/api/types.generated";
import { createReconnectingSocket } from "@/lib/reconnectingSocket";
import { useStore } from "@/store";
import { selectProjectOpen } from "@/store/slices/gui";

type TerminalSendFrame = TerminalInputFrame | TerminalResizeFrame;

const MIN_WIDTH = 320;
const DEFAULT_WIDTH = 480;
const WIDTH_KEY = "tcip.terminal_width";

function clampWidth(px: number): number {
  const max = Math.max(MIN_WIDTH, Math.round(window.innerWidth * 0.7));
  return Math.max(MIN_WIDTH, Math.min(Math.round(px), max));
}

/**
 * Field-station terminal theme. Claude Code draws with the 16 ANSI slots, so mapping
 * them to the app palette re-skins the real TUI: green → SI-green (the app accent), yellow
 * → late-summer gold, red → FP red, dim gray → sage. Background stays tcip-bg so the rail
 * sits flush with the app chrome; the cursor is persimmon, the accent the SeasonRail ends on.
 * The greens are lifted from the #507754 accent toward legibility so the TUI keeps contrast.
 */
const FIELD_STATION_THEME = {
  background: "#20211B", // tcip-panel: the warm bark surface used across the app chrome
  foreground: "#E7E5DC", // tcip-fg
  cursor: "#E6976B",
  cursorAccent: "#20211B",
  selectionBackground: "#50775455",
  black: "#33352C",
  red: "#EF5350",
  green: "#6E9A72", // SI-green, lightened for terminal contrast
  yellow: "#C9A24B",
  blue: "#7E9CB9",
  magenta: "#B48EAD",
  cyan: "#7FB0A9",
  white: "#E7E5DC",
  brightBlack: "#8C9082",
  brightRed: "#F28B82",
  brightGreen: "#8FC095", // brighter SI-green for the bold/success slot
  brightYellow: "#D9B96C",
  brightBlue: "#9DB8D2",
  brightMagenta: "#C9A9C0",
  brightCyan: "#9FC7C0",
  brightWhite: "#FFFFFF",
};

export function TerminalRail() {
  const open = useStore((s) => s.terminalOpen);
  const setOpen = useStore((s) => s.setTerminalOpen);
  const [status, setStatus] = useState<{ available: boolean; reason?: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Live PTY link state, surfaced as a header dot (green = attached, amber = (re)connecting).
  const [conn, setConn] = useState<"connecting" | "open" | "reconnecting">("connecting");
  const hostRef = useRef<HTMLDivElement | null>(null);
  const sessionRef = useRef<string | null>(null);
  const termRef = useRef<Terminal | null>(null);
  // The lifecycle effect's `send` closure, exposed so the pending-message effect below can
  // reach the live socket without a second write path.
  const sendRef = useRef<((payload: TerminalSendFrame) => void) | null>(null);

  // A request staged via `sendToAgentTerminal` (TuningTab, ResultsTab, ...): sent as terminal
  // input once the PTY socket is actually open, not before, so a message sent while the rail
  // is still (re)connecting isn't dropped.
  const pendingMessage = useStore((s) => s.pendingTerminalMessage);
  const clearPendingMessage = useStore((s) => s.clearPendingTerminalMessage);

  // Show a starter hint until a project is open (selectProjectOpen, the fact the footer also
  // names) or they've sent their first input; a first-time breeder otherwise meets a blank cursor.
  const projectOpen = useStore(selectProjectOpen);
  const [hasInput, setHasInput] = useState(false);
  const showStarterHint = !!status?.available && !projectOpen && !hasInput;

  const [width, setWidth] = useState<number>(() => {
    try {
      const v = parseInt(localStorage.getItem(WIDTH_KEY) ?? "", 10);
      return Number.isFinite(v) ? clampWidth(v) : DEFAULT_WIDTH;
    } catch {
      return DEFAULT_WIDTH;
    }
  });
  const widthRef = useRef(width);
  widthRef.current = width;

  function startResize(e: React.MouseEvent) {
    e.preventDefault();
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    // Rail is docked right, so its width grows as the pointer moves left.
    const onMove = (ev: MouseEvent) => setWidth(clampWidth(window.innerWidth - ev.clientX));
    const onUp = () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      try {
        localStorage.setItem(WIDTH_KEY, String(widthRef.current));
      } catch {
        /* width just won't persist */
      }
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  useEffect(() => {
    if (!open || status !== null) return;
    terminalApi
      .status()
      .then(setStatus)
      .catch(() =>
        // Backend unreachable is not "claude missing"; say so, and keep Retry viable.
        setStatus({
          available: false,
          reason: "Couldn't reach the TCIP backend. Is it running? Retry once it's up.",
        }),
      );
  }, [open, status]);

  // Terminal lifecycle: build xterm, attach the PTY WebSocket, wire input/resize.
  // Unmounting (rail closed) drops the socket; the server session (and Claude Code's
  // conversation) stays alive, and reopening replays the scrollback.
  useEffect(() => {
    if (!open || !status?.available || !hostRef.current) return;
    setConn("connecting");

    const term = new Terminal({
      theme: FIELD_STATION_THEME,
      fontFamily: "Consolas, Menlo, monospace",
      fontSize: 13,
      lineHeight: 1.15,
      cursorBlink: true,
      allowProposedApi: true,
      scrollback: 5000,
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.loadAddon(new Unicode11Addon());
    term.loadAddon(new WebLinksAddon());
    term.unicode.activeVersion = "11";
    const host = hostRef.current;
    term.open(host);
    fit.fit();
    term.focus();
    termRef.current = term;

    // Copy fix (root cause): Claude Code turns on SGR mouse tracking, which hands every mouse +
    // wheel event to it and disables xterm's own selection + scrollback. Swallow the mouse-mode
    // set/reset (DEC private 1000-1016) so tracking never engages and xterm stays a normal
    // selectable terminal: drag selects, the wheel scrolls the scrollback, drag-to-edge
    // auto-scrolls, a selection survives streaming output, and it works at the shell prompt after
    // /exit too. Claude is keyboard-driven; the wheel now scrolls the terminal scrollback instead
    // of Claude's view, which is what you want when copying.
    const MOUSE_MODES = new Set([1000, 1001, 1002, 1003, 1005, 1006, 1015, 1016]);
    const swallowMouseMode = (params: (number | number[])[]) => {
      for (const p of params) if (MOUSE_MODES.has(Array.isArray(p) ? p[0] : p)) return true;
      return false; // not a mouse mode (alt-screen, bracketed paste, cursor…), let xterm handle it
    };
    term.parser.registerCsiHandler({ prefix: "?", final: "h" }, swallowMouseMode);
    term.parser.registerCsiHandler({ prefix: "?", final: "l" }, swallowMouseMode);

    // Paste: Ctrl/Cmd+V fires a native paste event on xterm's hidden textarea, but the
    // browser doesn't always route it there (focus, or the app swallowing the key), so
    // Ctrl+V "did nothing". Intercept in the capture phase and hand the text to
    // term.paste(), which wraps it in bracketed-paste mode, so a multi-line paste
    // reaches Claude Code as one paste, not a line-per-Enter burst.
    const onPaste = (e: ClipboardEvent) => {
      const text = e.clipboardData?.getData("text");
      if (text) {
        term.paste(text);
        e.preventDefault();
        e.stopPropagation();
      }
    };

    // With mouse tracking suppressed above, xterm does normal selection, so the copy wiring is
    // simple: copy-on-select (a drag ending with a selection lands on the clipboard); right-click
    // copies the selection or else pastes (conhost style); Ctrl/Cmd+Shift+C / Ctrl+Insert copy;
    // Ctrl+Shift+V / Shift+Insert paste; and Ctrl+C copies when text is selected, else stays SIGINT.
    const copySelection = () => {
      const sel = term.getSelection();
      if (sel) void navigator.clipboard?.writeText(sel).catch(() => {});
    };
    const pasteClipboard = () => {
      void navigator.clipboard
        ?.readText()
        .then((t) => {
          if (t) term.paste(t);
        })
        .catch(() => {});
    };
    // Window-level: xterm finalizes drags via window listeners, so a release outside
    // the rail still completes the selection; a host listener would miss the copy.
    const onWindowMouseUp = () => {
      if (term.hasSelection()) copySelection();
    };
    const onContextMenu = (e: MouseEvent) => {
      e.preventDefault();
      if (term.hasSelection()) {
        copySelection();
        term.clearSelection();
      } else {
        pasteClipboard();
      }
    };
    term.attachCustomKeyEventHandler((e) => {
      if (e.type !== "keydown") return true;
      const mod = e.ctrlKey || e.metaKey;
      if (
        (mod && e.shiftKey && (e.key === "C" || e.key === "c")) ||
        (e.ctrlKey && e.key === "Insert")
      ) {
        e.preventDefault(); // otherwise Chrome also opens DevTools on Ctrl+Shift+C
        copySelection();
        return false;
      }
      if (
        e.ctrlKey &&
        !e.shiftKey &&
        !e.altKey &&
        !e.metaKey &&
        (e.key === "C" || e.key === "c") &&
        term.hasSelection()
      ) {
        copySelection(); // Windows-Terminal style: Ctrl+C copies a selection, else falls to SIGINT
        term.clearSelection();
        return false;
      }
      if (
        (mod && e.shiftKey && (e.key === "V" || e.key === "v")) ||
        (e.shiftKey && !mod && e.key === "Insert")
      ) {
        e.preventDefault();
        pasteClipboard();
        return false;
      }
      return true;
    });

    // Wheel scroll: mouse tracking is suppressed so drag selects locally, but Claude still expects
    // mouse reports (it enabled tracking; xterm just swallowed the enable), so forward the wheel as
    // SGR wheel events and Claude scrolls its own full-screen conversation (which has no xterm
    // scrollback to scroll). Without this the wheel falls through to xterm's alternate-screen
    // behavior and becomes cursor keys that Claude's prompt reads as history navigation.
    const onWheel = (e: WheelEvent) => {
      if (e.deltaY === 0) return;
      e.preventDefault();
      e.stopPropagation();
      const btn = e.deltaY < 0 ? 64 : 65; // SGR mouse: 64 = wheel up, 65 = wheel down
      const px =
        e.deltaMode === 1
          ? e.deltaY * 16
          : e.deltaMode === 2
            ? e.deltaY * term.rows * 16
            : e.deltaY;
      const ticks = Math.min(5, Math.max(1, Math.round(Math.abs(px) / 40)));
      for (let i = 0; i < ticks; i++) term.input(`\x1b[<${btn};1;1M`);
    };

    host.addEventListener("paste", onPaste, true);
    host.addEventListener("wheel", onWheel, { capture: true, passive: false });
    host.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("mouseup", onWindowMouseUp);

    // The URL provider re-checks closedByClient after the await, so a torn-down effect
    // (rail toggled, StrictMode remount) cannot attach a fresh socket to a disposed terminal.
    let closedByClient = false;
    const socket = createReconnectingSocket({
      url: async () => {
        try {
          if (!sessionRef.current) {
            const { session_id } = await terminalApi.createSession(term.rows, term.cols);
            sessionRef.current = session_id;
          }
        } catch (e) {
          setError(String(e));
          throw e;
        }
        if (closedByClient) throw new Error("terminal rail unmounted mid-connect");
        return terminalWsUrl(sessionRef.current);
      },
      onOpen: () => {
        setError(null);
        setConn("open");
        term.reset(); // the replay repaints the screen from scratch
        send({ type: "resize", rows: term.rows, cols: term.cols });
      },
      onMessage: (data) => term.write(data),
      // A close before open (or 1008 "unknown session") means the backend no longer knows this
      // session; retrying the dead id forever is the failure mode, so it is dropped here.
      onClose: (ev, opened) => {
        setConn("reconnecting");
        if (!opened || ev.code === 1008) sessionRef.current = null;
      },
    });

    const send = (payload: TerminalSendFrame) => socket.send(JSON.stringify(payload));
    sendRef.current = send;
    socket.start();

    const dataSub = term.onData((data) => {
      setHasInput(true);
      send({ type: "input", data });
    });
    const resizeSub = term.onResize(({ rows, cols }) => send({ type: "resize", rows, cols }));
    const observer = new ResizeObserver(() => {
      try {
        fit.fit();
      } catch {
        /* host momentarily zero-sized during layout */
      }
    });
    observer.observe(host);

    return () => {
      closedByClient = true;
      socket.stop();
      observer.disconnect();
      host.removeEventListener("paste", onPaste, true);
      host.removeEventListener("wheel", onWheel, true);
      host.removeEventListener("contextmenu", onContextMenu);
      window.removeEventListener("mouseup", onWindowMouseUp);
      dataSub.dispose();
      resizeSub.dispose();
      term.dispose();
      termRef.current = null;
      sendRef.current = null;
    };
  }, [open, status]);

  // Deliver a request staged via `sendToAgentTerminal`. Depends on `conn` (not just
  // `pendingMessage`) so a message staged before the socket finished (re)connecting is sent the
  // moment it opens, rather than being dropped or sent into a closed socket.
  useEffect(() => {
    if (!pendingMessage || conn !== "open") return;
    const send = sendRef.current;
    if (!send || !termRef.current) return;
    send({ type: "input", data: `${pendingMessage}\r` });
    setHasInput(true);
    clearPendingMessage();
  }, [pendingMessage, conn, clearPendingMessage]);

  async function restart() {
    const id = sessionRef.current;
    const term = termRef.current;
    if (!id) return;
    try {
      await terminalApi.restart(id, term?.rows ?? 30, term?.cols ?? 100);
      term?.reset();
      setHasInput(false); // a fresh conversation with still no project open re-shows the hint
    } catch (e) {
      setError(String(e));
    }
  }

  if (!open) return null;

  return (
    <aside
      style={{ width }}
      className="relative shrink-0 flex flex-col border-l border-tcip-border bg-tcip-panel"
      aria-label="Agent terminal"
    >
      {/* Drag the left edge to resize; width persists. */}
      <div
        onMouseDown={startResize}
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize agent terminal"
        title="Drag to resize"
        className="absolute left-0 top-0 bottom-0 -ml-1 w-1.5 z-10 cursor-col-resize hover:bg-tcip-accent/40"
      />
      <div className="h-9 shrink-0 flex items-center justify-between px-3 border-b border-tcip-border bg-tcip-panel">
        <div className="flex items-center gap-2">
          <span className="tcip-eyebrow">TCIP Agent</span>
          {status?.available && (
            <span
              className={`inline-block h-1.5 w-1.5 rounded-full ${
                conn === "open" ? "bg-tcip-accent" : "bg-tcip-warn animate-pulse"
              }`}
              title={
                conn === "open"
                  ? "Agent connected"
                  : conn === "reconnecting"
                    ? "Reconnecting…"
                    : "Connecting…"
              }
              aria-label={conn === "open" ? "Agent connected" : "Agent connecting"}
            />
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={restart}
            title="Restart the agent (ends its current conversation)"
            aria-label="Restart the agent"
            className="grid h-6 w-6 place-items-center rounded text-tcip-muted transition-colors hover:bg-tcip-hover hover:text-tcip-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-tcip-accent/70"
          >
            <svg viewBox="0 0 16 16" width="14" height="14" fill="none" aria-hidden="true">
              <path
                d="M12.8 8a4.8 4.8 0 1 1-1.4-3.4"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
              />
              <path
                d="M12.8 2.6v3h-3"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
          <button
            onClick={() => setOpen(false)}
            aria-label="Minimize agent terminal"
            title="Minimize (the agent keeps running); reopen from the TCIP Agent button"
            className="grid h-6 w-6 place-items-center rounded text-tcip-muted transition-colors hover:bg-tcip-hover hover:text-tcip-fg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-tcip-accent/70"
          >
            <svg viewBox="0 0 16 16" width="14" height="14" fill="none" aria-hidden="true">
              <path d="M4 11h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
        </div>
      </div>

      {status && !status.available ? (
        <div className="p-4 flex flex-col gap-3">
          <p className="text-[12px] text-tcip-muted">
            {status.reason ??
              "Claude Code is not available. Install the claude CLI and sign in to enable the agent terminal."}
          </p>
          <button className="tcip-btn self-start" onClick={() => setStatus(null)}>
            Retry
          </button>
        </div>
      ) : (
        <>
          {showStarterHint && (
            <div
              data-testid="terminal-starter-hint"
              className="shrink-0 px-3 py-2 border-b border-tcip-border bg-tcip-panel text-[11px] text-tcip-muted"
            >
              Tell the agent what you&rsquo;re working on, for example: &ldquo;I have photos of my{" "}
              <span className="italic">[crop]</span> and want to measure{" "}
              <span className="italic">[trait]</span>.&rdquo;
            </div>
          )}
          <div className="flex-1 min-h-0 relative">
            <div ref={hostRef} data-testid="terminal-host" className="absolute inset-0 p-2" />
            {error && (
              <div className="absolute bottom-2 left-2 right-2 text-[11px] text-tcip-fp bg-tcip-panel/90 border border-tcip-border rounded p-2">
                {error}
              </div>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
