/**
 * The agent rail — the real Claude Code CLI, embedded. An xterm.js terminal attached
 * over WebSocket to a server-side PTY running `claude` in the repo root, so the breeder
 * talks to the same agent (CLAUDE.md, skills, MCP tools, permission prompts) the
 * platform is designed around — full fidelity, no translation layer. The terminal is
 * recolored to the field-station palette via the ANSI theme (Claude Code paints its TUI
 * with ANSI colors, so it renders in our palette natively); its internal layout is
 * untouched — that's the exact-experience point.
 */

import { useEffect, useRef, useState } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { Unicode11Addon } from "@xterm/addon-unicode11";
import { WebLinksAddon } from "@xterm/addon-web-links";
import "@xterm/xterm/css/xterm.css";

import { terminalApi, terminalWsUrl } from "@/api/terminal";
import { useStore } from "@/store";

const MAX_BACKOFF_MS = 15_000;
const MIN_WIDTH = 320;
const DEFAULT_WIDTH = 480;
const WIDTH_KEY = "tcip.terminal_width";

function clampWidth(px: number): number {
  const max = Math.max(MIN_WIDTH, Math.round(window.innerWidth * 0.7));
  return Math.max(MIN_WIDTH, Math.min(Math.round(px), max));
}

/**
 * Field-station terminal theme. Claude Code draws with the 16 ANSI slots, so mapping
 * them to the app palette re-skins the real TUI: green → canopy, yellow → late-summer
 * gold, red → FP red, dim gray → sage. Background stays tcip-bg so the rail sits flush
 * with the app chrome; the cursor is persimmon — the same accent the SeasonRail ends on.
 */
const FIELD_STATION_THEME = {
  background: "#1E1E1E",
  foreground: "#E7E5DC",
  cursor: "#E6976B",
  cursorAccent: "#1E1E1E",
  selectionBackground: "#50775455",
  black: "#33352C",
  red: "#EF5350",
  green: "#7FA96A",
  yellow: "#C9A24B",
  blue: "#7E9CB9",
  magenta: "#B48EAD",
  cyan: "#7FB0A9",
  white: "#E7E5DC",
  brightBlack: "#8C9082",
  brightRed: "#F28B82",
  brightGreen: "#9FC48A",
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
  const hostRef = useRef<HTMLDivElement | null>(null);
  const sessionRef = useRef<string | null>(null);
  const termRef = useRef<Terminal | null>(null);

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
        // Backend unreachable is NOT "claude missing" — say so, and keep Retry viable.
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

    // Paste: Ctrl/Cmd+V fires a native paste event on xterm's hidden textarea, but the
    // browser doesn't always route it there (focus, or the app swallowing the key), so
    // Ctrl+V "did nothing". Intercept in the capture phase and hand the text to
    // term.paste(), which wraps it in bracketed-paste mode — so a multi-line paste
    // reaches Claude Code as ONE paste, not a line-per-Enter burst.
    const onPaste = (e: ClipboardEvent) => {
      const text = e.clipboardData?.getData("text");
      if (text) {
        term.paste(text);
        e.preventDefault();
        e.stopPropagation();
      }
    };
    const focusTerm = () => term.focus();
    host.addEventListener("paste", onPaste, true);
    host.addEventListener("mousedown", focusTerm);

    let ws: WebSocket | null = null;
    let closedByClient = false;
    let backoff = 500;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const send = (payload: unknown) => {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
    };

    const connect = async () => {
      if (closedByClient) return;
      try {
        if (!sessionRef.current) {
          const { session_id } = await terminalApi.createSession(term.rows, term.cols);
          sessionRef.current = session_id;
        }
      } catch (e) {
        setError(String(e));
        return;
      }
      // Re-check after the await: the effect may have cleaned up mid-request (rail
      // toggled, StrictMode remount) — a socket created now would attach to a
      // disposed terminal and leak.
      if (closedByClient) return;
      const socket = new WebSocket(terminalWsUrl(sessionRef.current));
      ws = socket;
      let opened = false;
      socket.onopen = () => {
        opened = true;
        backoff = 500;
        setError(null);
        term.reset(); // the replay repaints the screen from scratch
        send({ type: "resize", rows: term.rows, cols: term.cols });
      };
      socket.onmessage = (ev) => {
        if (typeof ev.data === "string") term.write(ev.data);
      };
      socket.onclose = (ev) => {
        if (closedByClient) return;
        // A close before open (or 1008 "unknown session") means the backend no longer
        // knows this session — e.g. it restarted. Retrying the dead id forever is the
        // silent-death failure mode; drop it so the next attempt re-creates a session.
        if (!opened || ev.code === 1008) {
          sessionRef.current = null;
        }
        const delay = backoff;
        backoff = Math.min(backoff * 2, MAX_BACKOFF_MS);
        reconnectTimer = setTimeout(() => void connect(), delay);
      };
    };
    void connect();

    const dataSub = term.onData((data) => send({ type: "input", data }));
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
      if (reconnectTimer !== null) clearTimeout(reconnectTimer);
      observer.disconnect();
      host.removeEventListener("paste", onPaste, true);
      host.removeEventListener("mousedown", focusTerm);
      dataSub.dispose();
      resizeSub.dispose();
      ws?.close();
      term.dispose();
      termRef.current = null;
    };
  }, [open, status]);

  async function restart() {
    const id = sessionRef.current;
    const term = termRef.current;
    if (!id) return;
    try {
      await terminalApi.restart(id, term?.rows ?? 30, term?.cols ?? 100);
      term?.reset();
    } catch (e) {
      setError(String(e));
    }
  }

  if (!open) return null;

  return (
    <aside
      style={{ width }}
      className="relative shrink-0 flex flex-col border-l border-tcip-border bg-tcip-bg"
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
        <span className="tcip-eyebrow">Agent · Claude Code</span>
        <div className="flex items-center gap-1">
          <button
            onClick={restart}
            title="Restart the agent (ends its current conversation)"
            className="text-[11px] text-tcip-muted hover:text-tcip-fg px-1.5 h-6 rounded hover:bg-tcip-hover transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-tcip-accent/70"
          >
            Restart
          </button>
          <button
            onClick={() => setOpen(false)}
            aria-label="Minimize agent terminal"
            title="Minimize — the agent keeps running; reopen from the ✦ Agent button"
            className="text-tcip-muted hover:text-tcip-fg text-[15px] leading-none px-1.5 h-6 rounded hover:bg-tcip-hover transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-tcip-accent/70"
          >
            –
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
        <div className="flex-1 min-h-0 relative">
          <div ref={hostRef} data-testid="terminal-host" className="absolute inset-0 p-2" />
          {error && (
            <div className="absolute bottom-2 left-2 right-2 text-[11px] text-tcip-fp bg-tcip-panel/90 border border-tcip-border rounded p-2">
              {error}
            </div>
          )}
        </div>
      )}
    </aside>
  );
}
