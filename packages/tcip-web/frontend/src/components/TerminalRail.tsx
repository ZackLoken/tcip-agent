/**
 * The agent rail: a real agent harness from the backend's provider table, embedded. An
 * xterm.js terminal attached over WebSocket to a server-side PTY running the harness in the
 * repo root, so the breeder talks to the agent the platform is designed around with no
 * translation layer. The terminal is recolored to the field-station palette via the ANSI
 * theme; the harness's own layout is untouched. Each launch's session-start ritual is shown
 * above the terminal, and staged requests are submitted to the session for its agent.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { Unicode11Addon } from "@xterm/addon-unicode11";
import { WebLinksAddon } from "@xterm/addon-web-links";
import "@xterm/xterm/css/xterm.css";

import { terminalApi, terminalWsUrl } from "@/api/terminal";
import type { TerminalInputFrame, TerminalResizeFrame } from "@/api/types.generated";
import { createReconnectingSocket } from "@/lib/reconnectingSocket";
import { useStore } from "@/store";

type TerminalSendFrame = TerminalInputFrame | TerminalResizeFrame;

const MIN_WIDTH = 320;
const DEFAULT_WIDTH = 480;
const WIDTH_KEY = "tcip.terminal_width";

function clampWidth(px: number): number {
  const max = Math.max(MIN_WIDTH, Math.round(window.innerWidth * 0.7));
  return Math.max(MIN_WIDTH, Math.min(Math.round(px), max));
}

/**
 * Field-station terminal theme. A terminal UI draws with the 16 ANSI slots, so mapping
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
  // The provider row the rail launches (the table's first row) and why it cannot launch; a
  // backend that could not be asked leaves no row and its own reason.
  const [status, setStatus] = useState<{ provider?: string; reason: string | null } | null>(null);
  const available = status?.reason === null;
  const [error, setError] = useState<string | null>(null);
  // Live PTY link state, surfaced as a header dot (green = attached, amber = (re)connecting).
  const [conn, setConn] = useState<"connecting" | "open" | "reconnecting">("connecting");
  const hostRef = useRef<HTMLDivElement | null>(null);
  const sessionRef = useRef<string | null>(null);
  const termRef = useRef<Terminal | null>(null);
  // True while a restart is asked for, and while staged requests are being submitted.
  const restartingRef = useRef(false);
  const submittingRef = useRef(false);
  const pendingMessages = useStore((s) => s.pendingTerminalMessages);

  // Submit staged requests, oldest first, to the session's current launch, each leaving the
  // queue once the backend took it; the backend delivers them after that launch's ritual.
  const submitStaged = useCallback(async () => {
    const id = sessionRef.current;
    if (!id || submittingRef.current) return;
    submittingRef.current = true;
    try {
      for (;;) {
        const [next] = useStore.getState().pendingTerminalMessages;
        if (next === undefined || sessionRef.current !== id || restartingRef.current) break;
        await terminalApi.submit(id, { text: next });
        useStore.getState().dropDeliveredTerminalMessages(1);
      }
    } catch (e) {
      setError(String(e));
    } finally {
      submittingRef.current = false;
    }
  }, []);

  // The launch's session-start ritual, shown until the breeder types.
  const [ritual, setRitual] = useState<string | null>(null);

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
      .then(({ providers: [row] }) =>
        setStatus({ provider: row.id, reason: row.unavailable_reason }),
      )
      .catch(() =>
        // Backend unreachable is not a missing harness; say so, and keep Retry viable.
        setStatus({
          reason: "Couldn't reach the TCIP backend. Is it running? Retry once it's up.",
        }),
      );
  }, [open, status]);

  // Terminal lifecycle: build xterm, attach the PTY WebSocket, wire input/resize. Closing the
  // rail drops the socket; the server session stays alive and reopening replays the scrollback.
  useEffect(() => {
    const provider = status?.provider;
    if (!open || status?.reason !== null || !provider || !hostRef.current) return;
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

    // Paste: Ctrl/Cmd+V fires a native paste event on xterm's hidden textarea, but the browser
    // does not always route it there (focus, or the app swallowing the key). Intercept in the
    // capture phase and hand the text to term.paste(), which wraps it in bracketed-paste mode,
    // so a multi-line paste reaches the agent as one paste, not a line-per-Enter burst.
    const onPaste = (e: ClipboardEvent) => {
      const text = e.clipboardData?.getData("text");
      if (text) {
        term.paste(text);
        e.preventDefault();
        e.stopPropagation();
      }
    };

    // Copy-on-select; right-click copies a selection or else pastes; Ctrl/Cmd+Shift+C and
    // Ctrl+Insert copy, Ctrl+Shift+V and Shift+Insert paste; Ctrl+C copies a selection, else SIGINT.
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

    host.addEventListener("paste", onPaste, true);
    host.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("mouseup", onWindowMouseUp);

    // The URL provider re-checks closedByClient after the await, so a torn-down effect
    // (rail toggled, StrictMode remount) cannot attach a fresh socket to a disposed terminal.
    let closedByClient = false;
    const socket = createReconnectingSocket({
      url: async () => {
        try {
          if (!sessionRef.current) {
            const created = await terminalApi.createSession({
              provider,
              rows: term.rows,
              cols: term.cols,
            });
            if (closedByClient) throw new Error("terminal rail unmounted mid-connect");
            sessionRef.current = created.session_id;
            setRitual(created.ritual);
            void submitStaged();
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
    socket.start();

    const dataSub = term.onData((data) => {
      setRitual(null);
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
      host.removeEventListener("contextmenu", onContextMenu);
      window.removeEventListener("mouseup", onWindowMouseUp);
      dataSub.dispose();
      resizeSub.dispose();
      term.dispose();
      termRef.current = null;
    };
  }, [open, status, submitStaged]);

  useEffect(() => {
    if (pendingMessages.length) void submitStaged();
  }, [pendingMessages, submitStaged]);

  async function restart() {
    const id = sessionRef.current;
    const term = termRef.current;
    if (!id || !term || !status?.provider) return;
    restartingRef.current = true;
    term.reset(); // the replacement process paints a fresh screen
    try {
      const restarted = await terminalApi.restart(id, {
        provider: status.provider,
        rows: term.rows,
        cols: term.cols,
      });
      setRitual(restarted.ritual);
    } catch (e) {
      setError(String(e));
    } finally {
      restartingRef.current = false;
      void submitStaged();
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
          {available && (
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

      {status && status.reason !== null ? (
        <div className="p-4 flex flex-col gap-3">
          <p className="text-[12px] text-tcip-muted">{status.reason}</p>
          <button className="tcip-btn self-start" onClick={() => setStatus(null)}>
            Retry
          </button>
        </div>
      ) : (
        <>
          {ritual && (
            <div
              data-testid="terminal-ritual"
              className="shrink-0 px-3 py-2 border-b border-tcip-border bg-tcip-panel text-[11px] text-tcip-muted"
            >
              {ritual}
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
