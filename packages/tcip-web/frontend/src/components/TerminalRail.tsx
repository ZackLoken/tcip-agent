/**
 * The agent rail: a real agent harness from the backend's provider table, embedded. An
 * xterm.js terminal attached over WebSocket to a server-side PTY running the harness in the
 * workspace, so the breeder talks to the agent the platform is designed around with no
 * translation layer. The terminal is recolored to the field-station palette via the ANSI
 * theme; the harness's own layout is untouched. Staged requests are submitted to the session for
 * its agent, and choosing another harness while an agent runs waits for that agent's hand-off
 * before the chosen one starts.
 */

import { FitAddon } from "@xterm/addon-fit";
import { Unicode11Addon } from "@xterm/addon-unicode11";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { Terminal } from "@xterm/xterm";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import "@xterm/xterm/css/xterm.css";

import { committedOf, StructuredRefusalError } from "@/api/http";
import { terminalApi, terminalWsUrl } from "@/api/terminal";
import {
  TERMINAL_ENDED_CLOSE_CODE,
  type ProviderStatus,
  type TerminalEnd,
  type TerminalFrame,
} from "@/api/types.generated";
import { createReconnectingSocket, type ReconnectingSocket } from "@/lib/reconnectingSocket";
import { useStore } from "@/store";
import type { TerminalPhase } from "@/store/slices/terminalOpen";

const MIN_WIDTH = 320;
const DEFAULT_WIDTH = 480;
const WIDTH_KEY = "tcip.terminal_width";

/** The provider table as the backend lists it, or the reason it could not be asked. */
type ProviderTable = { rows: ProviderStatus[]; reason: null } | { rows: null; reason: string };

/** The row the rail launches: the remembered one if the table lists it and it can launch,
 * else the first row that can launch, else the first row. */
function chooseRow(rows: ProviderStatus[], remembered: string | null): ProviderStatus | null {
  const launchable = rows.filter((row) => row.unavailable_reason === null);
  return launchable.find((row) => row.id === remembered) ?? launchable[0] ?? rows[0] ?? null;
}

/** What the harness picker says it does in each phase of the rail's agent. */
const PICKER_TITLE: Record<TerminalPhase, string> = {
  live: "The agent harness this terminal runs; choosing another asks the agent for its hand-off, then starts the chosen one",
  "handing-off":
    "The agent is writing its hand-off; the chosen harness starts once it is recorded, or at once on End",
  ending: "The agent is ending; the harness can be chosen once it has ended",
  ended: "The agent harness Start launches",
};

/** Whether the rail's agent is running in `phase`: live, or handing off before a switch. */
function running(phase: TerminalPhase): boolean {
  return phase === "live" || phase === "handing-off";
}

function clampWidth(px: number): number {
  const max = Math.max(MIN_WIDTH, Math.round(window.innerWidth * 0.7));
  return Math.max(MIN_WIDTH, Math.min(Math.round(px), max));
}

function persistWidth(px: number) {
  try {
    localStorage.setItem(WIDTH_KEY, String(px));
  } catch {
    /* width just won't persist */
  }
}

/**
 * The width in CSS pixels of one terminal column, measured off the screen xterm has painted
 * into `host` at `cols` columns; null while nothing is painted, when no step is known.
 */
function columnWidth(host: HTMLElement | null, cols: number): number | null {
  const screen = host?.querySelector(".xterm-screen");
  if (!screen || cols <= 0) return null;
  const px = screen.getBoundingClientRect().width / cols;
  return px > 0 ? px : null;
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
  // The provider table and the row the breeder chose from it; the chosen row's own reason says
  // why it cannot launch, and a backend that could not be asked leaves no table and its reason.
  const [table, setTable] = useState<ProviderTable | null>(null);
  const selected = useStore((s) => s.terminalProvider);
  const row = table?.rows ? chooseRow(table.rows, selected) : null;
  const reason = table === null ? null : (table.reason ?? row?.unavailable_reason ?? null);
  const available = table !== null && reason === null && row !== null;
  const [error, setError] = useState<string | null>(null);
  // Live PTY link state, surfaced as a header dot (green = attached, amber = (re)connecting).
  const [conn, setConn] = useState<"connecting" | "open" | "reconnecting">("connecting");
  // Callbacks read the phase through useStore.getState(), so it is fresh outside a render.
  const phase = useStore((s) => s.terminalPhase);
  const setPhase = useStore((s) => s.setTerminalPhase);
  const [confirmingExit, setConfirmingExit] = useState(false);
  const hostRef = useRef<HTMLDivElement | null>(null);
  const sessionRef = useRef<string | null>(null);
  const socketRef = useRef<ReconnectingSocket | null>(null);
  const termRef = useRef<Terminal | null>(null);
  // The submit loop under way, which an End lets settle before it asks the backend.
  const submissionRef = useRef<Promise<void> | null>(null);
  const pendingMessages = useStore((s) => s.pendingTerminalMessages);

  // Submit staged requests, oldest first, to the live session, each leaving the queue once the
  // backend took it; the backend delivers them once the agent can take input.
  const submitStaged = useCallback(() => {
    const id = sessionRef.current;
    if (!id || submissionRef.current) return;
    submissionRef.current = (async () => {
      try {
        for (;;) {
          const [next] = useStore.getState().pendingTerminalMessages;
          const { terminalPhase } = useStore.getState();
          if (next === undefined || sessionRef.current !== id || terminalPhase !== "live") break;
          await terminalApi.submit(id, { text: next });
          useStore.getState().dropDeliveredTerminalMessages(1);
        }
      } catch (e) {
        setError(String(e));
      }
    })().finally(() => {
      submissionRef.current = null;
    });
  }, []);

  /** Stop the socket and forget the session id, so the socket's next start launches an agent. */
  const detach = useCallback(() => {
    socketRef.current?.stop();
    sessionRef.current = null;
  }, []);

  /** Show that the agent ended: detached, and only Start launches another. */
  const showEnded = useCallback(() => {
    detach();
    setPhase("ended");
  }, [detach, setPhase]);

  /** Run again in `phase`: the socket runs on the session it holds, or through its own create
   *  when none, and the staged requests are submitted to that session. */
  const resume = useCallback(
    (phase: TerminalPhase) => {
      setPhase(phase);
      socketRef.current?.start();
      submitStaged();
    },
    [setPhase, submitStaged],
  );

  /** Launch the chosen row through the socket's own create, the ended agent's screen cleared
   *  while the launch is made. */
  const start = useCallback(() => {
    termRef.current?.reset();
    resume("live");
  }, [resume]);

  /** Start the row a switch chose, the agent it switched from having ended. */
  const switchToChosen = useCallback(() => {
    detach();
    start();
  }, [detach, start]);

  const [width, setWidth] = useState<number>(() => {
    try {
      const v = parseInt(localStorage.getItem(WIDTH_KEY) ?? "", 10);
      return Number.isFinite(v) ? clampWidth(v) : DEFAULT_WIDTH;
    } catch {
      return DEFAULT_WIDTH;
    }
  });

  const [dragging, setDragging] = useState(false);

  // A drag's listeners are installed again in the commit of each width it sets, so its release
  // persists the width on screen.
  useLayoutEffect(() => {
    if (!dragging) return;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    // Rail is docked right, so its width grows as the pointer moves left.
    const onMove = (ev: MouseEvent) => setWidth(clampWidth(window.innerWidth - ev.clientX));
    const onUp = () => {
      setDragging(false);
      persistWidth(width);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [dragging, width]);

  function startResize(e: React.MouseEvent) {
    e.preventDefault();
    setDragging(true);
  }

  /** Arrow keys on the handle resize by one terminal column; left widens, since the rail is docked right. */
  function resizeByKey(e: React.KeyboardEvent) {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const step = columnWidth(hostRef.current, termRef.current?.cols ?? 0);
    if (step === null) return;
    e.preventDefault();
    const next = clampWidth(width + (e.key === "ArrowLeft" ? step : -step));
    setWidth(next);
    persistWidth(next);
  }

  useEffect(() => {
    if (!open || table !== null) return;
    terminalApi
      .status()
      .then(({ providers }) => setTable({ rows: providers, reason: null }))
      .catch(() =>
        // Backend unreachable is not a missing harness; say so, and keep Retry viable.
        setTable({
          rows: null,
          reason: "Couldn't reach the TCIP backend. Is it running? Retry once it's up.",
        }),
      );
  }, [open, table]);

  // Terminal lifecycle: build xterm, attach the PTY WebSocket while the agent runs, wire input
  // and resize. Closing the rail drops the socket; the session survives and reopening replays it.
  useEffect(() => {
    if (!open || !available || !table.rows || !hostRef.current) return;
    const rows = table.rows;
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

    // Copy-on-select; Ctrl/Cmd+Shift+C and Ctrl+Insert copy; Ctrl+C copies a selection, else
    // SIGINT; a right click copies a selection, else pastes unless the program took the mouse.
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
      } else if (term.modes.mouseTrackingMode === "none") {
        pasteClipboard();
      }
    };
    // The platform's text-paste modifier; Ctrl+V on a Mac stays the program's own key.
    const pasteModifier = /^mac/i.test(navigator.platform) ? "metaKey" : "ctrlKey";
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
      // xterm sends Ctrl+V as ^V and cancels it; false leaves a paste key to the browser's paste
      // event, which xterm's own listener pastes.
      const pasteV =
        (e[pasteModifier] || (e.ctrlKey && e.shiftKey)) && (e.key === "V" || e.key === "v");
      return !(pasteV || (e.shiftKey && e.key === "Insert"));
    });

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
              // The choice as it stands at the create, read like phase in callbacks.
              provider: chooseRow(rows, useStore.getState().terminalProvider)!.id,
              rows: term.rows,
              cols: term.cols,
              user: useStore.getState().user,
            });
            if (closedByClient) throw new Error("terminal rail unmounted mid-connect");
            sessionRef.current = created.session_id;
            submitStaged();
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
      // The ended code closes a session whose agent ended, starting a switch's chosen row; any
      // other close retries.
      onClose: (ev) => {
        if (ev.code !== TERMINAL_ENDED_CLOSE_CODE) setConn("reconnecting");
        else if (useStore.getState().terminalPhase === "handing-off") switchToChosen();
        else showEnded();
      },
    });

    const send = (payload: TerminalFrame) => socket.send(JSON.stringify(payload));
    socketRef.current = socket;
    if (running(useStore.getState().terminalPhase)) socket.start();

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
      socket.stop();
      socketRef.current = null;
      observer.disconnect();
      host.removeEventListener("contextmenu", onContextMenu);
      window.removeEventListener("mouseup", onWindowMouseUp);
      dataSub.dispose();
      resizeSub.dispose();
      term.dispose();
      termRef.current = null;
    };
  }, [open, available, table, submitStaged, showEnded, switchToChosen]);

  useEffect(() => {
    if (pendingMessages.length) submitStaged();
  }, [pendingMessages, submitStaged]);

  /** End the running session's agent by the person at once: the socket stops and the submit loop
   *  settles before the end is asked; once it answers, the requests the agent was never given are
   *  staged ahead of the others and the rail shows that it ended, or, during a switch, starts the
   *  chosen row. The socket reattaches to the session in the phase it had when the backend did
   *  not end it. */
  async function endAgent() {
    const id = sessionRef.current;
    const was = useStore.getState().terminalPhase;
    if (!id || !running(was)) return;
    setPhase("ending");
    socketRef.current?.stop();
    await submissionRef.current;
    let answer: TerminalEnd | null = { undelivered: [] };
    try {
      answer = await terminalApi.end(id, { user: useStore.getState().user });
    } catch (e) {
      // A 404 says the backend no longer holds the session: it already ended.
      if (!(e instanceof StructuredRefusalError && e.status === 404)) {
        setError(String(e));
        answer = committedOf<TerminalEnd>(e);
      }
    }
    if (answer === null) {
      resume(was);
      return;
    }
    useStore.getState().restageTerminalMessages(answer.undelivered);
    if (was === "handing-off") switchToChosen();
    else showEnded();
  }

  /** Remember the chosen row at once when the rail holds no agent; while one is live, also ask
   *  it for its hand-off, the session's end then starting the chosen row. A session the backend
   *  ended at once starts it now; a refused hand-off chooses the row it ran again. */
  async function choose(provider: string) {
    const id = sessionRef.current;
    const ran = row?.id;
    const { setTerminalProvider, terminalPhase, user } = useStore.getState();
    if (!id) setTerminalProvider(provider);
    if (!id || !ran || terminalPhase !== "live") return;
    setTerminalProvider(provider);
    setPhase("handing-off");
    let ended: TerminalEnd | null;
    try {
      ended = await terminalApi.handOff(id, { user });
    } catch (e) {
      setError(String(e));
      setTerminalProvider(ran);
      setPhase("live");
      return;
    }
    if (ended === null) return;
    useStore.getState().restageTerminalMessages(ended.undelivered);
    if (useStore.getState().terminalPhase === "handing-off") switchToChosen();
  }

  if (!open) return null;

  return (
    <aside
      style={{ width }}
      className="relative shrink-0 flex flex-col border-l border-tcip-border bg-tcip-panel"
      aria-label="Agent terminal"
    >
      {/* Drag the left edge or press the arrow keys on it to resize; width persists. */}
      <div
        onMouseDown={startResize}
        role="presentation"
        className="absolute left-0 top-0 bottom-0 -ml-1 w-1.5 z-10 cursor-col-resize hover:bg-tcip-accent/40"
      >
        {/* A focusable separator is ARIA's window splitter; jsx-a11y reads the role as static. */}
        {/* eslint-disable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */}
        <div
          role="separator"
          tabIndex={0}
          aria-orientation="vertical"
          aria-label="Resize agent terminal"
          aria-valuenow={width}
          aria-valuemin={MIN_WIDTH}
          title="Drag, or press the arrow keys, to resize by one terminal column"
          onKeyDown={resizeByKey}
          className="h-full w-full focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-tcip-accent/70"
        />
        {/* eslint-enable jsx-a11y/no-noninteractive-element-interactions, jsx-a11y/no-noninteractive-tabindex */}
      </div>
      <div className="h-9 shrink-0 flex items-center justify-between px-3 border-b border-tcip-border bg-tcip-panel">
        <div className="flex items-center gap-2">
          <span className="text-[12px] font-medium text-tcip-muted">TCIP Agent</span>
          {table?.rows && (
            <select
              aria-label="Agent harness"
              title={PICKER_TITLE[phase]}
              disabled={phase === "ending" || phase === "handing-off"}
              value={row?.id ?? ""}
              onChange={(e) => void choose(e.target.value)}
              className="h-6 max-w-[11rem] rounded border border-tcip-border bg-tcip-panel px-1 text-[13px] text-tcip-fg focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-tcip-accent/70 disabled:opacity-40"
            >
              {table.rows.map((r) => (
                <option
                  key={r.id}
                  value={r.id}
                  disabled={r.unavailable_reason !== null}
                  title={r.unavailable_reason ?? undefined}
                >
                  {r.name}
                </option>
              ))}
            </select>
          )}
          {available && running(phase) && (
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
            onClick={() => setOpen(false)}
            aria-label="Minimize agent terminal"
            title="Minimize (the agent keeps running); reopen from the TCIP Agent button"
            className="grid h-6 w-6 place-items-center rounded text-tcip-muted transition-colors hover:bg-tcip-hover hover:text-tcip-fg focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-tcip-accent/70"
          >
            <svg viewBox="0 0 16 16" width="14" height="14" fill="none" aria-hidden="true">
              <path d="M4 11h8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </button>
          {confirmingExit ? (
            <>
              <span className="text-[12px] text-tcip-muted">End the agent?</span>
              <button
                onClick={() => {
                  setConfirmingExit(false);
                  void endAgent();
                }}
                className="tcip-btn-danger h-6 px-2 text-[12px]"
              >
                End
              </button>
              <button
                onClick={() => setConfirmingExit(false)}
                className="tcip-btn h-6 px-2 text-[12px]"
              >
                Cancel
              </button>
            </>
          ) : (
            running(phase) && (
              <button
                onClick={() => setConfirmingExit(true)}
                aria-label="End the agent"
                title={
                  phase === "handing-off"
                    ? "End the agent now, without waiting for its hand-off; the chosen harness then starts"
                    : "End the agent and its conversation; Start then launches a fresh one"
                }
                className="grid h-6 w-6 place-items-center rounded text-tcip-muted transition-colors hover:bg-tcip-hover hover:text-tcip-fg focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-tcip-accent/70"
              >
                <svg viewBox="0 0 16 16" width="14" height="14" fill="none" aria-hidden="true">
                  <path
                    d="M4.5 4.5l7 7M11.5 4.5l-7 7"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                  />
                </svg>
              </button>
            )
          )}
        </div>
      </div>

      {table !== null && reason !== null ? (
        <div className="p-4 flex flex-col gap-3">
          <p className="text-[13px] text-tcip-muted">{reason}</p>
          <button className="tcip-btn self-start" onClick={() => setTable(null)}>
            Retry
          </button>
        </div>
      ) : (
        <>
          <div className="flex-1 min-h-0 relative">
            <div ref={hostRef} data-testid="terminal-host" className="absolute inset-0 p-2" />
            {phase === "ended" && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-tcip-panel/90 p-4">
                <p className="text-[13px] text-tcip-muted">
                  The agent ended. Start launches an agent on the chosen harness.
                </p>
                <button className="tcip-btn-primary" onClick={start}>
                  Start
                </button>
              </div>
            )}
            {error && (
              <div className="absolute bottom-2 left-2 right-2 text-[13px] tcip-problem bg-tcip-panel/90 border border-tcip-border rounded p-2">
                {error}
              </div>
            )}
          </div>
        </>
      )}
    </aside>
  );
}
