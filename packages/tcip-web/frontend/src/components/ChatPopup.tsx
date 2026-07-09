/**
 * In-app agent chat. A floating panel that talks to the same Claude agent the project is
 * built around (via the tcip-web sidecar), so the human can ask questions without leaving
 * the current tab. When no agent backend is configured, the panel says so plainly instead
 * of breaking. See docs/chat-popup-design.md.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { chatApi, ChatSocket, type ChatEvent } from "@/api/chat";
import { useStore } from "@/store";

type Item =
  | { kind: "user"; text: string }
  | { kind: "assistant"; text: string }
  | { kind: "tool"; name: string; summary?: string; ok?: boolean }
  | { kind: "permission"; requestId: string; tool: string; detail: string };

function toItems(events: ChatEvent[]): Item[] {
  const items: Item[] = [];
  for (const ev of events) {
    switch (ev.type) {
      case "user_message":
        items.push({ kind: "user", text: String(ev.text ?? "") });
        break;
      case "assistant_delta": {
        const last = items[items.length - 1];
        if (last && last.kind === "assistant") last.text += String(ev.text ?? "");
        else items.push({ kind: "assistant", text: String(ev.text ?? "") });
        break;
      }
      case "tool_use":
        items.push({
          kind: "tool",
          name: String(ev.name ?? "tool"),
          summary: ev.input_summary ? String(ev.input_summary) : undefined,
        });
        break;
      case "tool_result": {
        // Attach the result to the most recent matching tool chip.
        const chip = [...items].reverse().find((i) => i.kind === "tool" && i.ok === undefined) as
          (Item & { kind: "tool" }) | undefined;
        if (chip) chip.ok = Boolean(ev.ok);
        break;
      }
      case "permission_request":
        items.push({
          kind: "permission",
          requestId: String(ev.request_id ?? ""),
          tool: String(ev.tool ?? "tool"),
          detail: String(ev.detail ?? ""),
        });
        break;
      default:
        break; // session_state / turn_done drive status, not the transcript body
    }
  }
  return items;
}

function isBusy(events: ChatEvent[]): boolean {
  let busy = false;
  for (const ev of events) {
    if (ev.type === "session_state") busy = ev.state === "running";
    else if (ev.type === "turn_done") busy = false;
  }
  return busy;
}

export function ChatPopup() {
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<{ available: boolean; reason?: string } | null>(null);
  const [events, setEvents] = useState<ChatEvent[]>([]);
  const [input, setInput] = useState("");
  const [includeContext, setIncludeContext] = useState(true);
  const [answered, setAnswered] = useState<Set<string>>(new Set());
  const [unread, setUnread] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sessionIdRef = useRef<string | null>(null);
  const socketRef = useRef<ChatSocket | null>(null);
  const openRef = useRef(open);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const activeTab = useStore((s) => s.gui.active_tab);
  const datasetDate = useStore((s) => s.gui.dataset.date);

  openRef.current = open;
  const busy = useMemo(() => isBusy(events), [events]);
  const items = useMemo(() => toItems(events), [events]);

  // Establish the session + socket the first time the panel opens (kept alive after,
  // so background output can raise the unread dot).
  useEffect(() => {
    if (!open) return;
    setUnread(false);
    if (status === null) {
      chatApi
        .status()
        .then(setStatus)
        .catch(() => setStatus({ available: false }));
    }
  }, [open, status]);

  useEffect(() => {
    if (!open || !status?.available || sessionIdRef.current) return;
    let cancelled = false;
    chatApi
      .createSession()
      .then(({ session_id }) => {
        if (cancelled) return;
        sessionIdRef.current = session_id;
        const socket = new ChatSocket(
          session_id,
          (ev) => {
            setEvents((prev) => [...prev, ev]);
            if (!openRef.current) setUnread(true);
          },
          () => setEvents([]), // reset on (re)connect; the backend replays the transcript
        );
        socket.connect();
        socketRef.current = socket;
      })
      .catch((e) => setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [open, status]);

  useEffect(() => () => socketRef.current?.close(), []);

  // Keep the newest message in view.
  useEffect(() => {
    if (open && scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [events, open]);

  async function send() {
    const text = input.trim();
    const id = sessionIdRef.current;
    if (!text || !id || busy) return;
    setInput("");
    setError(null);
    try {
      await chatApi.send(id, text, includeContext);
    } catch (e) {
      setError(String(e));
    }
  }

  async function stop() {
    const id = sessionIdRef.current;
    if (id) await chatApi.interrupt(id).catch(() => {});
  }

  async function decide(requestId: string, decision: "allow" | "deny") {
    const id = sessionIdRef.current;
    if (!id) return;
    setAnswered((prev) => new Set(prev).add(requestId));
    try {
      const res = await chatApi.permission(id, requestId, decision);
      if (res.status !== "ok") {
        // The agent process was gone — the decision never reached it. Don't pretend it did.
        setError("The agent isn't running anymore — that decision wasn't delivered.");
        setAnswered((prev) => {
          const next = new Set(prev);
          next.delete(requestId);
          return next;
        });
      }
    } catch {
      setError("Couldn't send that decision to the agent.");
    }
  }

  return (
    <div className="fixed bottom-9 right-3 z-40 flex flex-col items-end gap-2">
      {open && (
        <div className="w-[380px] max-h-[70vh] flex flex-col rounded-lg border border-tcip-border bg-tcip-panel shadow-lg">
          <div className="flex items-center justify-between px-3 h-9 border-b border-tcip-border shrink-0">
            <span className="text-[12px] font-medium text-tcip-fg">Agent chat</span>
            <button
              onClick={() => setOpen(false)}
              className="text-tcip-muted hover:text-tcip-fg text-[12px] px-1"
              aria-label="Close agent chat"
            >
              ✕
            </button>
          </div>

          {status && !status.available ? (
            <div className="p-4 text-[12px] text-tcip-muted">
              {status.reason ??
                "The agent backend isn't configured. Install Claude Code and sign in to enable chat."}
            </div>
          ) : (
            <>
              <div ref={scrollRef} className="flex-1 overflow-auto p-3 flex flex-col gap-2">
                {items.length === 0 && (
                  <p className="text-[12px] text-tcip-muted">
                    Ask the agent about your project — training runs, review queues, a metric that
                    dropped. It works in your active project.
                  </p>
                )}
                {items.map((item, i) => {
                  if (item.kind === "user") {
                    return (
                      <div
                        key={i}
                        className="self-end max-w-[85%] rounded-lg bg-tcip-accent/20 px-2.5 py-1.5 text-[12px] text-tcip-fg whitespace-pre-wrap"
                      >
                        {item.text}
                      </div>
                    );
                  }
                  if (item.kind === "assistant") {
                    return (
                      <div
                        key={i}
                        className="self-start max-w-[95%] text-[12px] text-tcip-fg whitespace-pre-wrap"
                      >
                        {item.text}
                      </div>
                    );
                  }
                  if (item.kind === "tool") {
                    return (
                      <div
                        key={i}
                        className="self-start text-[11px] font-mono text-tcip-muted"
                        title={item.summary}
                      >
                        ▸ {item.name}
                        {item.ok === undefined ? "" : item.ok ? " — ok" : " — failed"}
                      </div>
                    );
                  }
                  // permission
                  const done = answered.has(item.requestId);
                  return (
                    <div
                      key={i}
                      className="self-stretch rounded-lg border border-tcip-warn/50 bg-tcip-warn/10 p-2.5 flex flex-col gap-2"
                    >
                      <div className="text-[12px] text-tcip-fg">
                        Allow <span className="font-mono">{item.tool}</span>?
                      </div>
                      {item.detail && (
                        <div className="text-[11px] font-mono text-tcip-muted break-words">
                          {item.detail}
                        </div>
                      )}
                      <div className="flex gap-2">
                        <button
                          className="tcip-btn-primary h-7 flex-1"
                          disabled={done}
                          onClick={() => decide(item.requestId, "allow")}
                        >
                          Allow
                        </button>
                        <button
                          className="tcip-btn-danger h-7 flex-1"
                          disabled={done}
                          onClick={() => decide(item.requestId, "deny")}
                        >
                          Deny
                        </button>
                      </div>
                    </div>
                  );
                })}
                {error && <div className="text-[11px] text-tcip-fp">{error}</div>}
              </div>

              <div className="border-t border-tcip-border p-2 shrink-0 flex flex-col gap-1.5">
                <label className="flex items-center gap-1.5 text-[10px] text-tcip-muted select-none">
                  <input
                    type="checkbox"
                    checked={includeContext}
                    onChange={(e) => setIncludeContext(e.target.checked)}
                  />
                  Attach current view ({activeTab}
                  {datasetDate ? ` · ${datasetDate}` : ""})
                </label>
                <div className="flex gap-2 items-end">
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        void send();
                      }
                    }}
                    rows={1}
                    placeholder="Ask the agent…"
                    className="tcip-input flex-1 resize-none py-1.5 leading-snug"
                  />
                  {busy ? (
                    <button className="tcip-btn h-8" onClick={stop}>
                      Stop
                    </button>
                  ) : (
                    <button
                      className="tcip-btn-primary h-8"
                      onClick={send}
                      disabled={!input.trim()}
                    >
                      Send
                    </button>
                  )}
                </div>
              </div>
            </>
          )}
        </div>
      )}

      <button
        onClick={() => setOpen((o) => !o)}
        aria-label="Toggle agent chat"
        className="relative h-9 w-9 rounded-full bg-tcip-accent text-white text-[15px] shadow-md hover:bg-tcip-accent-hover"
      >
        ✦
        {unread && !open && (
          <span className="absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-tcip-warn border border-tcip-panel" />
        )}
      </button>
    </div>
  );
}
