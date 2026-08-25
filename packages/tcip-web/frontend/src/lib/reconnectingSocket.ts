/**
 * One reconnecting-WebSocket shape, shared by every socket this app opens: capped exponential
 * backoff on an unexpected close, a guard against stacking a second attempt while one is already
 * open or connecting, supersession so a replaced socket's late events are no-ops, and a
 * restartable start/stop pair. A caller keeps only what differs: its own URL, its own frame
 * handling (raw text here; JSON parsing, if any, is the caller's job), and whether/when a frame
 * marks the stream terminal.
 */

export const MAX_BACKOFF_MS = 15_000;

export interface ReconnectingSocketOptions {
  /** A fixed URL, or a provider called before each attempt (including each reconnect); async so
   *  a caller can create a server-side session first and open the socket only once it resolves. */
  url: string | (() => string | Promise<string>);
  onMessage: (data: string) => void;
  /** True once a frame marks the stream over; the helper then stops reconnecting. */
  isTerminal?: (data: string) => boolean;
  onConnecting?: () => void;
  onOpen?: () => void;
  /** `opened` is whether this attempt ever reached onopen, so a caller can tell a close-before-
   *  open (or a code like 1008) apart from a drop mid-session. */
  onClose?: (event: CloseEvent, opened: boolean) => void;
  onError?: () => void;
  maxBackoffMs?: number;
}

export interface ReconnectingSocket {
  /** Open a socket now (or re-open one after `stop()`); a no-op while one is already live. */
  start(): void;
  /** Close the live socket and cancel any pending reconnect; a later `start()` re-arms it. */
  stop(): void;
  send(data: string): void;
}

export function createReconnectingSocket(opts: ReconnectingSocketOptions): ReconnectingSocket {
  const maxBackoff = opts.maxBackoffMs ?? MAX_BACKOFF_MS;
  let ws: WebSocket | null = null;
  let stopped = true;
  let terminated = false;
  let connecting = false;
  let backoff = 500;
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  const clearReconnectTimer = () => {
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
  };

  const connect = async () => {
    if (stopped || terminated || connecting) return;
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
    connecting = true;
    opts.onConnecting?.();
    let url: string;
    try {
      url = typeof opts.url === "function" ? await opts.url() : opts.url;
    } catch {
      // A throwing provider ends the loop rather than scheduling a retry: the caller's own
      // onError decides whether/how to recover (this is the terminal rail's behavior today).
      connecting = false;
      opts.onError?.();
      return;
    }
    // The caller may have stopped this instance while the provider was pending.
    if (stopped) {
      connecting = false;
      return;
    }
    const socket = new WebSocket(url);
    ws = socket;
    let opened = false;
    socket.onopen = () => {
      if (socket !== ws) return;
      connecting = false;
      opened = true;
      backoff = 500;
      opts.onOpen?.();
    };
    socket.onmessage = (ev: MessageEvent) => {
      if (socket !== ws) return;
      if (typeof ev.data !== "string") return;
      if (opts.isTerminal?.(ev.data)) terminated = true;
      opts.onMessage(ev.data);
    };
    socket.onerror = () => {
      if (socket !== ws) return;
      opts.onError?.();
    };
    socket.onclose = (ev: CloseEvent) => {
      if (socket !== ws) return;
      connecting = false;
      ws = null;
      opts.onClose?.(ev, opened);
      if (stopped || terminated) return;
      const delay = backoff;
      backoff = Math.min(backoff * 2, maxBackoff);
      reconnectTimer = setTimeout(() => void connect(), delay);
    };
  };

  return {
    start() {
      stopped = false;
      terminated = false;
      void connect();
    },
    stop() {
      stopped = true;
      clearReconnectTimer();
      const socket = ws;
      ws = null; // supersede: the closing socket's own handlers become no-ops
      socket?.close();
    },
    send(data: string) {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(data);
    },
  };
}
