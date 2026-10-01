/**
 * A reconnecting WebSocket: capped exponential backoff on an unexpected close, at most one
 * attempt open or connecting, a replaced socket's late events ignored, and a restartable
 * start/stop pair. Frames arrive as raw text.
 */

export const MAX_BACKOFF_MS = 15_000;

export interface ReconnectingSocketOptions {
  /** A fixed URL, or a provider, possibly async, called before each attempt and reconnect. */
  url: string | (() => string | Promise<string>);
  onMessage: (data: string) => void;
  /** True once a frame marks the stream over; the helper then stops reconnecting. */
  isTerminal?: (data: string) => boolean;
  /** True when a frame resets the backoff to its floor; without it, opening resets it. */
  resetsBackoff?: (data: string) => boolean;
  onConnecting?: () => void;
  onOpen?: () => void;
  /** `opened` is whether this attempt ever reached onopen. */
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

/** The `onMessage`/`isTerminal`/`resetsBackoff` hooks for JSON frames, each frame parsed once. */
export function jsonFrameHandlers<T>(
  onMessage: (frame: T) => void,
  isTerminal?: (frame: T) => boolean,
  resetsBackoff?: (frame: T) => boolean,
): Pick<ReconnectingSocketOptions, "onMessage" | "isTerminal" | "resetsBackoff"> {
  let parsedFor: string | undefined;
  let parsed: T | undefined;
  const parse = (data: string): T | undefined => {
    if (data !== parsedFor) {
      parsedFor = data;
      try {
        parsed = JSON.parse(data) as T;
      } catch {
        parsed = undefined;
      }
    }
    return parsed;
  };
  return {
    onMessage: (data: string) => {
      const frame = parse(data);
      if (frame !== undefined) onMessage(frame);
    },
    isTerminal: isTerminal
      ? (data: string) => {
          const frame = parse(data);
          return frame !== undefined && isTerminal(frame);
        }
      : undefined,
    resetsBackoff: resetsBackoff
      ? (data: string) => {
          const frame = parse(data);
          return frame !== undefined && resetsBackoff(frame);
        }
      : undefined,
  };
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
      // Only a provider that returns a promise costs an await; a synchronous one must not pick
      // up a microtask tick the equivalent fixed string never had.
      const resolved = typeof opts.url === "function" ? opts.url() : opts.url;
      url = resolved instanceof Promise ? await resolved : resolved;
    } catch {
      // A throwing provider ends the loop rather than scheduling a retry: the caller's own
      // onError decides whether/how to recover.
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
      if (!opts.resetsBackoff) backoff = 500;
      opts.onOpen?.();
    };
    socket.onmessage = (ev: MessageEvent) => {
      if (socket !== ws) return;
      if (typeof ev.data !== "string") return;
      if (opts.resetsBackoff?.(ev.data)) backoff = 500;
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
      // Cleared here too, not just on open/close: a stop() while still CONNECTING must not
      // leave the guard set, or the start() that follows returns at it forever.
      connecting = false;
      const socket = ws;
      ws = null; // supersede: the closing socket's own handlers become no-ops
      socket?.close();
    },
    send(data: string) {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(data);
    },
  };
}
