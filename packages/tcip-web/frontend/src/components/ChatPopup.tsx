import { useState } from "react";

/**
 * Feature flag for the in-app agent chat pop-up (Phase C0 stub).
 *
 * The agent-driving backend does not exist yet — see
 * packages/tcip-web/docs/chat-popup-design.md for the design and phased plan.
 * Flip to true only when Phase C1 (chat HTTP/WS surface) lands; the flag is
 * removed entirely when Phase C2 (agent sidecar) ships.
 */
export const CHAT_POPUP_ENABLED = false;

interface ChatPopupProps {
  /** Overridable for tests; defaults to the module flag. */
  enabled?: boolean;
}

/**
 * Floating agent-chat launcher + panel shell. While disabled (the default) it
 * renders nothing, so mounting it in App has zero effect on existing behavior.
 * The enabled shell is a static placeholder — no network calls, no store writes.
 */
export function ChatPopup({ enabled = CHAT_POPUP_ENABLED }: ChatPopupProps) {
  const [open, setOpen] = useState(false);

  if (!enabled) return null;

  return (
    <div className="fixed bottom-9 right-3 z-40 flex flex-col items-end gap-2">
      {open && (
        <div className="w-[380px] rounded-lg border border-tcip-border bg-tcip-panel shadow-lg">
          <div className="flex items-center justify-between px-3 h-8 border-b border-tcip-border">
            <span className="text-[12px] font-medium text-tcip-fg">Agent chat</span>
            <button
              onClick={() => setOpen(false)}
              className="text-tcip-muted hover:text-tcip-fg text-[12px] px-1"
              aria-label="Close agent chat"
            >
              ✕
            </button>
          </div>
          <div className="p-3 text-[12px] text-tcip-muted">
            The agent chat backend is not implemented yet. See{" "}
            <span className="font-mono">docs/chat-popup-design.md</span> for the design and
            phased plan.
          </div>
          <div className="p-2 border-t border-tcip-border">
            <input
              disabled
              placeholder="Agent backend not configured"
              className="w-full h-7 px-2 rounded border border-tcip-border bg-tcip-bg text-[12px] text-tcip-muted"
            />
          </div>
        </div>
      )}
      <button
        onClick={() => setOpen((o) => !o)}
        aria-label="Toggle agent chat"
        className="h-9 w-9 rounded-full bg-tcip-accent text-white text-[15px] shadow-md hover:opacity-90"
      >
        ✦
      </button>
    </div>
  );
}
