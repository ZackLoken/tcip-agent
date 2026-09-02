import { useEffect } from "react";

import { useStore } from "@/store";
import type { Toast } from "@/store";

const LEVEL_CLASS: Record<Toast["level"], string> = {
  error: "border-tcip-fp/50 text-tcip-fp",
  info: "border-tcip-border text-tcip-fg",
  success: "border-tcip-tp/50 text-tcip-tp",
};

const DISMISS_LABEL_WORDS = 6;

/** The dismiss button's own name, so two toasts on screen at once never share one control
 * name: the message's first few words, the whole message when it is already that short. */
function dismissLabel(message: string): string {
  const words = message.trim().split(/\s+/);
  const lead = words.slice(0, DISMISS_LABEL_WORDS).join(" ");
  return `Dismiss: ${lead}${words.length > DISMISS_LABEL_WORDS ? "…" : ""}`;
}

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: number) => void }) {
  useEffect(() => {
    const id = window.setTimeout(() => onDismiss(toast.id), 6000);
    return () => window.clearTimeout(id);
  }, [toast.id, onDismiss]);

  const isError = toast.level === "error";
  return (
    <div
      role={isError ? "alert" : "status"}
      aria-live={isError ? "assertive" : "polite"}
      className={`flex items-start gap-2 rounded-md border bg-tcip-panel/95 px-3 py-2 text-[12px] shadow-lg ${LEVEL_CLASS[toast.level]}`}
    >
      <span className="flex-1 break-words">
        {toast.message}
        {toast.count && toast.count > 1 ? ` (×${toast.count})` : ""}
      </span>
      <button
        className="text-tcip-muted hover:text-tcip-fg"
        onClick={() => onDismiss(toast.id)}
        aria-label={dismissLabel(toast.message)}
      >
        ×
      </button>
    </div>
  );
}

/** Fixed-position transient notifications (API failures, etc.), driven by the store. */
export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  const dismiss = useStore((s) => s.dismissToast);
  if (!toasts.length) return null;
  return (
    <div className="fixed bottom-4 right-4 z-50 flex w-80 max-w-[calc(100vw-2rem)] flex-col gap-2">
      {toasts.map((t) => (
        <ToastItem key={t.id} toast={t} onDismiss={dismiss} />
      ))}
    </div>
  );
}
