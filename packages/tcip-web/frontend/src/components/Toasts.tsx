import { useEffect } from "react";

import { dismissLabel } from "@/lib/dismissLabel";
import { useStore } from "@/store";
import type { Toast } from "@/store";
import { LEVELS, textOf } from "@/store/slices/toasts";

function ToastItem({ toast, onDismiss }: { toast: Toast; onDismiss: (id: number) => void }) {
  useEffect(() => {
    const id = window.setTimeout(() => onDismiss(toast.id), 6000);
    return () => window.clearTimeout(id);
  }, [toast.id, onDismiss]);

  const { live, tone } = LEVELS[toast.level];
  const text = textOf(toast);
  return (
    <div
      role={live === "assertive" ? "alert" : "status"}
      aria-live={live}
      className={`flex items-start gap-2 rounded-md border border-tcip-border bg-tcip-panel/95 px-3 py-2 shadow-lg ${tone}`}
    >
      <span className="flex-1 break-words">
        {text}
        {toast.count && toast.count > 1 ? ` (×${toast.count})` : ""}
      </span>
      <button
        className="text-tcip-muted hover:text-tcip-fg"
        onClick={() => onDismiss(toast.id)}
        aria-label={dismissLabel(text)}
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
