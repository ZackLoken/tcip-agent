/**
 * A modal dialog shell for a destructive or otherwise consequential confirmation: a real
 * focus trap (Tab/Shift+Tab stay inside), focus moved to the first control on open and
 * returned to the button that opened it on close, Escape closes without confirming, and no
 * backdrop click dismisses it (a destructive dialog must not close on a stray click). No
 * dependency: the trap and the restore are implemented here rather than pulled in from a
 * library. Callers supply the body (a refusal, a warning, a name field, the confirm control)
 * as children; this component owns only the dialog's own accessibility contract.
 */

import { useEffect, useId, useRef } from "react";

const FOCUSABLE =
  "button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), " +
  'textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

interface Props {
  heading: string;
  onClose: () => void;
  children: React.ReactNode;
}

export function ConfirmDialog({ heading, onClose, children }: Props) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const headingId = useId();
  const openerRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    openerRef.current = document.activeElement as HTMLElement | null;
    const node = dialogRef.current;
    const first = node?.querySelector<HTMLElement>(FOCUSABLE);
    first?.focus();

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key !== "Tab" || !node) return;
      const focusable = Array.from(node.querySelectorAll<HTMLElement>(FOCUSABLE));
      if (focusable.length === 0) return;
      const firstEl = focusable[0];
      const lastEl = focusable[focusable.length - 1];
      if (e.shiftKey && document.activeElement === firstEl) {
        e.preventDefault();
        lastEl.focus();
      } else if (!e.shiftKey && document.activeElement === lastEl) {
        e.preventDefault();
        firstEl.focus();
      }
    }
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      openerRef.current?.focus();
    };
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={headingId}
        className="tcip-panel rounded-lg p-5 w-[440px] max-h-[85vh] overflow-auto"
      >
        <div id={headingId} className="text-[13px] font-semibold mb-3">
          {heading}
        </div>
        {children}
      </div>
    </div>
  );
}
