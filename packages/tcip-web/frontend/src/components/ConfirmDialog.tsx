/**
 * A modal dialog shell for a destructive or otherwise consequential confirmation: a real
 * focus trap (Tab/Shift+Tab stay inside), focus moved to the first control on open and
 * returned to the button that opened it on close, Escape closes without confirming, and no
 * backdrop click dismisses it (a destructive dialog must not close on a stray click). No
 * dependency: the trap and the restore are implemented here rather than pulled in from a
 * library. Callers supply the body (a refusal, a warning, a name field, the confirm control)
 * as children; this component owns only the dialog's own accessibility contract.
 *
 * Rendered through a portal into a container that is a sibling of the application root under
 * `document.body`, created on demand: the app mounts its whole tree as the one child of
 * `#root`, so a dialog rendered inside that tree could never mark anything around itself
 * inert. Marking `#root` itself inert while the dialog is open (restored on close and on
 * unmount) keeps a screen reader or a stray Tab out of the content behind the modal, since
 * the dialog itself now lives outside it.
 */

import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";

const FOCUSABLE =
  "button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), " +
  'textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

const DIALOG_HOST_ID = "tcip-dialog-host";

function dialogHost(): HTMLElement {
  const existing = document.getElementById(DIALOG_HOST_ID);
  if (existing) return existing;
  const host = document.createElement("div");
  host.id = DIALOG_HOST_ID;
  document.body.appendChild(host);
  return host;
}

interface Props {
  heading: string;
  onClose: () => void;
  children: React.ReactNode;
  /** Set while the dialog is still waiting on something (a preview fetch); carried as
   *  aria-busy so assistive technology knows the content is not yet final. */
  busy?: boolean;
}

export function ConfirmDialog({ heading, onClose, children, busy }: Props) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const headingId = useId();
  const openerRef = useRef<HTMLElement | null>(null);
  const [host] = useState(dialogHost);

  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    const appRoot = document.getElementById("root");
    appRoot?.setAttribute("inert", "");
    return () => {
      appRoot?.removeAttribute("inert");
    };
  }, []);

  // Recorded once on mount, restored once on unmount: kept out of the trap effect below, whose
  // own onClose reference changes on every parent render.
  useEffect(() => {
    openerRef.current = document.activeElement as HTMLElement | null;
    return () => {
      openerRef.current?.focus();
    };
  }, []);

  useEffect(() => {
    const node = dialogRef.current;
    const first = node?.querySelector<HTMLElement>(FOCUSABLE);
    first?.focus();

    function onKeyDown(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onCloseRef.current();
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
    };
  }, []);

  return createPortal(
    <div className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center">
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={headingId}
        aria-busy={busy ? "true" : undefined}
        className="tcip-panel rounded-lg p-5 w-[440px] max-h-[85vh] overflow-auto"
      >
        <h2 id={headingId} className="text-[13px] font-semibold mb-3">
          {heading}
        </h2>
        {children}
      </div>
    </div>,
    host,
  );
}
