/**
 * The one shell every modal dialog renders through, over Radix's Dialog: focus moves in on open
 * and returns to what had it on close, Tab stays inside, the page behind is hidden from assistive
 * technology and cannot be clicked, and Escape closes. The caller supplies the title, the body,
 * what closing does, and whether a click outside the panel closes it (`closeOnOutsideClick`,
 * required: a destructive confirmation refuses it).
 *
 * Focus returns to the element that had it when the shell mounted, since no trigger element
 * opens it. While the shell is mounted every keydown is marked as falling in a modal
 * (`keydownInModal`), so the page's own key listeners leave it alone.
 */

import * as Dialog from "@radix-ui/react-dialog";
import { useEffect, useState } from "react";

import { markKeydownsInModal } from "@/lib/modalKeydown";

interface Props {
  title: string;
  /** Runs on Escape, on an outside click when `closeOnOutsideClick`, and on the header's close
   *  button. */
  onClose: () => void;
  closeOnOutsideClick: boolean;
  /** Tailwind classes giving the panel its width. */
  className: string;
  /** Label of a close button beside the title; omitted where the body carries its own. */
  headerClose?: string;
  /** Set while the dialog is still waiting on something (a preview fetch); carried as
   *  aria-busy so assistive technology knows the content is not yet final. */
  busy?: boolean;
  /** Runs when focus is about to move in on open; preventing the event leaves the move to it. */
  onOpenAutoFocus?: (event: Event) => void;
  children: React.ReactNode;
}

export function ModalDialog({
  title,
  onClose,
  closeOnOutsideClick,
  className,
  headerClose,
  busy,
  onOpenAutoFocus,
  children,
}: Props) {
  useEffect(markKeydownsInModal, []);
  const [opener] = useState(() => document.activeElement);

  return (
    <Dialog.Root
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/60" />
        <Dialog.Content
          aria-describedby={undefined}
          aria-modal="true"
          aria-busy={busy ? "true" : undefined}
          onOpenAutoFocus={onOpenAutoFocus}
          onCloseAutoFocus={(e) => {
            e.preventDefault();
            if (opener instanceof HTMLElement) opener.focus();
          }}
          onInteractOutside={closeOnOutsideClick ? undefined : (e) => e.preventDefault()}
          className={`tcip-panel rounded-lg p-5 max-w-[90vw] max-h-[85vh] overflow-auto fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 ${className}`}
        >
          <div className="flex items-center justify-between mb-3">
            <Dialog.Title className="tcip-heading">{title}</Dialog.Title>
            {headerClose && (
              <Dialog.Close className="tcip-btn text-[13px]">{headerClose}</Dialog.Close>
            )}
          </div>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
