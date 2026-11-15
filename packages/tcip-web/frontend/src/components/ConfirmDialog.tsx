/**
 * A modal dialog for a destructive or otherwise consequential confirmation, rendered through
 * the shared modal shell: Escape closes without confirming, and no outside click dismisses it (a
 * destructive dialog must not close on a stray click). Callers supply the body (a refusal, a
 * warning, a name field, the confirm control) as children.
 */

import { ModalDialog } from "@/components/ModalDialog";

interface Props {
  heading: string;
  onClose: () => void;
  children: React.ReactNode;
  /** Set while the dialog is still waiting on something (a preview fetch); carried as
   *  aria-busy so assistive technology knows the content is not yet final. */
  busy?: boolean;
}

export function ConfirmDialog({ heading, onClose, children, busy }: Props) {
  return (
    <ModalDialog
      title={heading}
      onClose={onClose}
      closeOnOutsideClick={false}
      className="w-[440px]"
      busy={busy}
    >
      {children}
    </ModalDialog>
  );
}
