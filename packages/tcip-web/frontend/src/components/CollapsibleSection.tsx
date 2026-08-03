/**
 * The app's collapsible-section primitive: one chevron glyph and one trigger+content unit.
 *
 * `CollapsibleSection` runs either uncontrolled (it owns a `useDisclosure`, optionally persisted
 * under `storageKey`) or controlled (the caller supplies `open`/`onToggle`). The trigger is a real
 * button wired to the content region with `aria-expanded`/`aria-controls`, and the content unmounts
 * while closed. A section whose trigger and content live in different parts of the tree (a toolbar
 * shelf) uses `useDisclosure` + `DisclosureChevron` directly instead.
 */

import { useId, type ReactNode } from "react";

import { useDisclosure } from "@/hooks/useDisclosure";

export function DisclosureChevron({ open }: { open: boolean }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width="11"
      height="11"
      fill="none"
      aria-hidden="true"
      className={`shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
    >
      <path
        d="M6 4l4 4-4 4"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

type CollapsibleSectionProps = {
  title: ReactNode;
  /** Stays visible when the section is collapsed (a count, a status chip). */
  right?: ReactNode;
  /** A line under the trigger that stays visible when the section is collapsed. */
  caption?: ReactNode;
  children: ReactNode;
  className?: string;
} & (
  | { open: boolean; onToggle: () => void; storageKey?: never; defaultOpen?: never }
  | { open?: never; onToggle?: never; storageKey?: string; defaultOpen?: boolean }
);

export function CollapsibleSection({
  title,
  right,
  caption,
  children,
  className,
  open,
  onToggle,
  storageKey,
  defaultOpen = false,
}: CollapsibleSectionProps) {
  const own = useDisclosure(storageKey, defaultOpen);
  const isControlled = open !== undefined;
  const isOpen = isControlled ? open : own.open;
  const toggle = isControlled ? onToggle : own.toggle;
  const contentId = useId();

  return (
    <div className={className}>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={toggle}
          aria-expanded={isOpen}
          aria-controls={contentId}
          className="flex flex-1 items-center gap-2 text-left text-tcip-muted hover:text-tcip-fg"
        >
          <DisclosureChevron open={isOpen} />
          <span className="tcip-heading">{title}</span>
        </button>
        {right && <div className="text-[11px] text-tcip-muted">{right}</div>}
      </div>
      {caption && <p className="mt-1 text-[11px] text-tcip-muted">{caption}</p>}
      {isOpen && (
        <div id={contentId} className="mt-3">
          {children}
        </div>
      )}
    </div>
  );
}
