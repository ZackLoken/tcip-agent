/**
 * Open/closed state for a collapsible region, optionally remembered across sessions.
 *
 * With a `storageKey` the state is mirrored into localStorage ("1"/"0"), written only when the
 * user actually toggles, so a panel never persists a state nobody chose. A missing key falls back
 * to `defaultOpen`. Without a `storageKey` it is plain component state.
 */

import { useCallback, useState } from "react";

export interface Disclosure {
  open: boolean;
  toggle: () => void;
  setOpen: (next: boolean) => void;
}

export function useDisclosure(storageKey?: string, defaultOpen = false): Disclosure {
  const [open, setOpenState] = useState<boolean>(() => {
    if (!storageKey) return defaultOpen;
    try {
      const raw = localStorage.getItem(storageKey);
      return raw === null ? defaultOpen : raw === "1";
    } catch {
      return defaultOpen;
    }
  });

  const persist = useCallback(
    (next: boolean) => {
      if (!storageKey) return;
      try {
        localStorage.setItem(storageKey, next ? "1" : "0");
      } catch {
        /* private mode / disabled storage: the section just won't persist */
      }
    },
    [storageKey],
  );

  const setOpen = useCallback(
    (next: boolean) => {
      setOpenState(next);
      persist(next);
    },
    [persist],
  );

  const toggle = useCallback(() => setOpen(!open), [open, setOpen]);

  return { open, toggle, setOpen };
}
