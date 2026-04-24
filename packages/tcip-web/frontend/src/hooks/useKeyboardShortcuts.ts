import { useEffect } from "react";

export interface Shortcut {
  keys: string; // space-separated e.g. "Ctrl+z", "a", "ArrowLeft"
  action: (e: KeyboardEvent) => void;
  when?: () => boolean;
}

function matches(e: KeyboardEvent, keys: string): boolean {
  const parts = keys.split("+").map((p) => p.trim().toLowerCase());
  const key = parts[parts.length - 1];
  const wants = {
    ctrl: parts.includes("ctrl"),
    shift: parts.includes("shift"),
    alt: parts.includes("alt"),
    meta: parts.includes("meta"),
  };
  const eKey = e.key.toLowerCase();
  return (
    eKey === key &&
    e.ctrlKey === wants.ctrl &&
    e.shiftKey === wants.shift &&
    e.altKey === wants.alt &&
    e.metaKey === wants.meta
  );
}

export function useKeyboardShortcuts(shortcuts: Shortcut[]): void {
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const tgt = e.target as HTMLElement | null;
      if (tgt && (tgt.tagName === "INPUT" || tgt.tagName === "TEXTAREA" || tgt.isContentEditable)) {
        return;
      }
      for (const s of shortcuts) {
        if (s.when && !s.when()) continue;
        if (matches(e, s.keys)) {
          e.preventDefault();
          s.action(e);
          return;
        }
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [shortcuts]);
}
