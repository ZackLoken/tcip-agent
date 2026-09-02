import { useEffect } from "react";

export interface Shortcut {
  keys: string; // space-separated e.g. "Ctrl+z", "a", "ArrowLeft"
  action: (e: KeyboardEvent) => void;
  when?: () => boolean;
}

const FOCUSABLE_ROLES = new Set(["button", "tab", "checkbox", "switch", "menuitem"]);

/** Whether a keydown's target is a widget that owns its own key handling (a native form
 *  control, a contenteditable, or anything carrying an interactive ARIA role): the app's
 *  shortcuts never intercept a keydown aimed at one, so Enter/Space still activate a focused
 *  button and arrow keys still move within a focused tab list or checkbox group. */
function isFocusableControl(tgt: EventTarget | null): boolean {
  if (!tgt || !(tgt instanceof Element)) return false;
  if (
    tgt.tagName === "INPUT" ||
    tgt.tagName === "TEXTAREA" ||
    tgt.tagName === "SELECT" ||
    tgt.tagName === "BUTTON" ||
    (tgt instanceof HTMLElement && tgt.isContentEditable)
  ) {
    return true;
  }
  const role = tgt.getAttribute("role");
  return !!role && FOCUSABLE_ROLES.has(role);
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
      // A control that owns its own key handling keeps it: Enter/Space still activate a button.
      if (isFocusableControl(e.target)) {
        return;
      }
      for (const s of shortcuts) {
        if (!matches(e, s.keys)) continue;
        if (s.when && !s.when()) {
          // A gated-off modifier combo still hits browser chrome (Ctrl+S on a locked image
          // opens the Save-Page dialog); a gated-off bare key is left alone.
          if (e.ctrlKey || e.metaKey) e.preventDefault();
          continue;
        }
        e.preventDefault();
        s.action(e);
        return;
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [shortcuts]);
}
