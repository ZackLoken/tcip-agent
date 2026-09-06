import { useEffect } from "react";

export interface Shortcut {
  keys: string; // space-separated e.g. "Ctrl+z", "a", "ArrowLeft"
  action: (e: KeyboardEvent) => void;
  when?: () => boolean;
  /** True for a shortcut that still fires while focus sits on the one control carrying
   *  `data-keyboard-passthrough` (the toolbar control that arms this shortcut's own precondition
   *  and keeps focus after activation, so the shortcut would otherwise be dead for the keyboard
   *  user who just used it). Every other focusable control still swallows the keys, exactly as
   *  it does for a shortcut with no exemption: the flag names one control, never a class of
   *  them, so a select, an input, a textarea, a contenteditable or an unrelated ARIA widget is
   *  never exempted by it. Default false. */
  whileFocused?: boolean;
}

const FOCUSABLE_ROLES = new Set(["button", "tab", "checkbox", "switch", "menuitem"]);

/** A native form control or contenteditable whose own key handling a shortcut never overrides,
 *  `whileFocused` included: typing must never be intercepted. */
function isTextEntryControl(tgt: EventTarget | null): boolean {
  if (!tgt || !(tgt instanceof Element)) return false;
  return (
    tgt.tagName === "INPUT" ||
    tgt.tagName === "TEXTAREA" ||
    tgt.tagName === "SELECT" ||
    (tgt instanceof HTMLElement && tgt.isContentEditable)
  );
}

/** Whether a keydown's target is a widget that owns its own key handling (a text-entry control
 *  or anything carrying an interactive ARIA role): the app's shortcuts never intercept a keydown
 *  aimed at one, so Enter/Space still activate a focused button and arrow keys still move within
 *  a focused tab list or checkbox group. A `whileFocused` shortcut is let past this check only
 *  when the target itself carries `data-keyboard-passthrough` (see `Shortcut.whileFocused`). */
function isFocusableControl(tgt: EventTarget | null): boolean {
  if (!tgt || !(tgt instanceof Element)) return false;
  if (isTextEntryControl(tgt) || tgt.tagName === "BUTTON") return true;
  const role = tgt.getAttribute("role");
  return !!role && FOCUSABLE_ROLES.has(role);
}

function allowsKeyboardPassthrough(tgt: EventTarget | null): boolean {
  return tgt instanceof Element && tgt.hasAttribute("data-keyboard-passthrough");
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
      // A text-entry control's own key handling is never overridden, whileFocused included.
      if (isTextEntryControl(e.target)) {
        return;
      }
      const focusable = isFocusableControl(e.target);
      const passthrough = allowsKeyboardPassthrough(e.target);
      for (const s of shortcuts) {
        if (!matches(e, s.keys)) continue;
        // Otherwise a focusable control keeps its own handling unless exempted and focused.
        if (focusable && !(s.whileFocused && passthrough)) {
          return;
        }
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
