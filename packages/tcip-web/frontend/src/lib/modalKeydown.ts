const fellInModal = new WeakSet<KeyboardEvent>();

/** Marks every keydown dispatched from now on as one that fell while a modal was open, from the
 *  window's capture phase, ahead of every other listener and of the modal's own Escape handling
 *  that closes it. Returns the function that stops marking. */
export function markKeydownsInModal(): () => void {
  const mark = (e: KeyboardEvent) => {
    fellInModal.add(e);
  };
  window.addEventListener("keydown", mark, true);
  return () => window.removeEventListener("keydown", mark, true);
}

/** Whether a keydown was dispatched while a modal dialog was open. The window's key listeners
 *  that act on the page behind a modal read it and leave the key to the modal. */
export function keydownInModal(e: KeyboardEvent): boolean {
  return fellInModal.has(e);
}

/** The class a docked pane's tab carries as its content class, which marks the panel the docking
 *  layout renders for it and everything inside as the pane's. */
export const DOCKED_PANE_CLASS = "docked-pane";

/** Whether a keydown's target is a docked pane's panel or lies inside one. A pane outside the
 *  work area takes focus on something that is not a control, and the page's key listeners that
 *  act on the work area read it and leave the key to the pane. */
export function keydownInPane(e: KeyboardEvent): boolean {
  return e.target instanceof Element && e.target.closest(`.${DOCKED_PANE_CLASS}`) !== null;
}
