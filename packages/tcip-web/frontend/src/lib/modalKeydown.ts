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
