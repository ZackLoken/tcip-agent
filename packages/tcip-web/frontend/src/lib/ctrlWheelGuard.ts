/**
 * Stop the browser's own ctrl+wheel page zoom over the app, so the canvas' zoom is the only
 * zoom. Attached to the app root element, not document: an iframe-embedded tool (TensorBoard)
 * receives wheel events inside its own document, outside this listener, and keeps browser zoom. Plain (ctrl-less) wheel events pass through untouched.
 */
export function attachCtrlWheelGuard(el: HTMLElement): () => void {
  const onWheel = (e: WheelEvent) => {
    if (e.ctrlKey) e.preventDefault();
  };
  el.addEventListener("wheel", onWheel, { passive: false });
  return () => el.removeEventListener("wheel", onWheel);
}
