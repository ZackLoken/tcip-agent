import { useSyncExternalStore } from "react";

/**
 * This display's device-pixel count, the input the server derives every display bound from: the
 * screen's CSS size times its device pixel ratio. The screen rather than the canvas host, so a panel
 * or window resize does not change what an image request asks for.
 */
function displayPixels(): number {
  const ratio = window.devicePixelRatio;
  return Math.round(window.screen.width * ratio) * Math.round(window.screen.height * ratio);
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener("resize", onChange);
  return () => window.removeEventListener("resize", onChange);
}

/** The one display context a view's image and view requests carry, read again whenever the
 *  window resizes, which is when a move to another screen changes it. */
export function useDisplayPixels(): number {
  return useSyncExternalStore(subscribe, displayPixels);
}
