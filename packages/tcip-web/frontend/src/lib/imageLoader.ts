/**
 * Shared image loader for /api/images serves. Fetches so the response headers are readable (a
 * DOM `Image` reports only that a load failed), decodes the body into an image element, and
 * hands back the bitmap together with the serve facts the headers carry. Every object URL
 * created here is revoked once its bitmap decodes or the load is abandoned.
 */

export interface ImageServeFacts {
  /** Output size of the serve, parsed from X-TCIP-Served-Size ("WxH"). */
  servedSize: { w: number; h: number } | null;
  /** Where the display-stretch statistics came from (X-TCIP-Stats-Source). */
  statsSource: string | null;
  /** The display-bound stretch range the serve was rendered with (X-TCIP-Display-Bounds). */
  displayBounds: string | null;
  /** The condition the server named when it refused the request (X-TCIP-Image-Error). */
  imageError: string | null;
}

export interface LoadedImage extends ImageServeFacts {
  /** The decoded bitmap, or null when the load failed. */
  image: HTMLImageElement | null;
  ok: boolean;
  /** True when the caller's signal cancelled the load; no other field is meaningful then. */
  aborted: boolean;
}

const NO_FACTS: ImageServeFacts = {
  servedSize: null,
  statsSource: null,
  displayBounds: null,
  imageError: null,
};

function parseServedSize(raw: string | null): { w: number; h: number } | null {
  if (!raw) return null;
  const m = /^(\d+)x(\d+)$/.exec(raw.trim());
  return m ? { w: Number(m[1]), h: Number(m[2]) } : null;
}

function readFacts(headers: Headers): ImageServeFacts {
  return {
    servedSize: parseServedSize(headers.get("X-TCIP-Served-Size")),
    statsSource: headers.get("X-TCIP-Stats-Source"),
    displayBounds: headers.get("X-TCIP-Display-Bounds"),
    imageError: headers.get("X-TCIP-Image-Error"),
  };
}

function decodeBlobImage(objectUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = () => reject(new Error("image decode failed"));
    el.src = objectUrl;
  });
}

function wasAborted(e: unknown): boolean {
  return e instanceof DOMException && e.name === "AbortError";
}

/** Load one image serve; never throws. Pass a signal to abandon the load. */
export async function loadImage(
  url: string,
  opts: { signal?: AbortSignal } = {},
): Promise<LoadedImage> {
  let resp: Response;
  try {
    resp = await fetch(url, { signal: opts.signal });
  } catch (e) {
    return { ...NO_FACTS, image: null, ok: false, aborted: wasAborted(e) };
  }
  const facts = readFacts(resp.headers);
  if (!resp.ok) return { ...facts, image: null, ok: false, aborted: false };
  let blob: Blob;
  try {
    blob = await resp.blob();
  } catch (e) {
    return { ...facts, image: null, ok: false, aborted: wasAborted(e) };
  }
  const objectUrl = URL.createObjectURL(blob);
  try {
    const image = await decodeBlobImage(objectUrl);
    return { ...facts, image, ok: true, aborted: false };
  } catch {
    return { ...facts, image: null, ok: false, aborted: false };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}
