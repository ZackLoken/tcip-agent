/**
 * Shared image loader for /api/images serves. Fetches so the response headers are readable (a
 * DOM `Image` reports only that a load failed), decodes the body into an image element, and
 * hands back the bitmap together with the serve facts the headers carry. Every object URL
 * created here is revoked once its bitmap decodes or the load is abandoned.
 */

import { IMAGE_ERROR_HEADER } from "@/api/types.generated";
import type { CoverageViewing, StatsSource } from "@/api/types.generated";

export interface ImageServeFacts {
  /** Output size of the serve, parsed from X-TCIP-Served-Size ("WxH"). */
  servedSize: { w: number; h: number } | null;
  /** The raw X-TCIP-Served-Size string, byte-identical to what the server sent. */
  servedSizeRaw: string | null;
  /** Where the display-stretch statistics came from, parsed from X-TCIP-Stats-Source. */
  statsSource: StatsSource | null;
  /** The display-bound stretch range the serve was rendered with, parsed from
   *  X-TCIP-Display-Bounds: one pair per displayed band, positional against `bands`. A `null`
   *  half of a pair is a band whose pixels left the stretch no finite value to report. */
  displayBounds: CoverageViewing["display_bounds"];
  /** The condition the server named when it refused the request (X-TCIP-Image-Error). */
  imageError: string | null;
}

export interface LoadedImage extends ImageServeFacts {
  /** The decoded bitmap, or null when the load failed. */
  image: HTMLImageElement | null;
  ok: boolean;
  /** True when the caller's signal cancelled the load; no other field is meaningful then. */
  aborted: boolean;
  /** Which header failed to parse as JSON, when that is why `ok` is false; null otherwise. */
  headerParseError: string | null;
}

const NO_FACTS: ImageServeFacts = {
  servedSize: null,
  servedSizeRaw: null,
  statsSource: null,
  displayBounds: null,
  imageError: null,
};

function parseServedSize(raw: string | null): { w: number; h: number } | null {
  if (!raw) return null;
  const m = /^(\d+)x(\d+)$/.exec(raw.trim());
  return m ? { w: Number(m[1]), h: Number(m[2]) } : null;
}

type FactsResult = { ok: true; facts: ImageServeFacts } | { ok: false; error: string };

/** Parses `X-TCIP-Stats-Source` and `X-TCIP-Display-Bounds` as the JSON they now carry, refusing
 *  by header name rather than silently dropping a value that will not parse. */
function readFacts(headers: Headers): FactsResult {
  const statsSourceRaw = headers.get("X-TCIP-Stats-Source");
  let statsSource: StatsSource | null = null;
  if (statsSourceRaw !== null) {
    try {
      statsSource = JSON.parse(statsSourceRaw) as StatsSource;
    } catch {
      return { ok: false, error: `X-TCIP-Stats-Source did not parse: ${statsSourceRaw}` };
    }
  }

  const displayBoundsRaw = headers.get("X-TCIP-Display-Bounds");
  let displayBounds: CoverageViewing["display_bounds"] = null;
  if (displayBoundsRaw !== null) {
    try {
      displayBounds = JSON.parse(displayBoundsRaw) as CoverageViewing["display_bounds"];
    } catch {
      return { ok: false, error: `X-TCIP-Display-Bounds did not parse: ${displayBoundsRaw}` };
    }
  }

  const servedSizeRaw = headers.get("X-TCIP-Served-Size");
  return {
    ok: true,
    facts: {
      servedSize: parseServedSize(servedSizeRaw),
      servedSizeRaw,
      statsSource,
      displayBounds,
      imageError: headers.get(IMAGE_ERROR_HEADER),
    },
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
    return {
      ...NO_FACTS,
      image: null,
      ok: false,
      aborted: wasAborted(e),
      headerParseError: null,
    };
  }
  const parsed = readFacts(resp.headers);
  if (!parsed.ok) {
    return {
      ...NO_FACTS,
      image: null,
      ok: false,
      aborted: false,
      headerParseError: parsed.error,
    };
  }
  const facts = parsed.facts;
  if (!resp.ok) return { ...facts, image: null, ok: false, aborted: false, headerParseError: null };
  let blob: Blob;
  try {
    blob = await resp.blob();
  } catch (e) {
    return { ...facts, image: null, ok: false, aborted: wasAborted(e), headerParseError: null };
  }
  const objectUrl = URL.createObjectURL(blob);
  try {
    const image = await decodeBlobImage(objectUrl);
    return { ...facts, image, ok: true, aborted: false, headerParseError: null };
  } catch {
    return { ...facts, image: null, ok: false, aborted: false, headerParseError: null };
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}
