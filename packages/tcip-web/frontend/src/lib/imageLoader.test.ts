import { afterEach, describe, expect, it, vi } from "vitest";

import { IMAGE_ERROR_HEADER, OVERVIEWS_REQUIRED } from "@/api/types.generated";
import { loadImage, readFacts } from "@/lib/imageLoader";
import { IMAGE_ANSWERS } from "@/test/images";
import { fetchAnswers } from "@/test/refusals";

/** An image element that decodes whatever it is given at once, as jsdom loads no image. */
class DecodedImage {
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  set src(_url: string) {
    queueMicrotask(() => this.onload?.());
  }
}

/** Make every image fetch answer the produced image answer `served`. */
function answering(served: { status: number; headers: Record<string, string> }) {
  fetchAnswers(served.status, undefined, served.headers);
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("loadImage header parsing", () => {
  it("parses the served-size header into its pair", async () => {
    const served = IMAGE_ANSWERS.image;
    const { servedSize } = readFacts(new Headers(served.headers));
    expect(servedSize).not.toBeNull();
    answering(served);
    vi.stubGlobal("Image", DecodedImage);
    vi.stubGlobal("URL", { createObjectURL: () => "blob:served", revokeObjectURL: () => {} });
    const result = await loadImage("/api/images?path=x");
    expect(result.ok).toBe(true);
    expect(result.servedSize).toEqual(servedSize);
  });

  it("names the refusal condition the server sent", async () => {
    const refused = IMAGE_ANSWERS.image_overviews_required;
    expect(refused.headers[IMAGE_ERROR_HEADER]).toBe(OVERVIEWS_REQUIRED);
    answering(refused);
    const result = await loadImage("/api/images?path=x");
    expect(result.ok).toBe(false);
    expect(result.imageError).toBe(OVERVIEWS_REQUIRED);
    expect(result.servedSize).toBeNull();
  });

  it("names no condition for a refusal that carries none", async () => {
    answering(IMAGE_ANSWERS.image_missing);
    const result = await loadImage("/api/images?path=x");
    expect(result.ok).toBe(false);
    expect(result.imageError).toBeNull();
  });
});
