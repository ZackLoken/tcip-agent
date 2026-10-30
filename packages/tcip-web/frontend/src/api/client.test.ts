import { renderHook } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import { refusalOf } from "@/api/http";
import {
  NO_GESTURES,
  PROJECT_NOT_OPEN,
  type ProjectNotOpenRefusal,
  RENDER_CACHE_VERSION,
  type SavePayload,
  VERSION_CONFLICT,
  type VersionConflictRefusal,
} from "@/api/types.generated";
import { useDisplayPixels } from "@/hooks/useDisplayPixels";
import { BAND_ANSWERS } from "@/test/bands";
import { canvasBody } from "@/test/canvasBody";
import { LABEL_ANSWERS } from "@/test/labels";
import { PROJECT_FIXTURES } from "@/test/projectRecords";
import { fetchAnswers } from "@/test/refusals";

const conflict = (e: unknown) => refusalOf<VersionConflictRefusal>(e, VERSION_CONFLICT);
const mismatch = (e: unknown) => refusalOf<ProjectNotOpenRefusal>(e, PROJECT_NOT_OPEN);

/** The display pixel count the client's own producer reads off the test screen. */
function display(): number {
  return renderHook(() => useDisplayPixels()).result.current;
}

describe("annotate.save lost-update handling", () => {
  const request: SavePayload = {
    image_path: "x",
    annotations: [],
    base_mtime: "1",
    user: "breeder",
    gestures: NO_GESTURES,
  };

  it("returns the decoded document with its version token as the string it was served", async () => {
    // A version token is a string: a nanosecond mtime as a JSON number would be rounded by
    // JSON.parse and every echo would 409.
    const { saved } = LABEL_ANSWERS;
    fetchAnswers(200, saved);
    const res = await api.annotate.save({ ...request, base_mtime: null });
    expect(res.document.base_mtime).toBe(saved.document.base_mtime);
    expect(res.document.points).toHaveLength(1);
    expect(res.accepted).toEqual(saved.accepted);
  });

  it("throws the version-conflict 409, which reads as that refusal", async () => {
    const { status, body } = LABEL_ANSWERS.version_conflict;
    fetchAnswers(status, body);
    const raised = await api.annotate.save(request).catch((e: unknown) => e);
    expect(conflict(raised)).toEqual(body.detail);
    expect(mismatch(raised)).toBeNull();
  });

  it("throws on a refusal whose body carries no detail, as every JSON call does", async () => {
    // A transport-level refusal: a body no route wrote.
    const { status } = LABEL_ANSWERS.version_conflict;
    fetchAnswers(status, {});
    const save = api.annotate.save(request);
    await expect(save).rejects.toThrow(String(status));
  });
});

describe("canvas.pushState 409 recovery", () => {
  it("returns the backend's empty answer on a 200", async () => {
    fetchAnswers(200, null);
    const res = await api.canvas.pushState(canvasBody());
    expect(res).toBeNull();
  });

  it("throws the backend's project-not-open 409, which reads as that refusal", async () => {
    const { status, body } = PROJECT_FIXTURES.project_not_open;
    fetchAnswers(status, body);
    const raised = await api.canvas.pushState(canvasBody()).catch((e: unknown) => e);
    expect(mismatch(raised)).toEqual(body.detail);
    expect(conflict(raised)).toBeNull();
  });
});

describe("annotate.load", () => {
  it("reports the raster's own width and height, each in its own field", async () => {
    // Non-square on purpose: equal dimensions would read the same whichever field they land in.
    const { loaded } = LABEL_ANSWERS;
    expect(loaded.img_width).not.toBe(loaded.img_height);
    fetchAnswers(200, loaded);
    const labels = await api.annotate.load(loaded.image_path);
    expect(labels.img_width).toBe(loaded.img_width);
    expect(labels.img_height).toBe(loaded.img_height);
    expect(labels.base_mtime).toBe(loaded.base_mtime);
  });

  it("splits the served annotation list into the canvas buckets", async () => {
    const { loaded } = LABEL_ANSWERS;
    fetchAnswers(200, loaded);
    const labels = await api.annotate.load(loaded.image_path);
    const [box] = loaded.annotations;
    expect(box.bbox).not.toBeNull();
    expect([
      labels.boxes[0].x1,
      labels.boxes[0].y1,
      labels.boxes[0].x2,
      labels.boxes[0].y2,
    ]).toEqual(box.bbox);
    expect(
      [labels.boxes, labels.polygons, labels.points, labels.imageAnnotations].map((b) => b.length),
    ).toEqual([1, 1, 1, 1]);
  });
});

describe("query string assembly", () => {
  it("carries each named parameter, a bucket name with its separator encoded", async () => {
    fetchAnswers(200, LABEL_ANSWERS.proposals);
    await api.annotate.proposals("img.jpg", "m/2026-01-01");
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe(
      "/api/annotate/proposals?image_path=img.jpg&bucket=m%2F2026-01-01",
    );
  });

  it("keeps a parameter whose value is zero", async () => {
    fetchAnswers(200, BAND_ANSWERS.four_spectral);
    await api.images.bands("mosaic.tif");
    expect(vi.mocked(fetch).mock.calls[0][0]).toBe("/api/images/bands?path=mosaic.tif");
    expect(api.images.url("mosaic.tif", display(), { region: [0, 4067, 4067, 8134] })).toBe(
      `/api/images?path=mosaic.tif&region=0&region=4067&region=4067&region=8134&display_pixels=${display()}&v=${RENDER_CACHE_VERSION}`,
    );
  });
});

describe("request defaults", () => {
  it("sends the JSON content type on a request that names no headers of its own", async () => {
    fetchAnswers(200, null);
    await api.dataset.nav(7);
    const init = vi.mocked(fetch).mock.calls[0][1];
    expect(init?.headers).toEqual({ "Content-Type": "application/json" });
    expect(init?.method).toBe("POST");
    expect(init?.body).toBe(JSON.stringify({ current_image_index: 7 }));
  });
});

function parseQuery(url: string): URLSearchParams {
  return new URLSearchParams(url.split("?")[1] ?? "");
}

describe("images.url", () => {
  it("omits bands/stretch when not given, so a plain RGB request is unaffected", () => {
    const params = parseQuery(api.images.url("C:/data/images/2026-01-01/img1.jpg", display()));
    expect(params.get("path")).toBe("C:/data/images/2026-01-01/img1.jpg");
    expect(params.has("bands")).toBe(false);
    expect(params.has("stretch")).toBe(false);
  });

  it("names this display's pixel count and no size or encoding, whole or region", () => {
    vi.stubGlobal("devicePixelRatio", 2);
    const pixels = display();
    expect(pixels).toBe(3840 * 2160);
    for (const opts of [{}, { region: [0, 0, 512, 512] as const }]) {
      const params = parseQuery(api.images.url("C:/data/images/2026-01-01/img1.jpg", pixels, opts));
      expect(params.get("display_pixels")).toBe(String(pixels));
      for (const key of ["max_width", "width", "quality"]) expect(params.has(key)).toBe(false);
    }
  });

  it("carries bands/stretch through to the query string when given", () => {
    const params = parseQuery(
      api.images.url("C:/data/images/2026-01-01/img1.bandgroup", display(), {
        bands: ["Red", "Green", "Blue"],
        stretch: "minmax",
      }),
    );
    expect(params.getAll("bands")).toEqual(["Red", "Green", "Blue"]);
    expect(params.get("stretch")).toBe("minmax");
  });
});

describe("images.url region params", () => {
  it("carries the native-pixel rect corners through to the query string", () => {
    const params = parseQuery(
      api.images.url("C:/data/images/2026-01-01/mosaic.tif", display(), {
        region: [0, 4067, 4067, 8134],
      }),
    );
    expect(params.getAll("region")).toEqual(["0", "4067", "4067", "8134"]);
  });

  it("omits the region when none is requested", () => {
    const params = parseQuery(api.images.url("C:/data/images/2026-01-01/img1.jpg", display()));
    expect(params.has("region")).toBe(false);
  });
});

describe("images.bands", () => {
  it("hits GET /api/images/bands with the path and returns the reported contract", async () => {
    const produced = BAND_ANSWERS.four_spectral;
    fetchAnswers(200, produced);
    const res = await api.images.bands("C:/data/images/2026-01-01/DJI_0001.bandgroup");
    expect(res).toEqual(produced);
    const fetchMock = vi.mocked(fetch);
    expect(fetchMock.mock.calls[0][0]).toContain("/api/images/bands?path=");
  });
});
