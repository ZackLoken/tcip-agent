import { afterEach, describe, expect, it, vi } from "vitest";

import { classesApi } from "@/api/classes";

function stubFetch(body: unknown = { status: "ok" }) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => body,
      text: async () => JSON.stringify(body),
    } as Response),
  );
}

function sentBody(): Record<string, unknown> {
  return JSON.parse(String(vi.mocked(fetch).mock.calls[0][1]?.body));
}

describe("image-status writes carry the app-set identity", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("names the person in the single-image body, so the backend stamps them and not itself", async () => {
    stubFetch();
    await classesApi.setImageStatus(
      "C:/proj",
      "img1.jpg",
      "negative",
      "subject_a",
      "2026-01-01",
      "C:/data",
      "C:/data/annotations/2026-01-01",
      "breeder",
    );

    expect(sentBody().user).toBe("breeder");
    expect(sentBody().status).toBe("negative");
  });

  it("names the person in the bulk body, which writes the same store one call wider", async () => {
    stubFetch();
    await classesApi.setImageStatusBulk(
      "C:/proj",
      { "img1.jpg": "partial" },
      "subject_a",
      "2026-01-01",
      "C:/data",
      "C:/data/annotations/2026-01-01",
      "breeder",
    );

    expect(sentBody().user).toBe("breeder");
    expect(sentBody().statuses).toEqual({ "img1.jpg": "partial" });
  });

  it("leaves the field out when no name is set, which is what the backend fallback answers", async () => {
    stubFetch();
    await classesApi.setImageStatus(
      "C:/proj",
      "img1.jpg",
      "complete",
      "subject_a",
      "2026-01-01",
      "C:/data",
      "C:/data/annotations/2026-01-01",
      undefined,
    );

    expect("user" in sentBody()).toBe(false);
  });
});
