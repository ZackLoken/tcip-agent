import { afterEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";

function stubFetch(status: number, body: unknown = {}) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
      text: async () => JSON.stringify(body),
    } as Response),
  );
}

describe("annotate.save lost-update handling", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns ok + fresh mtimes on a 200", async () => {
    stubFetch(200, { base_mtimes: { detect: 123, segment: null } });
    const res = await api.annotate.save({ image_path: "x", boxes: [], polygons: [] });
    expect(res.status).toBe("ok");
    if (res.status === "ok") expect(res.base_mtimes.detect).toBe(123);
  });

  it("returns a conflict (not a thrown error) on a 409", async () => {
    stubFetch(409, { error: "label file changed since it was loaded" });
    const res = await api.annotate.save({
      image_path: "x",
      boxes: [],
      polygons: [],
      base_mtimes: { detect: 1, segment: null },
    });
    expect(res.status).toBe("conflict");
  });
});
