import { afterEach, describe, expect, it, vi } from "vitest";

import { asJson, getJson, postJson } from "@/api/http";

function res(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: "",
    json: async () => body,
  } as Response;
}

describe("http helpers", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("asJson returns the parsed body on a 2xx", async () => {
    expect(await asJson<{ a: number }>(res(200, { a: 1 }))).toEqual({ a: 1 });
  });

  it("asJson throws the backend detail on a non-2xx", async () => {
    await expect(asJson(res(404, { detail: "nope" }))).rejects.toThrow("nope");
  });

  it("getJson resolves the body via fetch", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(res(200, { ok: true })));
    expect(await getJson<{ ok: boolean }>("/x")).toEqual({ ok: true });
  });

  it("postJson throws (does not swallow) on a non-2xx", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(res(500, { detail: "boom" })));
    await expect(postJson("/x", {})).rejects.toThrow("boom");
  });
});
