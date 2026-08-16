import { afterEach, describe, expect, it, vi } from "vitest";

import { asJson, decodeRefusal, getJson, postJson, StructuredRefusalError } from "@/api/http";

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

  it("asJson keeps an object detail parsed instead of stringifying it", async () => {
    const detail = {
      kind: "operationalization",
      state: 2,
      trait: "catkin_50per_date",
      delivery_kind: "state_crossing_dates",
      message: "stated but not confirmed by the breeder",
    };
    const thrown = await asJson(res(400, { detail })).catch((e: unknown) => e);
    expect(thrown).toBeInstanceOf(StructuredRefusalError);
    const refusal = thrown as StructuredRefusalError;
    expect(refusal.detail).toEqual(detail);
    expect(refusal.status).toBe(400);
    expect(refusal.message).toBe("stated but not confirmed by the breeder");
    expect(refusal.message).not.toContain("[object Object]");
  });

  it("decodeRefusal falls back to the caller's own text when the body carries no detail", async () => {
    const thrown = await decodeRefusal(res(500, {}), "export_csv failed: 500");
    expect(thrown).not.toBeInstanceOf(StructuredRefusalError);
    expect(thrown.message).toBe("export_csv failed: 500");
  });
});
