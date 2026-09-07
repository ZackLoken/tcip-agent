import { afterEach, describe, expect, it, vi } from "vitest";

import {
  asJson,
  committedOf,
  decodeRefusal,
  getJson,
  isAuditEntryNotWritten,
  postJson,
  StructuredRefusalError,
} from "@/api/http";

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

  it("a plain string detail still carries its status, so a caller can branch on it", async () => {
    const thrown = await asJson(res(404, { detail: "sweep not found: hpo-1" })).catch(
      (e: unknown) => e,
    );
    expect(thrown).toBeInstanceOf(StructuredRefusalError);
    const refusal = thrown as StructuredRefusalError;
    expect(refusal.status).toBe(404);
    expect(refusal.message).toBe("sweep not found: hpo-1");
  });

  it("isAuditEntryNotWritten reads the marker at any status, never just 409", async () => {
    const at409 = await asJson(
      res(409, { detail: { error: "audit_entry_not_written", message: "m", committed: null } }),
    ).catch((e: unknown) => e);
    const at500 = await asJson(
      res(500, { detail: { error: "audit_entry_not_written", message: "m", committed: null } }),
    ).catch((e: unknown) => e);
    expect(isAuditEntryNotWritten(at409)).toBe(true);
    expect(isAuditEntryNotWritten(at500)).toBe(true);
  });

  it("isAuditEntryNotWritten is false for an ordinary refusal, or a non-error value", () => {
    expect(isAuditEntryNotWritten(new StructuredRefusalError({ error: "other" }, 409, "m"))).toBe(
      false,
    );
    expect(isAuditEntryNotWritten(new Error("plain"))).toBe(false);
    expect(isAuditEntryNotWritten(null)).toBe(false);
  });

  it("committedOf returns the committed body only for the marker, null otherwise", async () => {
    const committed = { status: "ok", n: 1 };
    const gap = await asJson(
      res(409, { detail: { error: "audit_entry_not_written", message: "m", committed } }),
    ).catch((e: unknown) => e);
    expect(committedOf<typeof committed>(gap)).toEqual(committed);

    const nullCommitted = await asJson(
      res(409, { detail: { error: "audit_entry_not_written", message: "m", committed: null } }),
    ).catch((e: unknown) => e);
    expect(committedOf(nullCommitted)).toBeNull();

    const ordinary = await asJson(res(400, { detail: "bad request" })).catch((e: unknown) => e);
    expect(committedOf(ordinary)).toBeNull();
  });
});
