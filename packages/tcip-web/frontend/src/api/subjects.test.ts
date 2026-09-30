import { afterEach, describe, expect, it, vi } from "vitest";

import { subjectsApi, derivedSubjectColor, setSubjectColorRegistry } from "@/api/subjects";
import { SUBJECT_COLORS, subjectColor } from "@/api/subjects";

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

describe("subjectColor collision-free registry slots", () => {
  afterEach(() => {
    setSubjectColorRegistry([]); // don't leak one test's registry into the next
  });

  it("confirms the premise: two names can share a bare hash slot before any registry loads", () => {
    expect(derivedSubjectColor("fruit")).toBe(derivedSubjectColor("leaf"));
  });

  it("gives two colliding names in one registry two different colors", () => {
    setSubjectColorRegistry(["fruit", "leaf"]);
    expect(subjectColor("fruit")).not.toBe(subjectColor("leaf"));
    expect(SUBJECT_COLORS).toContain(subjectColor("fruit"));
    expect(SUBJECT_COLORS).toContain(subjectColor("leaf"));
  });

  it("leaves a lone subject on its own hash color", () => {
    setSubjectColorRegistry(["solo"]);
    expect(subjectColor("solo")).toBe(derivedSubjectColor("solo"));
  });
});

describe("image-status writes carry the app-set identity", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("names the person in the single-image body, so the backend stamps them and not itself", async () => {
    stubFetch();
    await subjectsApi.setImageStatus(
      "img1.jpg",
      "negative",
      "subject_a",
      "2026-01-01",
      "C:/data",
      "breeder",
    );

    expect(sentBody().user).toBe("breeder");
    expect(sentBody().status).toBe("negative");
  });

  it("names the person in the bulk body, which writes the same store one call wider", async () => {
    stubFetch();
    await subjectsApi.setImageStatusBulk(
      { "img1.jpg": "partial" },
      "subject_a",
      "2026-01-01",
      "C:/data",
      "breeder",
    );

    expect(sentBody().user).toBe("breeder");
    expect(sentBody().statuses).toEqual({ "img1.jpg": "partial" });
  });

  it("leaves the field out when no name is set, which is what the backend fallback answers", async () => {
    stubFetch();
    await subjectsApi.setImageStatus(
      "img1.jpg",
      "complete",
      "subject_a",
      "2026-01-01",
      "C:/data",
      undefined,
    );

    expect("user" in sentBody()).toBe(false);
  });
});

describe("loadImageStatus admits only the whole declared response", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns a response of known statuses and string stale names", async () => {
    const body = { statuses: { "a.jpg": "complete" }, stale_definition: ["a.jpg"] };
    stubFetch(body);
    await expect(subjectsApi.loadImageStatus("subject_a", null, "C:/data")).resolves.toEqual(body);
  });

  it.each([
    [{ statuses: [], stale_definition: [] }],
    [{ statuses: { "a.jpg": 42 }, stale_definition: [] }],
    [{ statuses: { "a.jpg": "done" }, stale_definition: [] }],
    [{ statuses: {}, stale_definition: [7] }],
    [{ statuses: {} }],
  ])("refuses %j, naming it", async (body) => {
    stubFetch(body);
    await expect(subjectsApi.loadImageStatus("subject_a", null, "C:/data")).rejects.toThrow(
      /another shape/,
    );
  });
});
