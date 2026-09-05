import { afterEach, describe, expect, it, vi } from "vitest";

import { classesApi, derivedSubjectColor, setSubjectColorRegistry } from "@/api/classes";
import { SUBJECT_COLORS, subjectColor } from "@/api/classes";

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

  it("gives two colliding names in one registry two different colours", () => {
    setSubjectColorRegistry(["fruit", "leaf"]);
    expect(subjectColor("fruit")).not.toBe(subjectColor("leaf"));
    expect(SUBJECT_COLORS).toContain(subjectColor("fruit"));
    expect(SUBJECT_COLORS).toContain(subjectColor("leaf"));
  });

  it("leaves a lone subject on its own hash colour", () => {
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
