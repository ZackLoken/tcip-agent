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

describe("registry save", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("carries the version it was loaded at, so a stale save is refused", async () => {
    stubFetch({ status: "ok", n_subjects: 1, subjects_path: "s", version: "v2" });
    await subjectsApi.save({ bush: {} }, "C:/data", "v1", "jordan");
    expect(sentBody()).toEqual({
      subjects: { bush: {} },
      dataset_root: "C:/data",
      version: "v1",
      user: "jordan",
    });
  });
});
