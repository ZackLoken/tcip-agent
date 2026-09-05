import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act, renderHook } from "@testing-library/react";

import {
  loadSubjectColorOverrides,
  resetSubjectColorOverride,
  setSubjectColorOverride,
  subjectColorOverride,
  useSubjectColors,
} from "@/lib/subjectColors";

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  localStorage.clear();
});

describe("subjectColorOverride", () => {
  it("is null for a subject nobody has recoloured", () => {
    expect(subjectColorOverride("bush")).toBeNull();
  });

  it("returns the hex a caller set for that subject only", () => {
    setSubjectColorOverride("bush", "#123456");
    expect(subjectColorOverride("bush")).toBe("#123456");
    expect(subjectColorOverride("leaf")).toBeNull();
  });

  it("persists the whole map under one localStorage key", () => {
    setSubjectColorOverride("bush", "#123456");
    setSubjectColorOverride("leaf", "#abcdef");
    expect(loadSubjectColorOverrides()).toEqual({ bush: "#123456", leaf: "#abcdef" });
  });
});

describe("resetSubjectColorOverride", () => {
  it("removes only the named subject's override", () => {
    setSubjectColorOverride("bush", "#123456");
    setSubjectColorOverride("leaf", "#abcdef");
    resetSubjectColorOverride("bush");
    expect(subjectColorOverride("bush")).toBeNull();
    expect(subjectColorOverride("leaf")).toBe("#abcdef");
  });
});

describe("useSubjectColors", () => {
  it("bumps its tick when an override is set or reset, so a memo depending on it recomputes", () => {
    const { result } = renderHook(() => useSubjectColors());
    const first = result.current;

    act(() => setSubjectColorOverride("bush", "#123456"));
    expect(result.current).not.toBe(first);

    const second = result.current;
    act(() => resetSubjectColorOverride("bush"));
    expect(result.current).not.toBe(second);
  });
});
