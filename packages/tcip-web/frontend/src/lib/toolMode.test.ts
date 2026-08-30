import { describe, expect, it } from "vitest";

import { nextMode } from "@/lib/toolMode";

describe("nextMode", () => {
  it("cycles box -> polygon -> point -> box", () => {
    expect(nextMode("box")).toBe("polygon");
    expect(nextMode("polygon")).toBe("point");
    expect(nextMode("point")).toBe("box");
  });
});
