import { describe, expect, it } from "vitest";

import { cleanPath } from "@/lib/paths";

describe("cleanPath", () => {
  it("trims surrounding whitespace", () => {
    expect(cleanPath("  C:\\data\\Valley_Farm  ")).toBe("C:\\data\\Valley_Farm");
  });
  it('strips wrapping double quotes (Windows "Copy as path")', () => {
    expect(cleanPath('"C:\\data\\Valley_Farm"')).toBe("C:\\data\\Valley_Farm");
  });
  it("strips wrapping single quotes and surrounding spaces together", () => {
    expect(cleanPath(" 'C:/data/Valley_Farm' ")).toBe("C:/data/Valley_Farm");
  });
  it("leaves a clean path untouched", () => {
    expect(cleanPath("C:/data/Valley_Farm")).toBe("C:/data/Valley_Farm");
  });
});
