import { describe, expect, it } from "vitest";

import { isAnnotatorName } from "@/store/slices/user";
import names from "@/test/personNames.json";

describe("isAnnotatorName", () => {
  it("admits every name the backend's door admits, from the one list both are tested on", () => {
    for (const [stated] of names.admitted) {
      expect(isAnnotatorName(stated)).toBe(true);
    }
  });

  it("refuses every name the door refuses", () => {
    for (const stated of names.refused) {
      expect(isAnnotatorName(stated)).toBe(false);
    }
  });
});
