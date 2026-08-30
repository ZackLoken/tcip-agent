import { describe, expect, it } from "vitest";

import * as store from "@/store";

/** AppState, CanvasState, AgentActivity, Toast and Banner are interfaces: TypeScript erases them
 *  at compile time, so they carry no runtime binding here even though they are exported types.
 *  This only proves the module's two real (value) exports are exactly useStore and
 *  subjectColor, not that every type export is covered. */
describe("store module export surface", () => {
  it("exports exactly useStore and subjectColor at runtime", () => {
    expect(Object.keys(store).sort()).toEqual(["subjectColor", "useStore"]);
  });
});
