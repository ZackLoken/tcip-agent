import { describe, expect, it } from "vitest";

import { attachCtrlWheelGuard } from "@/lib/ctrlWheelGuard";

describe("attachCtrlWheelGuard", () => {
  it("preventDefaults ctrl+wheel at the attached root and passes plain wheel through", () => {
    const el = document.createElement("div");
    document.body.appendChild(el);
    const detach = attachCtrlWheelGuard(el);

    const ctrlWheel = new WheelEvent("wheel", { ctrlKey: true, cancelable: true, bubbles: true });
    el.dispatchEvent(ctrlWheel);
    expect(ctrlWheel.defaultPrevented).toBe(true);

    const plainWheel = new WheelEvent("wheel", { cancelable: true, bubbles: true });
    el.dispatchEvent(plainWheel);
    expect(plainWheel.defaultPrevented).toBe(false);

    detach();
    const afterDetach = new WheelEvent("wheel", { ctrlKey: true, cancelable: true, bubbles: true });
    el.dispatchEvent(afterDetach);
    expect(afterDetach.defaultPrevented).toBe(false);
    el.remove();
  });
});
