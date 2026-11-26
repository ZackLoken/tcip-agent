import { describe, expect, it, vi } from "vitest";

import { takingOnce } from "@/lib/takenEvents";
import { PANEL_EVENTS, PANEL_EVENTS_AT_CONNECT } from "@/test/panelEvents";

describe("takingOnce", () => {
  it("hands an event its handler threw on to the next delivery, and drops it once taken", () => {
    const handler = vi.fn().mockImplementationOnce(() => {
      throw new Error("not handled");
    });
    const take = takingOnce(handler);

    expect(() => take(PANEL_EVENTS.canvas_state_request)).toThrow("not handled");
    take(PANEL_EVENTS_AT_CONNECT.canvas_state_request);
    take(PANEL_EVENTS_AT_CONNECT.canvas_state_request);

    expect(handler).toHaveBeenCalledTimes(2);
  });
});
