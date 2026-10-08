import { afterEach, describe, expect, it, vi } from "vitest";

import { sessionsApi } from "@/api/sessions";

const REF = { project_id: "a1b2c3d4e5f6", started: "2026-10-07T14:00:00+00:00" };

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("sessionsApi.endAsPageLeaves", () => {
  it("sends the session reference as one beacon when the browser accepts it", async () => {
    const beacon = vi.fn().mockReturnValue(true);
    vi.stubGlobal("navigator", { sendBeacon: beacon });
    const send = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("unused"));

    sessionsApi.endAsPageLeaves(REF);

    expect(beacon).toHaveBeenCalledTimes(1);
    const [url, body] = beacon.mock.calls[0] as [string, Blob];
    expect(url).toContain("/api/sessions/end");
    expect(body.type).toBe("application/json");
    const text = await new Promise<string>((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.readAsText(body);
    });
    expect(text).toBe(JSON.stringify(REF));
    expect(send).not.toHaveBeenCalled();
  });

  it("falls back to a keepalive fetch when the browser refuses the beacon", () => {
    vi.stubGlobal("navigator", { sendBeacon: vi.fn().mockReturnValue(false) });
    const send = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));

    sessionsApi.endAsPageLeaves(REF);

    expect(send).toHaveBeenCalledTimes(1);
    expect(send).toHaveBeenCalledWith(expect.stringContaining("/api/sessions/end"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(REF),
      keepalive: true,
    });
  });

  it("falls back to a keepalive fetch when the beacon throws or does not exist", () => {
    const send = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));
    vi.stubGlobal("navigator", {
      sendBeacon: vi.fn().mockImplementation(() => {
        throw new TypeError("beacon refused");
      }),
    });
    sessionsApi.endAsPageLeaves(REF);
    vi.stubGlobal("navigator", {});
    sessionsApi.endAsPageLeaves(REF);

    expect(send).toHaveBeenCalledTimes(2);
  });
});
