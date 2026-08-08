import { afterEach, describe, expect, it, vi } from "vitest";

// The rail's open state is read at store-module init, so each case re-imports the
// store against a controlled localStorage.
describe("terminal rail open policy", () => {
  afterEach(() => {
    localStorage.removeItem("tcip.terminal_open");
    vi.resetModules();
  });

  it("starts closed even when a previous session left it open", async () => {
    localStorage.setItem("tcip.terminal_open", "1");
    vi.resetModules();
    const { useStore } = await import("@/store");
    expect(useStore.getState().terminalOpen).toBe(false);
  });

  it("toggling the rail does not persist the open state across sessions", async () => {
    vi.resetModules();
    const { useStore } = await import("@/store");
    useStore.getState().setTerminalOpen(true);
    expect(useStore.getState().terminalOpen).toBe(true);
    expect(localStorage.getItem("tcip.terminal_open")).toBeNull();
  });
});
