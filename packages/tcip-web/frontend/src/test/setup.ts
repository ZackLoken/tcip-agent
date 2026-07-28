// Extends Vitest's `expect` with jest-dom matchers (toBeInTheDocument, etc.) and
// registers automatic cleanup after each test.
import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver — recharts' ResponsiveContainer (used by ResultsTab's chart)
// needs one to mount at all. A minimal no-op stub is enough for tests, which never assert on
// actual measured layout.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}
