// Extends Vitest's `expect` with jest-dom matchers (toBeInTheDocument, etc.) and
// registers automatic cleanup after each test.
import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver: recharts' ResponsiveContainer (used by ResultsTab's chart)
// needs one to mount at all. A minimal no-op stub is enough for tests, which never assert on
// actual measured layout.
if (typeof globalThis.ResizeObserver === "undefined") {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

// jsdom reports a 0x0 screen, which makes every display-pixel count the client sends zero, a
// request the server refuses. A 1920x1080 screen stands in so producer-built URLs are valid.
Object.defineProperty(window.screen, "width", { value: 1920, configurable: true });
Object.defineProperty(window.screen, "height", { value: 1080, configurable: true });

// Node 22+ defines experimental localStorage/sessionStorage globals that are undefined without
// --localstorage-file and shadow jsdom's; back the missing ones with an in-memory stand-in.
function memoryStorage(): Storage {
  const store = new Map<string, string>();
  return {
    get length() {
      return store.size;
    },
    clear: () => store.clear(),
    getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
    key: (i: number) => [...store.keys()][i] ?? null,
    removeItem: (k: string) => {
      store.delete(k);
    },
    setItem: (k: string, v: string) => {
      store.set(k, String(v));
    },
  };
}
if (!globalThis.localStorage) {
  Object.defineProperty(globalThis, "localStorage", {
    value: memoryStorage(),
    configurable: true,
    writable: true,
  });
}
if (!globalThis.sessionStorage) {
  Object.defineProperty(globalThis, "sessionStorage", {
    value: memoryStorage(),
    configurable: true,
    writable: true,
  });
}
