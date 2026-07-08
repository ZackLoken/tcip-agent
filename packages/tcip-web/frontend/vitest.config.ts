import { defineConfig, mergeConfig } from "vitest/config";

import viteConfig from "./vite.config";

// Reuse the app's Vite config (notably the "@" alias) for tests, adding a jsdom
// environment and jest-dom matchers. Tests live next to source as *.test.ts(x).
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: "jsdom",
      setupFiles: ["./src/test/setup.ts"],
      include: ["src/**/*.test.{ts,tsx}"],
      css: false,
      restoreMocks: true,
    },
  }),
);
