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
      coverage: {
        provider: "v8",
        include: ["src/**/*.{ts,tsx}"],
        exclude: ["src/**/*.test.{ts,tsx}", "src/test/**", "src/api/*.generated.ts"],
        // The floor is the coverage measured when the gate was written: statements 91.65,
        // branches 88.66, functions 80.09, lines 91.65 percent, each rounded down.
        thresholds: { statements: 91, branches: 88, functions: 80, lines: 91 },
      },
    },
  }),
);
