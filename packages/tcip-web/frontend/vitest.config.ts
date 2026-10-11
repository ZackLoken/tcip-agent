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
        // Lowest of four AST-remapped V8 coverage runs under vitest 5 on this tree, rounded down:
        // statements 86.07, branches 79.6, functions 85.00, lines 87.94; runs vary by hundredths.
        thresholds: { statements: 86, branches: 79, functions: 85, lines: 87 },
      },
    },
  }),
);
