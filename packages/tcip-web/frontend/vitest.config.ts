import { join, posix, relative } from "node:path";

import { normalizePath } from "vite";
import { defineConfig, mergeConfig } from "vitest/config";

import { nonTestSources } from "./scripts/routeReads.mjs";
import tsconfig from "./tsconfig.json" with { type: "json" };
import viteConfig from "./vite.config";

// The app's source directory, relative to this one, as tsconfig.json includes it.
const [SRC] = tsconfig.include;
const SOURCE_DIRECTORY = join(import.meta.dirname, SRC);

// Reuse the app's Vite config (notably its tsconfig paths) for tests, adding a jsdom
// environment and jest-dom matchers. Tests live next to source as *.test.ts(x).
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: "jsdom",
      setupFiles: [join(SRC, "test", "setup.ts")],
      include: [posix.join(SRC, "**", "*.test.{ts,tsx}"), "scripts/**/*.test.mjs"],
      css: false,
      restoreMocks: true,
      unstubGlobals: true,
      provide: { sourceDirectory: SOURCE_DIRECTORY },
      coverage: {
        provider: "v8",
        include: nonTestSources(SOURCE_DIRECTORY).map((file) =>
          normalizePath(relative(import.meta.dirname, file)),
        ),
        exclude: ["src/api/*.generated.ts"],
        // Lowest of four AST-remapped V8 coverage runs under vitest 5 on this tree, rounded down:
        // statements 86.07, branches 79.6, functions 85.00, lines 87.94; runs vary by hundredths.
        thresholds: { statements: 86, branches: 79, functions: 85, lines: 87 },
      },
    },
  }),
);
