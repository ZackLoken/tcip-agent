// Node's file and path calls for the tests, typed here since the frontend's tsconfig has no @types/node.
import { inject } from "vitest";

declare module "vitest" {
  export interface ProvidedContext {
    sourceDirectory: string;
  }
}

// @ts-expect-error TS2307: no @types/node in the frontend tsconfig
const fs = (await import("node:fs")) as {
  readFileSync: (path: string, encoding: "utf8") => string;
};
// @ts-expect-error TS2307: no @types/node in the frontend tsconfig
const path = (await import("node:path")) as { resolve: (...paths: string[]) => string };

export const { readFileSync } = fs;
export const { resolve } = path;

/** The source directory `tsconfig.json` includes, as `vitest.config.ts` resolves and provides it. */
export const SOURCE_DIRECTORY = inject("sourceDirectory");
