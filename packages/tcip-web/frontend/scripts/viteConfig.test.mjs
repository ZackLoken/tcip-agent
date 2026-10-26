import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import { servedConfig } from "./viteConfig.mjs";

const roots = [];

afterEach(() => {
  roots.splice(0).forEach((root) => rmSync(root, { recursive: true }));
});

describe("the served Vite configuration", () => {
  it("carries the proxy a TypeScript configuration builds from a module it imports", async () => {
    const root = mkdtempSync(join(tmpdir(), "vite-config-"));
    roots.push(root);
    writeFileSync(
      join(root, "entries.ts"),
      'export const ENTRIES: { path: string; ws: boolean }[] = [{ path: "/orchard", ws: true }];\n',
    );
    writeFileSync(
      join(root, "vite.config.ts"),
      'import { ENTRIES } from "./entries";\n\n' +
        "export default {\n" +
        "  server: {\n" +
        "    proxy: Object.fromEntries(\n" +
        '      ENTRIES.map(({ path, ws }) => [path, { target: "http://127.0.0.1:1", ws }]),\n' +
        "    ),\n" +
        "  },\n" +
        "};\n",
    );
    expect((await servedConfig(root)).server.proxy).toEqual({
      "/orchard": { target: "http://127.0.0.1:1", ws: true },
    });
  });
});
