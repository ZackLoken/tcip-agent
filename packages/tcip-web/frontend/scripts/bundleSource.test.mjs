import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { build } from "vite";
import { afterEach, describe, expect, it } from "vitest";

import { buildRecord, sourceHash } from "./bundleSource.mjs";

const roots = [];

afterEach(() => {
  roots.splice(0).forEach((root) => rmSync(root, { recursive: true }));
});

/** A scratch frontend root that registers the bundle record plugin and builds into `out`. */
function scratchRoot() {
  const root = mkdtempSync(join(tmpdir(), "bundle-source-"));
  roots.push(root);
  mkdirSync(join(root, "src"));
  mkdirSync(join(root, "public"));
  writeFileSync(join(root, "tsconfig.json"), JSON.stringify({ include: ["src"] }));
  writeFileSync(join(root, "public", "logo.txt"), "logo");
  writeFileSync(join(root, "src", "main.ts"), "console.log('orchard');\n");
  writeFileSync(
    join(root, "index.html"),
    '<!doctype html><script type="module" src="/src/main.ts"></script>\n',
  );
  writeFileSync(
    join(root, "vite.config.ts"),
    `import { bundleRecord } from ${JSON.stringify(
      pathToFileURL(join(import.meta.dirname, "bundleSource.mjs")).href,
    )};\n\nexport default { plugins: [bundleRecord()], build: { outDir: "out" } };\n`,
  );
  return root;
}

describe("the bundle's build record", () => {
  it("is the record the source's own identity gives, written beside the built bundle", async () => {
    const root = scratchRoot();
    await build({ root, logLevel: "silent" });
    expect(JSON.parse(readFileSync(join(root, "out", "build.json"), "utf8"))).toEqual(
      buildRecord(root),
    );
  });

  it("is still written when the build empties an output directory that held an older one", async () => {
    const root = scratchRoot();
    mkdirSync(join(root, "out"));
    writeFileSync(join(root, "out", "build.json"), '{"source":"older"}');
    await build({ root, logLevel: "silent" });
    expect(JSON.parse(readFileSync(join(root, "out", "build.json"), "utf8")).source).toBe(
      sourceHash(root),
    );
  });
});

describe("the source's identity", () => {
  it("moves with a file under the source directory, under public, and at the root", () => {
    const root = scratchRoot();
    const before = sourceHash(root);
    const seen = new Set([before]);
    for (const file of [join("src", "main.ts"), join("public", "logo.txt"), "index.html"]) {
      writeFileSync(join(root, file), `${file} changed\n`);
      seen.add(sourceHash(root));
    }
    expect(seen.size).toBe(4);
  });

  it("does not move with a file the bundle is not built from", () => {
    const root = scratchRoot();
    const before = sourceHash(root);
    writeFileSync(join(root, "README.md"), "notes\n");
    expect(sourceHash(root)).toBe(before);
  });

  it("does not move with a test file or a file under the source directory's test folder", () => {
    const root = scratchRoot();
    const before = sourceHash(root);
    mkdirSync(join(root, "src", "test"));
    writeFileSync(join(root, "src", "main.test.ts"), "it('runs', () => {});\n");
    writeFileSync(join(root, "src", "test", "answers.json"), "{}\n");
    expect(sourceHash(root)).toBe(before);
  });
});
