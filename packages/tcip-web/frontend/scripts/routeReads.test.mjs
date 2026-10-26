import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";

import { afterEach, describe, expect, it } from "vitest";

import {
  backendPathLiterals,
  declaredShapes,
  nonTestSources,
  routeReads,
  stringConstants,
  stringUnions,
} from "./routeReads.mjs";

const trees = [];

/** Write each text of `texts` at its path under a fresh scratch source directory; return it. */
function tree(texts) {
  const src = mkdtempSync(join(tmpdir(), "route-reads-"));
  trees.push(src);
  for (const [relative, text] of Object.entries(texts)) {
    mkdirSync(dirname(join(src, relative)), { recursive: true });
    writeFileSync(join(src, relative), text);
  }
  return src;
}

afterEach(() => {
  trees.splice(0).forEach((src) => rmSync(src, { recursive: true }));
});

describe("the frontend's non-test sources", () => {
  it("leave out a test file and everything under the test folder, and nothing else", () => {
    const src = tree({
      "api/reads.ts": "",
      "contest/reads.tsx": "",
      "api/reads.test.ts": "",
      "test/reads.ts": "",
      "test/nested/reads.ts": "",
      "api/notes.md": "",
    });
    expect(nonTestSources(src)).toEqual([
      join(src, "api", "reads.ts"),
      join(src, "contest", "reads.tsx"),
    ]);
  });
});

describe("the frontend's route reads", () => {
  it("are property accesses on ROUTES, never a comment or a string naming one", () => {
    const src = tree({
      "api/reads.ts": "// ROUTES.getHealth\nconst s = 'ROUTES.getRoot';\nROUTES . getState;\n",
    });
    expect(routeReads(nonTestSources(src))).toEqual(["getState"]);
  });
});

describe("the frontend's backend path literals", () => {
  it("are found in a string and a template literal", () => {
    const src = tree({ "api/paths.ts": 'fetch("/api/orchard");\nfetch(`/ws/orchard/${id}`);\n' });
    const file = join(src, "api", "paths.ts");
    expect(backendPathLiterals(nonTestSources(src))).toEqual([
      { file, text: "/api/orchard" },
      { file, text: "/ws/orchard/" },
    ]);
  });
});

describe("the frontend's declarations", () => {
  it("are object shapes with every field the checker resolves, never a commented one", () => {
    const src = tree({
      "tsconfig.json": '{ "compilerOptions": { "noLib": true } }',
      "api/shapes.ts":
        "// interface Commented { z: string }\ninterface Base { a: string }\n" +
        "export interface Child extends Base { b?: number }\ntype Plain = { c: string };\n",
    });
    const file = join(src, "api", "shapes.ts");
    expect(declaredShapes(nonTestSources(src))).toEqual([
      { file, name: "Base", fields: ["a"] },
      { file, name: "Child", fields: ["a", "b"] },
      { file, name: "Plain", fields: ["c"] },
    ]);
  });

  it("are string literal unions an alias or a property declares, null left out", () => {
    const src = tree({
      "api/unions.ts":
        'type Phase = "live" | "ended";\n' +
        'interface Row { state: "open" | "shut" | null; count: 1 | 2; label: string }\n' +
        'const text = \'type Fake = "a" | "b";\';\n',
    });
    const file = join(src, "api", "unions.ts");
    expect(stringUnions(nonTestSources(src))).toEqual([
      { file, alias: "Phase", members: ["ended", "live"] },
      { file, alias: null, members: ["open", "shut"] },
    ]);
  });

  it("are exported string constants, never an unexported, a let or a number", () => {
    const src = tree({
      "api/constants.ts":
        'export const RULE = "^a\\\\d$";\nconst HIDDEN = "b";\nexport let LOOSE = "c";\n' +
        "export const COUNT = 1;\n",
    });
    expect(stringConstants(nonTestSources(src))).toEqual([
      { file: join(src, "api", "constants.ts"), name: "RULE", value: "^a\\d$" },
    ]);
  });
});
