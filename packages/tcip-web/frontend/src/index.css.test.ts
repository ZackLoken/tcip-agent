// Compiles index.css through PostCSS + Tailwind (same pipeline as the real
// build) and asserts the keyboard focus-visible ring rules exist on the shared
// component classes. Guards the accessibility fix: a typo'd token or dropped
// @apply utility fails here instead of silently shipping invisible focus.
import postcss, { type Declaration, type Rule } from "postcss";
import tailwindcss from "tailwindcss";
import { beforeAll, describe, expect, it } from "vitest";

import baseConfig from "../tailwind.config";

/** Merge declarations from every rule matching `selector` exactly. */
function declsFor(root: postcss.Root, selector: string): Map<string, string> {
  const decls = new Map<string, string>();
  root.walkRules((rule: Rule) => {
    if (rule.selector !== selector) return;
    rule.walkDecls((decl: Declaration) => {
      decls.set(decl.prop, decl.value);
    });
  });
  return decls;
}

describe("index.css component layer", () => {
  let root: postcss.Root;

  beforeAll(async () => {
    // Read from disk: vitest's `css: false` stubs CSS imports (even ?raw) to
    // empty strings. @types/node isn't installed in this package, so import
    // node:fs dynamically — vitest runs in Node, so it resolves at runtime.
    // @ts-expect-error TS2307: no @types/node in the frontend tsconfig
    const { readFileSync } = await import("node:fs");
    // Vitest runs with cwd = frontend root (vitest.config.ts lives there).
    const rawCss: string = readFileSync("src/index.css", "utf8");

    // Raw content stub instead of file globs so the test emits the component
    // classes without scanning the source tree.
    const result = await postcss([
      tailwindcss({
        ...baseConfig,
        content: [
          {
            raw: '<div class="tcip-btn tcip-btn-primary tcip-btn-danger tcip-input tcip-select"></div>',
            extension: "html",
          },
        ],
      }),
    ]).process(rawCss, { from: "src/index.css" });
    root = result.root;
  });

  // .tcip-btn-primary/.tcip-btn-danger inherit via `@apply tcip-btn`, and
  // .tcip-select via `@apply tcip-input` — assert all so the chain can't break.
  it.each([".tcip-btn", ".tcip-btn-primary", ".tcip-btn-danger", ".tcip-input", ".tcip-select"])(
    "%s has a visible focus-visible ring",
    (base) => {
      const decls = declsFor(root, `${base}:focus-visible`);
      // ring-2 → box-shadow composed from the Tailwind ring variables
      expect(decls.get("box-shadow")).toContain("--tw-ring-offset-shadow");
      // ring-tcip-accent/70 → SI green at 70% opacity
      expect(decls.get("--tw-ring-color")).toBe("rgb(80 119 84 / 0.7)");
      // ring-offset-1 + ring-offset-tcip-bg → 1px gap in the page background
      expect(decls.get("--tw-ring-offset-width")).toBe("1px");
      expect(decls.get("--tw-ring-offset-color")).toBe("#1E1E1E");
    },
  );

  it("buttons suppress the UA outline only under focus-visible", () => {
    const decls = declsFor(root, ".tcip-btn:focus-visible");
    expect(decls.get("outline")).toBe("2px solid transparent");
    // No plain :focus outline suppression on buttons — mouse clicks keep
    // default behavior and only keyboard focus swaps to the ring.
    expect(declsFor(root, ".tcip-btn:focus").size).toBe(0);
  });

  it("inputs keep the accent border swap on focus", () => {
    const decls = declsFor(root, ".tcip-input:focus");
    expect(decls.get("border-color")).toContain("80 119 84");
  });
});
