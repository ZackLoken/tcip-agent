// Compiles index.css through Tailwind and Lightning CSS (the real build's pipeline) and asserts the
// focus-visible ring rules on the shared component classes, so a typo'd token fails here.
import { compile, optimize } from "@tailwindcss/node";
import postcss, { type Declaration, type Rule } from "postcss";
import { beforeAll, describe, expect, it } from "vitest";

/** The enclosing at-rule of `rule` that conditions it (`@media` or `@supports`), if any. */
function conditionOf(rule: Rule): string | null {
  for (let node = rule.parent; node && node.type !== "root"; node = node.parent) {
    if (node.type === "atrule" && (node.name === "media" || node.name === "supports")) {
      return `@${node.name} ${node.params}`;
    }
  }
  return null;
}

/** Merge declarations from every rule matching `selector` exactly under `condition`. */
function declsFor(
  root: postcss.Root,
  selector: string,
  condition: string | null = null,
): Map<string, string> {
  const decls = new Map<string, string>();
  root.walkRules((rule: Rule) => {
    if (rule.selector !== selector || conditionOf(rule) !== condition) return;
    rule.walkDecls((decl: Declaration) => {
      decls.set(decl.prop, decl.value);
    });
  });
  return decls;
}

/** The value a `var(--name)` reference takes from the theme layer's root declarations. */
function themeValue(root: postcss.Root, reference: string | undefined): string | undefined {
  const name = /^var\((--[\w-]+)\)$/.exec(reference ?? "")?.[1];
  return name ? declsFor(root, ":root, :host").get(name) : undefined;
}

describe("index.css component layer", () => {
  let root: postcss.Root;

  beforeAll(async () => {
    // vitest's `css: false` stubs CSS imports, so read from disk; no @types/node here.
    // @ts-expect-error TS2307: no @types/node in the frontend tsconfig
    const { readFileSync } = (await import("node:fs")) as {
      readFileSync: (path: string, encoding: "utf8") => string;
    };
    // @ts-expect-error TS2307: no @types/node in the frontend tsconfig
    const { resolve } = (await import("node:path")) as { resolve: (path: string) => string };
    // Vitest runs with cwd = frontend root (vitest.config.ts lives there).
    const rawCss = readFileSync("src/index.css", "utf8");

    // An explicit candidate list instead of a source scan, so only the component classes emit.
    const compiler = await compile(rawCss, { base: resolve("src"), onDependency: () => {} });
    const built = compiler.build([
      "tcip-btn",
      "tcip-btn-primary",
      "tcip-btn-danger",
      "tcip-input",
      "tcip-select",
    ]);
    root = postcss.parse(optimize(built).code);
  });

  // .tcip-btn-primary/.tcip-btn-danger inherit via `@apply tcip-btn`, and
  // .tcip-select via `@apply tcip-input`: assert all so the chain can't break.
  it.each([".tcip-btn", ".tcip-btn-primary", ".tcip-btn-danger", ".tcip-input", ".tcip-select"])(
    "%s has a visible focus-visible ring",
    (base) => {
      const decls = declsFor(root, `${base}:focus-visible`);
      // ring-2 → box-shadow composed from the Tailwind ring variables
      expect(decls.get("box-shadow")).toContain("--tw-ring-offset-shadow");
      // ring-tcip-accent/70 → SI green at 70% opacity (b3 is 179/255), and the same
      // token mixed at 70% where the browser supports color-mix
      expect(decls.get("--tw-ring-color")).toBe("#507754b3");
      const mixed = declsFor(
        root,
        `${base}:focus-visible`,
        "@supports (color: color-mix(in lab, red, red))",
      );
      expect(mixed.get("--tw-ring-color")).toBe(
        "color-mix(in oklab, var(--color-tcip-accent) 70%, transparent)",
      );
      expect(themeValue(root, "var(--color-tcip-accent)")).toBe("#507754");
      // ring-offset-1 + ring-offset-tcip-bg → 1px gap in the page background
      expect(decls.get("--tw-ring-offset-width")).toBe("1px");
      expect(themeValue(root, decls.get("--tw-ring-offset-color"))).toBe("#1e1e1e");
    },
  );

  it("buttons suppress the UA outline only under focus-visible", () => {
    const decls = declsFor(root, ".tcip-btn:focus-visible");
    expect(decls.get("outline-style")).toBe("none");
    // Forced-colors mode keeps a transparent outline the system can paint.
    const forced = declsFor(root, ".tcip-btn:focus-visible", "@media (forced-colors: active)");
    expect(forced.get("outline")).toBe("2px solid #0000");
    // No plain :focus outline suppression on buttons: mouse clicks keep
    // default behavior and only keyboard focus swaps to the ring.
    expect(declsFor(root, ".tcip-btn:focus").size).toBe(0);
  });

  it("inputs keep the accent border swap on focus", () => {
    const decls = declsFor(root, ".tcip-input:focus");
    expect(themeValue(root, decls.get("border-color"))).toBe("#507754");
  });
});
