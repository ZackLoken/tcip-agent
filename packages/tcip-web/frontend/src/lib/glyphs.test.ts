import { describe, expect, it } from "vitest";

import { isNonTestSource } from "../../scripts/routeReads.mjs";

import { UNSET_GLYPH } from "@/lib/glyphs";

const EM_DASH = String.fromCharCode(0x2014);

// Every non-test source .tsx under src, as raw text keyed by path, resolved eagerly (a Vite
// build-time feature, not a filesystem read) so the check runs synchronously against what ships.
const sourceFiles = Object.entries(
  import.meta.glob("/src/**/*.tsx", {
    eager: true,
    query: "?raw",
    import: "default",
  }) as Record<string, string>,
).filter(([path]) => isNonTestSource(path.slice("/src/".length)));

describe("UNSET_GLYPH", () => {
  it("is a colon, never an em dash or a hyphen", () => {
    expect(UNSET_GLYPH).toBe(":");
  });
});

describe("unset-value placeholder sweep", () => {
  it("renders no em dash or bare hyphen as a value placeholder anywhere under src", () => {
    const offenders = sourceFiles
      .filter(([, content]) => content.includes(EM_DASH) || content.includes(">-<"))
      .map(([path]) => path);
    expect(offenders).toEqual([]);
  });

  it("gives every <option> whose only child is the bare glyph an aria-label", () => {
    const glyphOptionPattern = new RegExp(
      `<option\\b[^>]*>\\s*\\{${UNSET_GLYPH.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\}\\s*</option>`,
      "g",
    );
    const offenders = sourceFiles.flatMap(([path, content]) => {
      const matches = content.match(glyphOptionPattern) ?? [];
      return matches.filter((tag) => !tag.includes("aria-label")).map(() => path);
    });
    expect(offenders).toEqual([]);
  });
});
