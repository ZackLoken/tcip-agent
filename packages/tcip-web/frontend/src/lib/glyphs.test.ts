import { describe, expect, it } from "vitest";

import { nonTestSources } from "../../scripts/routeReads.mjs";

import { UNSET_GLYPH } from "@/lib/glyphs";
import { readFileSync, SOURCE_DIRECTORY } from "@/test/node";

const EM_DASH = String.fromCharCode(0x2014);

// Every non-test source .tsx under the source directory, as its path and its text.
const sourceFiles = nonTestSources(SOURCE_DIRECTORY)
  .filter((path) => path.endsWith(".tsx"))
  .map((path) => [path, readFileSync(path, "utf8")]);

describe("UNSET_GLYPH", () => {
  it("is a colon, never an em dash or a hyphen", () => {
    expect(UNSET_GLYPH).toBe(":");
  });
});

describe("unset-value placeholder sweep", () => {
  it("renders no em dash or bare hyphen as a value placeholder in any source", () => {
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
