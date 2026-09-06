import { describe, expect, it } from "vitest";

import { UNSET_GLYPH } from "@/lib/glyphs";

const EM_DASH = String.fromCharCode(0x2014);

// Every source .tsx under src, as raw text keyed by path, resolved eagerly (a Vite build-time
// feature, not a filesystem read) so the check runs synchronously against what actually ships.
const sourceFiles = import.meta.glob("/src/**/*.tsx", {
  eager: true,
  query: "?raw",
  import: "default",
}) as Record<string, string>;

describe("UNSET_GLYPH", () => {
  it("is the owner's chosen colon, never an em dash or a hyphen", () => {
    expect(UNSET_GLYPH).toBe(":");
  });
});

describe("unset-value placeholder sweep", () => {
  it("renders no em dash or bare hyphen as a value placeholder anywhere under src", () => {
    const offenders = Object.entries(sourceFiles)
      .filter(([path]) => !path.endsWith(".test.tsx"))
      .filter(([, content]) => content.includes(EM_DASH) || content.includes(">-<"))
      .map(([path]) => path);
    expect(offenders).toEqual([]);
  });

  it("keeps a file rendering the bare glyph as option text beside an aria-label, as coverage only: a static sweep cannot see which JSX branch renders", () => {
    const offenders = Object.entries(sourceFiles)
      .filter(([path]) => !path.endsWith(".test.tsx"))
      .filter(([, content]) => content.includes(`>${UNSET_GLYPH}<`))
      .filter(([, content]) => !content.includes("aria-label"))
      .map(([path]) => path);
    expect(offenders).toEqual([]);
  });
});
