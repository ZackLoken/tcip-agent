import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";

import { AttributeEditors } from "@/components/annotate/AttributeEditors";

const REGISTRY = {
  bud: { attributes: { opening: { type: "categorical" as const, values: ["closed", "open"] } } },
};

describe("AttributeEditors unset option naming", () => {
  it("names the always-glyph reset option after the attribute it clears", () => {
    render(
      <AttributeEditors subject="bud" attributes={{}} registry={REGISTRY} onChange={vi.fn()} />,
    );

    const select = screen.getByRole("combobox") as HTMLSelectElement;
    expect(within(select).getByRole("option", { name: "no opening value" })).toBeInTheDocument();
  });
});
