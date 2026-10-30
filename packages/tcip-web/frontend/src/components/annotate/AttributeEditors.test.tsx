import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AttributeEditors } from "@/components/annotate/AttributeEditors";
import { LABEL_ANSWERS, SERVED_ATTRIBUTES } from "@/test/labels";

const REGISTRY = LABEL_ANSWERS.registry_loaded.subjects ?? {};
const [SUBJECT] = Object.keys(REGISTRY);
const [ATTRIBUTE] = Object.keys(SERVED_ATTRIBUTES);

describe("AttributeEditors unset option naming", () => {
  it("names the always-glyph reset option after the attribute it clears", () => {
    render(
      <AttributeEditors subject={SUBJECT} attributes={{}} registry={REGISTRY} onChange={vi.fn()} />,
    );

    const select = screen.getByRole("combobox");
    expect(
      within(select).getByRole("option", { name: `no ${ATTRIBUTE} value` }),
    ).toBeInTheDocument();
  });
});
