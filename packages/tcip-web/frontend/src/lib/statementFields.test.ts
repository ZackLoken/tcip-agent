import { describe, expect, it } from "vitest";

import { STATEMENT_FIELD_LABELS } from "./statementFields";

describe("STATEMENT_FIELD_LABELS", () => {
  it("labels the majority crossing marker without the reserved word provisional", () => {
    expect(STATEMENT_FIELD_LABELS.majority_provisional).toBe(
      "Majority milestone crossing pending breeder confirmation",
    );
  });

  it("keeps every label non-empty, since a blank label would render as nothing where a missing one falls back to the field name", () => {
    for (const [field, label] of Object.entries(STATEMENT_FIELD_LABELS)) {
      expect(label, field).not.toBe("");
    }
  });
});
