import { describe, expect, it } from "vitest";

import { schemaChangeSweepToast } from "@/lib/registrySweep";

describe("schemaChangeSweepToast", () => {
  it("is null with nothing predating the change and no warning, whatever newly_stamped holds", () => {
    const toast = schemaChangeSweepToast({
      newly_stamped: { leaf: 3 },
      predating_vocabulary: {},
      warning: null,
    });
    expect(toast).toBeNull();
  });
});
