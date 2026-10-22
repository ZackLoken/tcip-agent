import { describe, expect, it } from "vitest";

import { formatNumber } from "@/lib/numberFormat";

describe("formatNumber", () => {
  it("writes a small learning rate in scientific notation at four significant figures", () => {
    expect(formatNumber(0.00009045084971874738)).toBe("9.045e-5");
  });

  it("writes a mid-range value at four significant figures with no trailing zeros", () => {
    expect(formatNumber(0.873412)).toBe("0.8734");
    expect(formatNumber(12.3456)).toBe("12.35");
    expect(formatNumber(0.001)).toBe("0.001");
  });

  it("writes a whole number as it is and a large fraction in scientific notation", () => {
    expect(formatNumber(1234)).toBe("1234");
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(123456.789)).toBe("1.235e+5");
  });
});
