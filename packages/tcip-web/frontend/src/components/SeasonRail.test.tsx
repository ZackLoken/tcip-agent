import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { SeasonRail } from "@/components/SeasonRail";

afterEach(cleanup);

describe("SeasonRail", () => {
  it("renders a tick per ISO capture date", () => {
    render(<SeasonRail dates={["2026-02-11", "2026-03-01", "2026-03-20"]} />);
    expect(screen.getAllByTestId("season-tick")).toHaveLength(3);
  });

  it("summarizes the span for assistive tech", () => {
    render(<SeasonRail dates={["2026-02-11", "2026-03-01"]} />);
    expect(screen.getByRole("img")).toHaveAttribute(
      "aria-label",
      expect.stringContaining("from Feb 11 to Mar 1"),
    );
  });

  it("shows the undated bucket count", () => {
    render(<SeasonRail dates={["2026-02-11", "undated"]} />);
    expect(screen.getByText("1 undated")).toBeInTheDocument();
    expect(screen.getAllByTestId("season-tick")).toHaveLength(1);
  });

  it("renders nothing when there are no dates", () => {
    const { container } = render(<SeasonRail dates={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("orders a season that crosses a year boundary chronologically", () => {
    // The winter window: Dec 2025 → Feb 2026. The December tick must sit left of February.
    render(<SeasonRail dates={["2025-12-20", "2026-01-15", "2026-02-11"]} />);
    const ticks = screen.getAllByTestId("season-tick");
    const left = (el: Element) =>
      parseFloat((el.getAttribute("style") || "").match(/left: ([\d.]+)%/)![1]);
    expect(left(ticks[0])).toBe(0); // earliest (Dec 20) at the left
    expect(left(ticks[2])).toBe(100); // latest (Feb 11) at the right
  });

  it("drops ISO-shaped but invalid dates instead of mislabeling them", () => {
    render(<SeasonRail dates={["2026-02-11", "2026-13-40"]} />);
    // Only the one real date renders a tick; the invalid one is counted as undated.
    expect(screen.getAllByTestId("season-tick")).toHaveLength(1);
    expect(screen.getByText("1 undated")).toBeInTheDocument();
  });

  it("handles a single date without dividing by zero", () => {
    render(<SeasonRail dates={["2026-02-11"]} active="2026-02-11" />);
    const ticks = screen.getAllByTestId("season-tick");
    expect(ticks).toHaveLength(1);
    // Centered when there's only one point.
    expect(ticks[0].getAttribute("style")).toContain("left: 50%");
  });
});
