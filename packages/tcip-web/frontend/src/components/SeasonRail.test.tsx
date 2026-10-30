import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { type Capture, UNDATED_BUCKET } from "@/api/types.generated";
import { SeasonRail } from "@/components/SeasonRail";
import { PICKER } from "@/test/projectRecords";

afterEach(cleanup);

const VALLEY = PICKER.valley.tree.captures;

/** A capture the tree states as dated by `day`, named by it as the backend names one. */
const dated = (day: string): Capture => ({ name: day, date: day });

const UNDATED: Capture = { name: UNDATED_BUCKET, date: null };

describe("SeasonRail", () => {
  it("renders a tick per dated capture", () => {
    render(<SeasonRail captures={VALLEY} />);
    expect(screen.getAllByTestId("season-tick")).toHaveLength(VALLEY.length);
  });

  it("summarizes the span for assistive tech", () => {
    // The produced valley's captures run from 2026-02-11 to 2026-03-24.
    render(<SeasonRail captures={VALLEY} />);
    expect(screen.getByRole("img")).toHaveAttribute(
      "aria-label",
      expect.stringContaining("from Feb 11 to Mar 24"),
    );
  });

  it("counts a capture no date names as undated", () => {
    render(<SeasonRail captures={[...VALLEY, UNDATED]} />);
    expect(screen.getByText("1 undated")).toBeInTheDocument();
    expect(screen.getAllByTestId("season-tick")).toHaveLength(VALLEY.length);
  });

  it("renders nothing when there are no captures", () => {
    const { container } = render(<SeasonRail captures={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("places a season that crosses a year boundary by its days", () => {
    // The winter window: Dec 2025 → Feb 2026, in the order the tree lists it. The December tick
    // must sit left of February.
    render(
      <SeasonRail captures={[dated("2025-12-20"), dated("2026-01-15"), dated("2026-02-11")]} />,
    );
    const ticks = screen.getAllByTestId("season-tick");
    const left = (el: Element) =>
      parseFloat((el.getAttribute("style") || "").match(/left: ([\d.]+)%/)![1]);
    expect(left(ticks[0])).toBe(0); // earliest (Dec 20) at the left
    expect(left(ticks[2])).toBe(100); // latest (Feb 11) at the right
  });

  it("handles a single date without dividing by zero", () => {
    const [only] = VALLEY;
    render(<SeasonRail captures={[only]} active={only.name} />);
    const ticks = screen.getAllByTestId("season-tick");
    expect(ticks).toHaveLength(1);
    // Centered when there's only one point.
    expect(ticks[0].getAttribute("style")).toContain("left: 50%");
  });
});
