import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { CoverageChrome } from "@/components/Canvas/CoverageChrome";

afterEach(() => {
  cleanup();
});

function baseProps() {
  return {
    derivation: "one display-bounded serve per cell",
    gridFetchError: null,
    readError: null,
    countsError: null,
    overlayOn: false,
    onToggleOverlay: vi.fn(),
    currentCellName: "A1",
    currentCellComplete: false,
    currentCellStale: false,
    otherLattice: null,
    onAttest: vi.fn(),
  };
}

describe("CoverageChrome", () => {
  it("the overlay toggle has an accessible name and states its own on/off state", () => {
    render(<CoverageChrome {...baseProps()} overlayOn />);
    const toggle = screen.getByRole("button", { name: /Overlay on/ });
    expect(toggle).toHaveAttribute("aria-pressed", "true");
  });

  it("states the grid's own derivation and that a cell is not a training tile", () => {
    render(<CoverageChrome {...baseProps()} />);
    expect(
      screen.getByText(/one display-bounded serve per cell.*A cell is not a training tile/),
    ).toBeInTheDocument();
  });

  it("shows a grid fetch failure as the error, never the derivation line", () => {
    render(<CoverageChrome {...baseProps()} gridFetchError="band group incomplete" />);
    expect(
      screen.getByText(/coverage grid unavailable: band group incomplete/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/A cell is not a training tile/)).not.toBeInTheDocument();
  });

  it("labels the control Attest when the cell has never been attested", () => {
    render(<CoverageChrome {...baseProps()} />);
    expect(screen.getByRole("button", { name: "Attest A1 complete" })).toBeInTheDocument();
  });

  it("labels the control Unattest when the cell is attested and fresh", () => {
    render(<CoverageChrome {...baseProps()} currentCellComplete />);
    expect(screen.getByRole("button", { name: "Unattest A1" })).toBeInTheDocument();
  });

  it("labels the control Re-attest, naming staleness, when the cell is attested but stale", () => {
    render(<CoverageChrome {...baseProps()} currentCellComplete currentCellStale />);
    expect(
      screen.getByRole("button", { name: "Re-attest A1 (changed since attested)" }),
    ).toBeInTheDocument();
  });

  it("an ordinary Attest press writes immediately, complete=true", () => {
    const onAttest = vi.fn();
    render(<CoverageChrome {...baseProps()} onAttest={onAttest} />);
    fireEvent.click(screen.getByRole("button", { name: "Attest A1 complete" }));
    expect(onAttest).toHaveBeenCalledWith(true);
  });

  it("an ordinary Unattest press writes immediately, complete=false", () => {
    const onAttest = vi.fn();
    render(<CoverageChrome {...baseProps()} currentCellComplete onAttest={onAttest} />);
    fireEvent.click(screen.getByRole("button", { name: "Unattest A1" }));
    expect(onAttest).toHaveBeenCalledWith(false);
  });

  it("a lattice-changing write needs a second confirming press before it fires", () => {
    const onAttest = vi.fn();
    render(
      <CoverageChrome
        {...baseProps()}
        otherLattice={{ count: 3, cols: 4, rows: 2 }}
        onAttest={onAttest}
      />,
    );
    const first = screen.getByRole("button", { name: "Attest A1 complete" });
    fireEvent.click(first);
    expect(onAttest).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", {
        name: "Confirm: attest A1 (discards 3 previous attestations)",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText(/3 cells attested on a previous lattice \(4x2\)/)).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: "Confirm: attest A1 (discards 3 previous attestations)" }),
    );
    expect(onAttest).toHaveBeenCalledWith(true);
  });

  it("a confirmation armed for one cell does not carry over to the next", () => {
    const onAttest = vi.fn();
    const { rerender } = render(
      <CoverageChrome
        {...baseProps()}
        otherLattice={{ count: 1, cols: 2, rows: 2 }}
        onAttest={onAttest}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Attest A1 complete" }));
    expect(screen.getByText(/Confirm:/)).toBeInTheDocument();

    rerender(
      <CoverageChrome
        {...baseProps()}
        currentCellName="B2"
        otherLattice={{ count: 1, cols: 2, rows: 2 }}
        onAttest={onAttest}
      />,
    );
    expect(screen.queryByText(/Confirm:/)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Attest B2 complete" })).toBeInTheDocument();
  });

  it("no control shown when no cell is under the viewport", () => {
    render(<CoverageChrome {...baseProps()} currentCellName={null} />);
    expect(screen.queryByText(/Attest/)).not.toBeInTheDocument();
  });

  it("a completeness read failure hides the attestation control and shows the error", () => {
    render(<CoverageChrome {...baseProps()} readError="network down" />);
    expect(screen.getByText(/completeness unavailable: network down/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /Attest/ })).not.toBeInTheDocument();
  });
});
