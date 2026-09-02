import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { CoverageChrome } from "@/components/Canvas/CoverageChrome";

beforeEach(() => {
  localStorage.clear();
});

afterEach(() => {
  cleanup();
});

function baseProps() {
  return {
    subject: "fruit",
    derivation: "cells sized to one full-resolution screenful",
    gridFetchError: null,
    readError: null,
    countsError: null,
    overlayOn: true,
    onToggleOverlay: vi.fn(),
    currentCellName: "A1",
    currentCellComplete: false,
    currentCellStale: false,
    otherLattice: null,
    sweptOtherLattice: null,
    swept: new Set<string>(),
    activeComplete: new Set<string>(),
    activeStale: new Set<string>(),
    annotationCounts: {},
    onAttest: vi.fn(),
  };
}

function openKey() {
  fireEvent.click(screen.getByRole("button", { name: "Key" }));
}

describe("CoverageChrome", () => {
  it("the overlay toggle has an accessible name and states its own on/off state", () => {
    render(<CoverageChrome {...baseProps()} overlayOn />);
    const toggle = screen.getByRole("button", { name: /Overlay on/ });
    expect(toggle).toHaveAttribute("aria-pressed", "true");
  });

  it("the heading names the active subject", () => {
    render(<CoverageChrome {...baseProps()} />);
    expect(screen.getByText("Coverage for fruit")).toBeInTheDocument();
  });

  it("states the grid's own derivation and that a cell is not a training tile", () => {
    render(<CoverageChrome {...baseProps()} />);
    expect(
      screen.getByText(
        /cells sized to one full-resolution screenful.*A cell is not a training tile/,
      ),
    ).toBeInTheDocument();
  });

  it("shows a grid fetch failure as the error, never the derivation line", () => {
    render(<CoverageChrome {...baseProps()} gridFetchError="band group incomplete" />);
    expect(
      screen.getByText(/coverage grid unavailable: band group incomplete/),
    ).toBeInTheDocument();
    expect(screen.queryByText(/A cell is not a training tile/)).not.toBeInTheDocument();
  });

  it("labels the control Attest, naming the subject, when the cell has never been attested", () => {
    render(<CoverageChrome {...baseProps()} />);
    expect(
      screen.getByRole("button", { name: "Attest A1 complete for fruit" }),
    ).toBeInTheDocument();
  });

  it("labels the control Unattest, naming the subject, when the cell is attested and fresh", () => {
    render(<CoverageChrome {...baseProps()} currentCellComplete />);
    expect(screen.getByRole("button", { name: "Unattest A1 for fruit" })).toBeInTheDocument();
  });

  it("labels the control Re-attest, naming the subject and staleness, when the cell is stale", () => {
    render(<CoverageChrome {...baseProps()} currentCellComplete currentCellStale />);
    expect(
      screen.getByRole("button", { name: "Re-attest A1 for fruit (changed since attested)" }),
    ).toBeInTheDocument();
  });

  it("an ordinary Attest press writes immediately, complete=true", () => {
    const onAttest = vi.fn();
    render(<CoverageChrome {...baseProps()} onAttest={onAttest} />);
    fireEvent.click(screen.getByRole("button", { name: "Attest A1 complete for fruit" }));
    expect(onAttest).toHaveBeenCalledWith(true);
  });

  it("an ordinary Unattest press writes immediately, complete=false", () => {
    const onAttest = vi.fn();
    render(<CoverageChrome {...baseProps()} currentCellComplete onAttest={onAttest} />);
    fireEvent.click(screen.getByRole("button", { name: "Unattest A1 for fruit" }));
    expect(onAttest).toHaveBeenCalledWith(false);
  });

  it("no attestation control, and no subject-naming label, without an active subject", () => {
    render(<CoverageChrome {...baseProps()} subject={null} />);
    expect(screen.queryByRole("button", { name: /Attest/ })).not.toBeInTheDocument();
    expect(screen.getByText("Coverage grid")).toBeInTheDocument();
  });

  it("the attest control is offered only while the overlay is on", () => {
    render(<CoverageChrome {...baseProps()} overlayOn={false} />);
    expect(screen.queryByRole("button", { name: /Attest/ })).not.toBeInTheDocument();
    expect(screen.getByText(/Turn the overlay on to attest A1/)).toBeInTheDocument();
  });

  it("a lattice-changing write needs a second confirming press before it fires, with a Cancel beside it", () => {
    const onAttest = vi.fn();
    render(
      <CoverageChrome
        {...baseProps()}
        otherLattice={{ count: 3, cols: 4, rows: 2 }}
        onAttest={onAttest}
      />,
    );
    const first = screen.getByRole("button", { name: "Attest A1 complete for fruit" });
    fireEvent.click(first);
    expect(onAttest).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", {
        name: "Confirm: attest A1 for fruit (discards 3 previous attestations)",
      }),
    ).toBeInTheDocument();
    expect(screen.getByText(/3 cells attested on a previous lattice \(4x2\)/)).toBeInTheDocument();

    const cancel = screen.getByRole("button", { name: "Cancel" });
    fireEvent.click(cancel);
    expect(onAttest).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Cancel" })).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Attest A1 complete for fruit" }),
    ).toBeInTheDocument();
  });

  it("a lattice-changing write fires on the second press", () => {
    const onAttest = vi.fn();
    render(
      <CoverageChrome
        {...baseProps()}
        otherLattice={{ count: 1, cols: 2, rows: 2 }}
        onAttest={onAttest}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Attest A1 complete for fruit" }));
    fireEvent.click(
      screen.getByRole("button", {
        name: "Confirm: attest A1 for fruit (discards 1 previous attestation)",
      }),
    );
    expect(onAttest).toHaveBeenCalledWith(true);
  });

  it("arming and cancelling a confirmation is announced through a status region", () => {
    render(<CoverageChrome {...baseProps()} otherLattice={{ count: 1, cols: 2, rows: 2 }} />);
    fireEvent.click(screen.getByRole("button", { name: "Attest A1 complete for fruit" }));
    expect(screen.getByRole("status").textContent).toMatch(/Confirmation armed/);
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.getByRole("status").textContent).toMatch(/cancelled/);
  });

  it("a confirmation armed for one cell does not carry over to the next", () => {
    const onAttest = vi.fn();
    // One shared reference isolates the reset this test names (currentCellName).
    const otherLattice = { count: 1, cols: 2, rows: 2 };
    const { rerender } = render(
      <CoverageChrome {...baseProps()} otherLattice={otherLattice} onAttest={onAttest} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Attest A1 complete for fruit" }));
    expect(screen.getByText(/Confirm:/)).toBeInTheDocument();

    rerender(
      <CoverageChrome
        {...baseProps()}
        currentCellName="B2"
        otherLattice={otherLattice}
        onAttest={onAttest}
      />,
    );
    expect(screen.queryByText(/Confirm:/)).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Attest B2 complete for fruit" }),
    ).toBeInTheDocument();
  });

  it("no control or overlay-off message shown when no cell is under the viewport", () => {
    render(<CoverageChrome {...baseProps()} currentCellName={null} />);
    expect(screen.queryByText(/Attest/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Turn the overlay on/)).not.toBeInTheDocument();
  });

  it("the Key is closed by default and opens on a click, not hover", () => {
    render(<CoverageChrome {...baseProps()} />);
    const key = screen.getByRole("button", { name: "Key" });
    expect(key).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText(/swept: the recorded sweep history/)).not.toBeInTheDocument();
    fireEvent.click(key);
    expect(key).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/swept: the recorded sweep history, any session/)).toBeInTheDocument();
  });

  it("the Key names every lattice mark, since the overlay itself names none", () => {
    render(<CoverageChrome {...baseProps()} />);
    openKey();
    expect(screen.getByText(/swept: the recorded sweep history, any session/)).toBeInTheDocument();
    expect(screen.getByText(/saved annotations/i)).toBeInTheDocument();
    expect(screen.getAllByText(/attested complete/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/changed since attested/i)).toBeInTheDocument();
  });

  it("a completeness read failure states the fix-first fact, in the breeder's own words, and is announced", () => {
    render(<CoverageChrome {...baseProps()} readError="plot.json: not valid JSON" />);
    const message = screen
      .getAllByRole("status")
      .find((el) => el.textContent?.includes("this image's labels could not be read"));
    expect(message).toBeTruthy();
    expect(message!.textContent).toContain(
      "this image's labels could not be read, so nothing can be attested until they are fixed",
    );
    expect(message!.textContent).toContain("plot.json: not valid JSON");
    expect(screen.queryByRole("button", { name: /Attest/ })).not.toBeInTheDocument();
  });

  it("a completeness failure on a raster with no grid states the error, not an empty derivation line", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        derivation=""
        currentCellName={null}
        readError="network down"
      />,
    );
    expect(screen.getByText(/network down/)).toBeInTheDocument();
    expect(screen.queryByText(/A cell is not a training tile/)).not.toBeInTheDocument();
  });

  it("states a sweep record on a previous lattice, the way the attestation equivalent does", () => {
    render(<CoverageChrome {...baseProps()} sweptOtherLattice={{ count: 2, cols: 6, rows: 6 }} />);
    expect(screen.getByText(/2 cells swept on a previous lattice \(6x6\)/)).toBeInTheDocument();
  });

  it("carries a visually hidden list of the lattice's own held states, for the accessibility tree", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        swept={new Set(["B1", "B2"])}
        activeComplete={new Set(["B2"])}
        activeStale={new Set(["B2"])}
        annotationCounts={{ A1: 2, B2: 1 }}
      />,
    );
    expect(screen.getByText("swept: B1, B2")).toBeInTheDocument();
    expect(screen.getByText("saved for fruit: A1 (2), B2 (1)")).toBeInTheDocument();
    expect(screen.getByText("attested: B2")).toBeInTheDocument();
    expect(screen.getByText("changed since attested: B2")).toBeInTheDocument();
  });

  it("omits a state line entirely when no cell holds it", () => {
    render(<CoverageChrome {...baseProps()} />);
    expect(screen.queryByText(/^swept:/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^attested:/)).not.toBeInTheDocument();
  });

  it("the panel collapses and remembers its state across a remount, like the app's other disclosures", () => {
    const { unmount } = render(<CoverageChrome {...baseProps()} />);
    expect(screen.getByText(/A cell is not a training tile/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Coverage for fruit/ }));
    expect(screen.queryByText(/A cell is not a training tile/)).not.toBeInTheDocument();
    unmount();

    render(<CoverageChrome {...baseProps()} />);
    expect(screen.queryByText(/A cell is not a training tile/)).not.toBeInTheDocument();
  });
});
