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
    reason: null,
    settled: true,
    freshDerivationDiffers: null,
    onRederiveLattice: vi.fn(),
    onSetGridZoom: vi.fn(),
    gridFetchError: null,
    readError: null,
    countsError: null,
    canOverlay: true,
    overlayOn: true,
    onToggleOverlay: vi.fn(),
    currentCellName: "A1",
    currentCellComplete: false,
    currentCellStale: false,
    otherLattice: null,
    replaceRequired: null,
    onArmReplace: vi.fn(),
    swept: new Set<string>(),
    pending: new Set<string>(),
    coarserCount: 0,
    workingScale: null,
    workingScaleReason: null,
    fitScale: null,
    activeComplete: new Set<string>(),
    activeStale: new Set<string>(),
    activeCellsAttestedView: {},
    otherComplete: {},
    annotationCounts: {},
    onAttest: vi.fn(),
  };
}

function bar(zoom: number) {
  return { zoom, source: "set by user:breeder at 2026-09-03T00:00:00+00:00" };
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
    expect(
      screen.queryByText(/swept: every part of the cell has been on screen at the working scale/),
    ).not.toBeInTheDocument();
    fireEvent.click(key);
    expect(key).toHaveAttribute("aria-expanded", "true");
    expect(
      screen.getByText(
        /swept: every part of the cell has been on screen at the working scale, any session/,
      ),
    ).toBeInTheDocument();
  });

  it("the Key names every lattice mark, since the overlay itself names none", () => {
    render(<CoverageChrome {...baseProps()} />);
    openKey();
    expect(
      screen.getByText(
        /swept: every part of the cell has been on screen at the working scale, any session/,
      ),
    ).toBeInTheDocument();
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

  it("states a replace hold, the way the attestation equivalent states its own", () => {
    render(
      <CoverageChrome {...baseProps()} replaceRequired={{ cellsSeen: 2, cols: 6, rows: 6 }} />,
    );
    expect(
      screen.getByText(
        /2 cells seen on a previous lattice \(6x6\); progress on this lattice is not saved/,
      ),
    ).toBeInTheDocument();
  });

  it("states a replace hold with no cells seen by naming the record itself, never 0 cells", () => {
    render(
      <CoverageChrome {...baseProps()} replaceRequired={{ cellsSeen: 0, cols: 6, rows: 6 }} />,
    );
    expect(
      screen.getByText(
        /a previous lattice's record \(6x6\) with no cells seen; progress on this lattice is not saved/,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/0 cells seen/)).not.toBeInTheDocument();
  });

  it("renders the replace control with the panel collapsed, the overlay off and no current cell", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        replaceRequired={{ cellsSeen: 3, cols: 4, rows: 4 }}
        overlayOn={false}
        currentCellName={null}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Coverage for fruit/ })); // collapse
    expect(screen.getByRole("button", { name: "Replace" })).toBeInTheDocument();
  });

  it("a Replace press arms a confirmation, a second press confirms it", () => {
    const onArmReplace = vi.fn();
    render(
      <CoverageChrome
        {...baseProps()}
        replaceRequired={{ cellsSeen: 3, cols: 4, rows: 4 }}
        onArmReplace={onArmReplace}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Replace" }));
    expect(onArmReplace).not.toHaveBeenCalled();
    const confirm = screen.getByRole("button", {
      name: "Confirm: discard 3 cells seen on the previous lattice",
    });
    fireEvent.click(confirm);
    expect(onArmReplace).toHaveBeenCalledTimes(1);
  });

  it("Cancel leaves the hold: no call, and the control returns to its unarmed label", () => {
    const onArmReplace = vi.fn();
    render(
      <CoverageChrome
        {...baseProps()}
        replaceRequired={{ cellsSeen: 1, cols: 2, rows: 2 }}
        onArmReplace={onArmReplace}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Replace" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(onArmReplace).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Replace" })).toBeInTheDocument();
    expect(screen.getByText(/1 cell seen on a previous lattice \(2x2\)/)).toBeInTheDocument();
  });

  it("the replace control's armed state is tied to the hold, not the current cell", () => {
    const replaceRequired = { cellsSeen: 1, cols: 2, rows: 2 };
    const { rerender } = render(
      <CoverageChrome {...baseProps()} replaceRequired={replaceRequired} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Replace" }));
    expect(screen.getByText(/Confirm: discard/)).toBeInTheDocument();

    rerender(
      <CoverageChrome {...baseProps()} currentCellName="B2" replaceRequired={replaceRequired} />,
    );
    expect(screen.getByText(/Confirm: discard/)).toBeInTheDocument();
  });

  it("the attest control's confirmation is unaffected by a standing replace hold", () => {
    const onAttest = vi.fn();
    render(
      <CoverageChrome
        {...baseProps()}
        replaceRequired={{ cellsSeen: 1, cols: 2, rows: 2 }}
        onAttest={onAttest}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Attest A1 complete for fruit" }));
    expect(onAttest).toHaveBeenCalledWith(true);
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

  it("collapsing the chrome closes an already-open Key", () => {
    render(<CoverageChrome {...baseProps()} />);
    openKey();
    expect(
      screen.getByText(
        /swept: every part of the cell has been on screen at the working scale, any session/,
      ),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Coverage for fruit/ }));
    expect(
      screen.queryByText(
        /swept: every part of the cell has been on screen at the working scale, any session/,
      ),
    ).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Key" })).toHaveAttribute("aria-expanded", "false");
  });

  it("a collapsed panel still shows the read-error notice and the status region", () => {
    render(<CoverageChrome {...baseProps()} readError="plot.json: not valid JSON" />);
    fireEvent.click(screen.getByRole("button", { name: /Coverage for fruit/ }));
    expect(screen.getByText(/this image's labels could not be read/)).toBeInTheDocument();
    expect(screen.getAllByRole("status").length).toBeGreaterThan(0);
    // The body's own content is gone, proof the notice lives outside it.
    expect(screen.queryByText(/A cell is not a training tile/)).not.toBeInTheDocument();
  });

  it("a collapsed panel still shows a previous-lattice notice", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        otherLattice={{ count: 2, cols: 4, rows: 4 }}
        replaceRequired={{ cellsSeen: 1, cols: 6, rows: 6 }}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: /Coverage for fruit/ }));
    expect(screen.getByText(/2 cells attested on a previous lattice \(4x4\)/)).toBeInTheDocument();
    expect(screen.getByText(/1 cell seen on a previous lattice \(6x6\)/)).toBeInTheDocument();
  });

  it("the read-error text strips the record's dictionary, keeping the reader's own sentence", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        readError="record 0 carries no string subject: {'id': 1, 'category_id': 0}"
      />,
    );
    expect(screen.getByText(/record 0 carries no string subject$/)).toBeInTheDocument();
    expect(screen.queryByText(/'id': 1/)).not.toBeInTheDocument();
  });

  it("another subject's attestation reaches the hidden state list", () => {
    render(<CoverageChrome {...baseProps()} otherComplete={{ leaf: ["B1"] }} />);
    expect(screen.getByText("attested for leaf: B1")).toBeInTheDocument();
  });

  it("the overlay toggle is withdrawn, and the attest control offered directly, when the raster cannot draw an overlay", () => {
    render(<CoverageChrome {...baseProps()} canOverlay={false} overlayOn={false} />);
    expect(screen.queryByRole("button", { name: /Overlay/ })).not.toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Attest A1 complete for fruit" }),
    ).toBeInTheDocument();
    expect(screen.queryByText(/Turn the overlay on/)).not.toBeInTheDocument();
  });

  it("states the served working scale from the served fields, never a literal", () => {
    render(<CoverageChrome {...baseProps()} workingScale={bar(0.25)} />);
    expect(screen.getByText(/Working scale for fruit: 25\.0%/)).toBeInTheDocument();
    expect(screen.getByText(/set by user:breeder at/)).toBeInTheDocument();
  });

  it("states the reason sentence instead when there is no bar", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        workingScaleReason="no saved box or polygon annotation of fruit"
      />,
    );
    expect(screen.getByText("no saved box or polygon annotation of fruit")).toBeInTheDocument();
  });

  it("notes when the bar sits coarser than the whole-image view", () => {
    render(<CoverageChrome {...baseProps()} workingScale={bar(0.02)} fitScale={0.1} />);
    expect(
      screen.getByText(/fruit's working scale \(2\.0%\) is coarser than the whole-image view/),
    ).toBeInTheDocument();
  });

  it("says nothing about the fit scale when the bar sits within it", () => {
    render(<CoverageChrome {...baseProps()} workingScale={bar(0.5)} fitScale={0.1} />);
    expect(screen.queryByText(/coarser than the whole-image view/)).not.toBeInTheDocument();
  });

  it("states the coarser-cells remainder against the bar, and the no-bar wording without one", () => {
    const { rerender } = render(
      <CoverageChrome {...baseProps()} workingScale={bar(0.5)} coarserCount={2} />,
    );
    expect(
      screen.getByText(/2 cells on record were seen at a coarser scale than fruit's working scale/),
    ).toBeInTheDocument();

    rerender(<CoverageChrome {...baseProps()} workingScale={null} coarserCount={2} />);
    expect(
      screen.getByText(
        /2 cells on record were fully on screen; no working scale to judge them against/,
      ),
    ).toBeInTheDocument();
  });

  it("renders no coarser-cells line when the count is zero, bar present or absent", () => {
    const { rerender } = render(
      <CoverageChrome {...baseProps()} workingScale={bar(0.5)} coarserCount={0} />,
    );
    expect(screen.queryByText(/on record/)).not.toBeInTheDocument();

    rerender(<CoverageChrome {...baseProps()} workingScale={null} coarserCount={0} />);
    expect(screen.queryByText(/on record/)).not.toBeInTheDocument();
  });

  it("carries a not-yet-saved line for pending cells that have also swept, in the hidden state list", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        swept={new Set(["A1", "B2"])}
        pending={new Set(["A1", "B2"])}
      />,
    );
    expect(screen.getByText("not yet saved: A1, B2")).toBeInTheDocument();
  });

  it("excludes a pending cell that has not yet swept, the same set the overlay marks", () => {
    render(
      <CoverageChrome {...baseProps()} swept={new Set(["A1"])} pending={new Set(["A1", "B2"])} />,
    );
    expect(screen.getByText("not yet saved: A1")).toBeInTheDocument();
    expect(screen.queryByText(/B2/)).not.toBeInTheDocument();
  });

  it("states the current cell's attestation scale provenance for a screen-reader user", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        activeCellsAttestedView={{
          A1: {
            view_scale: 0.5,
            working_scale_at_write: bar(0.4),
            seen_on_record: { at_scale: 0.6, grid_matched: true },
          },
        }}
      />,
    );
    expect(
      screen.getByText(/A1 attested at 50\.0% zoom, seen on record at 60\.0%/),
    ).toBeInTheDocument();
  });

  it("states no working scale recorded at attestation, with no verdict, when the scale at write is null", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        activeCellsAttestedView={{
          A1: {
            view_scale: 0.5,
            working_scale_at_write: null,
            seen_on_record: { at_scale: 0.6, grid_matched: true },
          },
        }}
      />,
    );
    expect(
      screen.getByText(
        /A1 attested at 50\.0% zoom, seen on record at 60\.0%, no working scale recorded at attestation/,
      ),
    ).toBeInTheDocument();
    expect(screen.queryByText(/below it/)).not.toBeInTheDocument();
  });

  it("names the working scale the attestation was judged against", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        activeCellsAttestedView={{
          A1: {
            view_scale: 0.5,
            working_scale_at_write: bar(0.4),
            seen_on_record: { at_scale: 0.6, grid_matched: true },
          },
        }}
      />,
    );
    expect(screen.getByText(/against a working scale of 40\.0%/)).toBeInTheDocument();
  });

  it("states not seen on record when the coverage record never showed the cell", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        activeCellsAttestedView={{
          A1: {
            view_scale: 0.5,
            working_scale_at_write: null,
            seen_on_record: { at_scale: null, grid_matched: false },
          },
        }}
      />,
    );
    expect(screen.getByText(/A1 attested at 50\.0% zoom, not seen on record/)).toBeInTheDocument();
  });

  it("shows the no-lattice reason once settled, with the grid-zoom control to set one", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        derivation=""
        currentCellName={null}
        reason="set the grid zoom to derive a coverage lattice for fruit"
        settled
      />,
    );
    expect(
      screen.getByText("set the grid zoom to derive a coverage lattice for fruit"),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Grid zoom for fruit")).toBeInTheDocument();
  });

  it("shows no reason line while the grid fetch has not settled yet", () => {
    render(
      <CoverageChrome
        {...baseProps()}
        derivation=""
        currentCellName={null}
        reason="set the grid zoom to derive a coverage lattice for fruit"
        settled={false}
      />,
    );
    expect(
      screen.queryByText("set the grid zoom to derive a coverage lattice for fruit"),
    ).not.toBeInTheDocument();
  });

  it("the grid-zoom control posts the entered positive zoom", () => {
    const onSetGridZoom = vi.fn();
    render(<CoverageChrome {...baseProps()} onSetGridZoom={onSetGridZoom} />);
    fireEvent.change(screen.getByLabelText("Grid zoom for fruit"), { target: { value: "1.5" } });
    fireEvent.click(screen.getByRole("button", { name: "Set" }));
    expect(onSetGridZoom).toHaveBeenCalledWith(1.5);
  });

  it("the grid-zoom control refuses a non-positive entry, never posting it", () => {
    const onSetGridZoom = vi.fn();
    render(<CoverageChrome {...baseProps()} onSetGridZoom={onSetGridZoom} />);
    fireEvent.change(screen.getByLabelText("Grid zoom for fruit"), { target: { value: "0" } });
    expect(screen.getByRole("button", { name: "Set" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Set" }));
    expect(onSetGridZoom).not.toHaveBeenCalled();
  });

  it("offers a re-derive control once the current zoom would derive a different lattice", () => {
    const onRederiveLattice = vi.fn();
    render(
      <CoverageChrome
        {...baseProps()}
        freshDerivationDiffers
        onRederiveLattice={onRederiveLattice}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Re-derive lattice" }));
    expect(onRederiveLattice).toHaveBeenCalledTimes(1);
  });

  it("offers no re-derive control when the recorded lattice still matches the current zoom", () => {
    render(<CoverageChrome {...baseProps()} freshDerivationDiffers={false} />);
    expect(screen.queryByRole("button", { name: "Re-derive lattice" })).not.toBeInTheDocument();
  });
});
