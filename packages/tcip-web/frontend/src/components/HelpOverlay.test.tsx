import { afterEach, describe, expect, it } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly —
// a leftover overlay from one test would leak into the next.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { HelpOverlay } from "@/components/HelpOverlay";

afterEach(cleanup);

describe("HelpOverlay", () => {
  it("toggles with '?' and closes with Escape", () => {
    render(<HelpOverlay activeTab="annotate" />);
    expect(screen.queryByText(/keyboard & mouse reference/i)).not.toBeInTheDocument();

    fireEvent.keyDown(document.body, { key: "?" });
    expect(screen.getByText(/keyboard & mouse reference/i)).toBeInTheDocument();

    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(screen.queryByText(/keyboard & mouse reference/i)).not.toBeInTheDocument();
  });

  it("ignores '?' typed into a focused form control", () => {
    render(
      <div>
        <input data-testid="field" />
        <HelpOverlay activeTab="annotate" />
      </div>,
    );
    // "?" aimed at a text field must insert the character, not toggle the overlay.
    fireEvent.keyDown(screen.getByTestId("field"), { key: "?" });
    expect(screen.queryByText(/keyboard & mouse reference/i)).not.toBeInTheDocument();

    // Same key with no form control focused still toggles.
    fireEvent.keyDown(document.body, { key: "?" });
    expect(screen.getByText(/keyboard & mouse reference/i)).toBeInTheDocument();
  });

  it("documents Review verdicts as retraining-only (no GT rewrite)", () => {
    render(<HelpOverlay activeTab="review" />);
    fireEvent.keyDown(document.body, { key: "?" });

    // Accept/Reject must state they don't change GT (matches ReviewTab semantics);
    // the old "add-to-GT" / "delete GT" wording must not come back.
    expect(screen.getAllByText(/does not change GT/)).toHaveLength(2);
    expect(screen.getByText(/the only action that changes GT files/)).toBeInTheDocument();
    expect(screen.queryByText(/add-to-GT/)).not.toBeInTheDocument();
    expect(screen.queryByText(/delete GT/)).not.toBeInTheDocument();
  });

  it("lists the Annotate shortcuts actually registered in AnnotateTab", () => {
    render(<HelpOverlay activeTab="annotate" />);
    fireEvent.keyDown(document.body, { key: "?" });

    for (const key of ["Ctrl+Y", "v", "s", "0–9", "Double-click", "Right-click"]) {
      expect(screen.getByText(key)).toBeInTheDocument();
    }
  });
});
