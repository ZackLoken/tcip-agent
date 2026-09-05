import { afterEach, describe, expect, it } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly:
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

  it("documents Review verdicts as writing ground truth (matches ReviewTab semantics)", () => {
    render(<HelpOverlay activeTab="review" />);
    fireEvent.keyDown(document.body, { key: "?" });

    // Verdicts author GT: accepting an FP adds the prediction, rejecting a TP/FN
    // deletes the object. The old "does not change GT" wording must not come back:
    // it told reviewers a destructive key was safe.
    expect(screen.getByText(/adds the prediction to GT/)).toBeInTheDocument();
    expect(screen.getByText(/deletes the ground-truth object/)).toBeInTheDocument();
    expect(screen.getByText(/Save the edited shape to ground truth/)).toBeInTheDocument();
    expect(screen.queryByText(/does not change GT/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Edit in Annotate tab/)).not.toBeInTheDocument();
  });

  it("lists the Annotate shortcuts actually registered in AnnotateTab", () => {
    render(<HelpOverlay activeTab="annotate" />);
    fireEvent.keyDown(document.body, { key: "?" });

    for (const key of ["Ctrl+Y", "v", "s", "x", "0–9", "Double-click", "Right-click"]) {
      expect(screen.getByText(key)).toBeInTheDocument();
    }
  });
});
