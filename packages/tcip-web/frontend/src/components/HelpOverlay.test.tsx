import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly:
// a leftover overlay from one test would leak into the next.

import { HelpOverlay } from "@/components/HelpOverlay";
import { ANNOTATE_KEYS } from "@/lib/annotateKeys";

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

  it("lists every Annotate binding from the one declaration the tab binds", () => {
    render(<HelpOverlay activeTab="annotate" />);
    fireEvent.keyDown(document.body, { key: "?" });

    for (const binding of Object.values(ANNOTATE_KEYS)) {
      expect(screen.getByText(binding.desc)).toBeInTheDocument();
    }
    for (const key of ["0–9", "Double-click", "Right-click"]) {
      expect(screen.getByText(key)).toBeInTheDocument();
    }
  });
});
