import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly:
// a leftover overlay from one test would leak into the next.

import { HelpOverlay } from "@/components/HelpOverlay";
import { ANNOTATE_KEYS } from "@/lib/annotateKeys";

afterEach(cleanup);

describe("HelpOverlay", () => {
  it("opens with '?' and closes with Escape", () => {
    render(<HelpOverlay activeTab="annotate" />);
    expect(screen.queryByText(/^Shortcuts$/)).not.toBeInTheDocument();

    fireEvent.keyDown(document.body, { key: "?" });
    expect(screen.getByText(/^Shortcuts$/)).toBeInTheDocument();

    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(screen.queryByText(/^Shortcuts$/)).not.toBeInTheDocument();
  });

  it("is a labeled, modal dialog that takes focus on open and returns it on close", async () => {
    const opener = document.createElement("div");
    opener.tabIndex = 0;
    document.body.appendChild(opener);
    render(<HelpOverlay activeTab="annotate" />);
    opener.focus();

    fireEvent.keyDown(opener, { key: "?" });
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAccessibleName("Shortcuts");
    expect(dialog).toContainElement(document.activeElement as HTMLElement);

    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    await waitFor(() => expect(opener).toHaveFocus());
    opener.remove();
  });

  it("closes on a click outside the panel", async () => {
    render(<HelpOverlay activeTab="annotate" />);
    fireEvent.keyDown(document.body, { key: "?" });

    const backdrop = screen.getByRole("dialog").previousElementSibling as HTMLElement;
    await userEvent.pointer({ keys: "[MouseLeft]", target: backdrop });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("ignores '?' typed into a focused form control", () => {
    render(
      <div>
        <input data-testid="field" />
        <HelpOverlay activeTab="annotate" />
      </div>,
    );
    // "?" aimed at a text field must insert the character, not open the overlay.
    fireEvent.keyDown(screen.getByTestId("field"), { key: "?" });
    expect(screen.queryByText(/^Shortcuts$/)).not.toBeInTheDocument();

    // Same key with no form control focused still opens it.
    fireEvent.keyDown(document.body, { key: "?" });
    expect(screen.getByText(/^Shortcuts$/)).toBeInTheDocument();
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
