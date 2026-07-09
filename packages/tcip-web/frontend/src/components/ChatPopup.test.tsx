import { afterEach, describe, expect, it } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { CHAT_POPUP_ENABLED, ChatPopup } from "@/components/ChatPopup";

afterEach(cleanup);

describe("ChatPopup", () => {
  it("is disabled by default and renders nothing", () => {
    // Phase C0 contract: the flag stays off until the chat backend exists,
    // and the mounted component must have zero DOM footprint.
    expect(CHAT_POPUP_ENABLED).toBe(false);
    const { container } = render(<ChatPopup />);
    expect(container).toBeEmptyDOMElement();
  });

  it("when enabled, shows a launcher that toggles the stubbed panel", () => {
    render(<ChatPopup enabled />);
    expect(screen.queryByText(/agent chat backend is not implemented/i)).not.toBeInTheDocument();

    fireEvent.click(screen.getByLabelText("Toggle agent chat"));
    expect(screen.getByText(/agent chat backend is not implemented/i)).toBeInTheDocument();
    // Composer is a dead placeholder — no backend to talk to.
    expect(screen.getByPlaceholderText("Agent backend not configured")).toBeDisabled();

    fireEvent.click(screen.getByLabelText("Close agent chat"));
    expect(screen.queryByText(/agent chat backend is not implemented/i)).not.toBeInTheDocument();
  });
});
