import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { ConfirmDialog } from "@/components/ConfirmDialog";

afterEach(cleanup);

describe("ConfirmDialog", () => {
  it("is a labelled, modal dialog", () => {
    render(
      <ConfirmDialog heading="Remove x" onClose={vi.fn()}>
        {"body"}
      </ConfirmDialog>,
    );
    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAccessibleName("Remove x");
  });

  it("moves focus to the first control on open", () => {
    render(
      <ConfirmDialog heading="Remove x" onClose={vi.fn()}>
        <input aria-label="name" />
        <button type="button">confirm</button>
      </ConfirmDialog>,
    );
    expect(screen.getByLabelText("name")).toHaveFocus();
  });

  it("returns focus to the invoking control on close", () => {
    const opener = document.createElement("button");
    opener.textContent = "opener";
    document.body.appendChild(opener);
    opener.focus();

    const { unmount } = render(
      <ConfirmDialog heading="Remove x" onClose={vi.fn()}>
        <input aria-label="name" />
      </ConfirmDialog>,
    );
    expect(screen.getByLabelText("name")).toHaveFocus();
    unmount();
    expect(opener).toHaveFocus();
    opener.remove();
  });

  it("closes on Escape without confirming", () => {
    const onClose = vi.fn();
    render(
      <ConfirmDialog heading="Remove x" onClose={onClose}>
        <input aria-label="name" />
      </ConfirmDialog>,
    );
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("keeps Tab and Shift+Tab inside the dialog", () => {
    render(
      <ConfirmDialog heading="Remove x" onClose={vi.fn()}>
        <input aria-label="first" />
        <button type="button">last</button>
      </ConfirmDialog>,
    );
    const first = screen.getByLabelText("first");
    const last = screen.getByRole("button", { name: "last" });
    expect(first).toHaveFocus();

    last.focus();
    fireEvent.keyDown(window, { key: "Tab" });
    expect(first).toHaveFocus();

    first.focus();
    fireEvent.keyDown(window, { key: "Tab", shiftKey: true });
    expect(last).toHaveFocus();
  });

  it("does not close on a backdrop click", () => {
    const onClose = vi.fn();
    const { container } = render(
      <ConfirmDialog heading="Remove x" onClose={onClose}>
        <input aria-label="name" />
      </ConfirmDialog>,
    );
    const backdrop = container.firstElementChild as HTMLElement;
    fireEvent.click(backdrop);
    expect(onClose).not.toHaveBeenCalled();
  });
});
