import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ColorPickerModal } from "@/components/ColorPickerModal";

afterEach(cleanup);

function renderPicker(onCancel = vi.fn()) {
  return render(
    <ColorPickerModal
      title="Lake color"
      initialColor="#507754"
      onSubmit={vi.fn()}
      onCancel={onCancel}
    />,
  );
}

describe("ColorPickerModal", () => {
  it("moves focus into the hex input, its text selected, and returns it to the opener on close", async () => {
    const opener = document.createElement("button");
    document.body.appendChild(opener);
    opener.focus();

    const { unmount } = renderPicker();
    const hex = screen.getByRole("textbox", { name: "hex color" }) as HTMLInputElement;
    expect(hex).toHaveFocus();
    expect(hex.selectionStart).toBe(0);
    expect(hex.selectionEnd).toBe("#507754".length);

    unmount();
    await waitFor(() => expect(opener).toHaveFocus());
    opener.remove();
  });

  it("cancels on Escape", async () => {
    const onCancel = vi.fn();
    renderPicker(onCancel);
    await userEvent.keyboard("{Escape}");
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("cancels on a click outside the panel", async () => {
    const onCancel = vi.fn();
    renderPicker(onCancel);
    const backdrop = screen.getByRole("dialog").previousElementSibling as HTMLElement;
    await userEvent.pointer({ keys: "[MouseLeft]", target: backdrop });
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("keeps Tab inside the panel", async () => {
    renderPicker();
    const dialog = screen.getByRole("dialog");
    for (let i = 0; i < 40; i++) {
      await userEvent.tab();
      expect(dialog).toContainElement(document.activeElement as HTMLElement);
    }
  });
});
