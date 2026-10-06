import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { Stepper } from "@/components/Stepper";

afterEach(cleanup);

function renderStepper(over: Partial<Parameters<typeof Stepper>[0]> = {}) {
  const onStep = vi.fn();
  const onJump = vi.fn();
  render(
    <Stepper
      label="Item"
      noun="item"
      name="fruit box"
      position={2}
      total={5}
      canPrev
      canNext
      onStep={onStep}
      onJump={onJump}
      {...over}
    />,
  );
  return { onStep, onJump };
}

describe("Stepper", () => {
  it("names its controls by the noun and shows the position over the total", () => {
    renderStepper();
    expect(screen.getByRole("button", { name: "Previous item" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Next item" })).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Item position" })).toHaveValue("2");
    expect(screen.getByText("/ 5")).toBeInTheDocument();
    expect(screen.getByText("fruit box")).toBeInTheDocument();
  });

  it("steps by one either way and jumps to a typed position on Enter", () => {
    const { onStep, onJump } = renderStepper();
    fireEvent.click(screen.getByRole("button", { name: "Next item" }));
    fireEvent.click(screen.getByRole("button", { name: "Previous item" }));
    expect(onStep.mock.calls).toEqual([[1], [-1]]);

    const input = screen.getByRole("textbox", { name: "Item position" });
    fireEvent.focus(input);
    fireEvent.change(input, { target: { value: "4" } });
    fireEvent.keyDown(input, { key: "Enter" });
    expect(onJump).toHaveBeenCalledWith(4);
  });

  it("shows an empty position and the unset glyph with nothing current", () => {
    renderStepper({ name: null, position: 0, total: 0, canPrev: false, canNext: false });
    expect(screen.getByRole("textbox", { name: "Item position" })).toHaveValue("");
    expect(screen.getByRole("button", { name: "Next item" })).toBeDisabled();
  });
});
