import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render } from "@testing-library/react";

import { useKeyboardShortcuts, type Shortcut } from "@/hooks/useKeyboardShortcuts";

afterEach(() => {
  cleanup();
});

function Harness({ onKey, onEnter }: { onKey: () => void; onEnter: () => void }) {
  useKeyboardShortcuts([
    { keys: "a", action: onKey },
    { keys: "enter", action: onEnter },
  ]);
  return (
    <div>
      <select data-testid="sel">
        <option>x</option>
      </select>
      <button data-testid="btn">Go</button>
    </div>
  );
}

describe("useKeyboardShortcuts", () => {
  it("fires on a plain keypress but ignores keys aimed at a focused SELECT", () => {
    const onKey = vi.fn();
    const { getByTestId } = render(<Harness onKey={onKey} onEnter={vi.fn()} />);

    // Plain key (target = body) → shortcut fires.
    document.body.dispatchEvent(new KeyboardEvent("keydown", { key: "a", bubbles: true }));
    expect(onKey).toHaveBeenCalledTimes(1);

    // Same key while a <select> is the target → ignored (regression: arrow keys /
    // digits on an open dropdown must change the dropdown, not step images).
    getByTestId("sel").dispatchEvent(new KeyboardEvent("keydown", { key: "a", bubbles: true }));
    expect(onKey).toHaveBeenCalledTimes(1);
  });

  it("leaves Enter on a focused button unprevented and unfired", () => {
    const onEnter = vi.fn();
    const { getByTestId } = render(<Harness onKey={vi.fn()} onEnter={onEnter} />);

    const btn = getByTestId("btn");
    const event = new KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });
    btn.dispatchEvent(event);

    expect(onEnter).not.toHaveBeenCalled();
    expect(event.defaultPrevented).toBe(false);
  });
});

function PassthroughHarness({ shortcuts }: { shortcuts: Shortcut[] }) {
  useKeyboardShortcuts(shortcuts);
  return (
    <div>
      <button data-testid="plain-button">Plain</button>
      <button data-testid="passthrough-button" data-keyboard-passthrough="">
        Passthrough
      </button>
      <select data-testid="a-select">
        <option value="">x</option>
      </select>
      <input data-testid="an-input" />
    </div>
  );
}

describe("useKeyboardShortcuts whileFocused exemption", () => {
  it("fires a whileFocused shortcut when focus sits on the one control marked data-keyboard-passthrough", () => {
    const action = vi.fn();
    const { getByTestId } = render(
      <PassthroughHarness shortcuts={[{ keys: "shift+h", action, whileFocused: true }]} />,
    );

    fireEvent.keyDown(getByTestId("passthrough-button"), { key: "H", shiftKey: true });
    expect(action).toHaveBeenCalledTimes(1);
  });

  it("still swallows the same shortcut on an ordinary button with no passthrough attribute", () => {
    const action = vi.fn();
    const { getByTestId } = render(
      <PassthroughHarness shortcuts={[{ keys: "shift+h", action, whileFocused: true }]} />,
    );

    fireEvent.keyDown(getByTestId("plain-button"), { key: "H", shiftKey: true });
    expect(action).not.toHaveBeenCalled();
  });

  it("never exempts a select, even with whileFocused set", () => {
    const action = vi.fn();
    const { getByTestId } = render(
      <PassthroughHarness shortcuts={[{ keys: "shift+h", action, whileFocused: true }]} />,
    );

    fireEvent.keyDown(getByTestId("a-select"), { key: "H", shiftKey: true });
    expect(action).not.toHaveBeenCalled();
  });

  it("never exempts a text input, even with whileFocused set", () => {
    const action = vi.fn();
    const { getByTestId } = render(
      <PassthroughHarness shortcuts={[{ keys: "shift+h", action, whileFocused: true }]} />,
    );

    fireEvent.keyDown(getByTestId("an-input"), { key: "H", shiftKey: true });
    expect(action).not.toHaveBeenCalled();
  });

  it("fires normally on the window for a shortcut with no whileFocused flag", () => {
    const action = vi.fn();
    render(<PassthroughHarness shortcuts={[{ keys: "a", action }]} />);

    fireEvent.keyDown(window, { key: "a" });
    expect(action).toHaveBeenCalledTimes(1);
  });
});
