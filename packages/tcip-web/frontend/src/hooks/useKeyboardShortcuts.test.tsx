import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render } from "@testing-library/react";

import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";

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
