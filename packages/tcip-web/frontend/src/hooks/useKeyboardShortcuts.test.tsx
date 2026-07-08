import { describe, expect, it, vi } from "vitest";
import { render } from "@testing-library/react";

import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";

function Harness({ onKey }: { onKey: () => void }) {
  useKeyboardShortcuts([{ keys: "a", action: onKey }]);
  return (
    <div>
      <select data-testid="sel">
        <option>x</option>
      </select>
    </div>
  );
}

describe("useKeyboardShortcuts", () => {
  it("fires on a plain keypress but ignores keys aimed at a focused SELECT", () => {
    const onKey = vi.fn();
    const { getByTestId } = render(<Harness onKey={onKey} />);

    // Plain key (target = body) → shortcut fires.
    document.body.dispatchEvent(new KeyboardEvent("keydown", { key: "a", bubbles: true }));
    expect(onKey).toHaveBeenCalledTimes(1);

    // Same key while a <select> is the target → ignored (regression: arrow keys /
    // digits on an open dropdown must change the dropdown, not step images).
    getByTestId("sel").dispatchEvent(new KeyboardEvent("keydown", { key: "a", bubbles: true }));
    expect(onKey).toHaveBeenCalledTimes(1);
  });
});
