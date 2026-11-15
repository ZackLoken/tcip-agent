import { cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { ModalDialog } from "@/components/ModalDialog";
import { keydownInModal } from "@/lib/modalKeydown";

afterEach(cleanup);

/** The answer `keydownInModal` gives each keydown that reaches a window listener. */
function windowAnswers(): { answers: boolean[]; stop: () => void } {
  const answers: boolean[] = [];
  const listener = (e: KeyboardEvent) => answers.push(keydownInModal(e));
  window.addEventListener("keydown", listener);
  return { answers, stop: () => window.removeEventListener("keydown", listener) };
}

function renderModal() {
  return render(
    <ModalDialog title="Remove x" onClose={() => {}} closeOnOutsideClick={false} className="w-72">
      <input aria-label="name" />
    </ModalDialog>,
  );
}

describe("keydownInModal", () => {
  it("is true for a keydown while a modal is open and false once it is closed", () => {
    const { answers, stop } = windowAnswers();
    const { unmount } = renderModal();
    fireEvent.keyDown(document.body, { key: "a" });
    unmount();
    fireEvent.keyDown(document.body, { key: "a" });
    stop();
    expect(answers).toEqual([true, false]);
  });

  it("stays true for the keydown that closes the modal, which Escape does before the window's listeners run", () => {
    const { unmount } = renderModal();
    document.addEventListener("keydown", () => unmount(), { capture: true, once: true });
    const { answers, stop } = windowAnswers();
    fireEvent.keyDown(document.body, { key: "Escape" });
    stop();
    expect(answers).toEqual([true]);
  });

  it("marks for each of two open modals alone, so closing one leaves the other marking", () => {
    const first = renderModal();
    const second = renderModal();
    const { answers, stop } = windowAnswers();
    first.unmount();
    fireEvent.keyDown(document.body, { key: "a" });
    second.unmount();
    fireEvent.keyDown(document.body, { key: "a" });
    stop();
    expect(answers).toEqual([true, false]);
  });
});
