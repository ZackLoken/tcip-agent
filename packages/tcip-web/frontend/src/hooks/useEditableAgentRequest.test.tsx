import { afterEach, describe, expect, it } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";

import { useEditableAgentRequest } from "@/hooks/useEditableAgentRequest";

function Harness({ defaultRequest }: { defaultRequest: string }) {
  const { request, setRequest } = useEditableAgentRequest(defaultRequest);
  return (
    <div>
      <span data-testid="request">{request}</span>
      <button onClick={() => setRequest("edited by hand")}>edit</button>
    </div>
  );
}

afterEach(cleanup);

describe("useEditableAgentRequest", () => {
  it("follows the default while the text is untouched", () => {
    const { rerender } = render(<Harness defaultRequest="first" />);
    expect(screen.getByTestId("request").textContent).toBe("first");
    rerender(<Harness defaultRequest="second" />);
    expect(screen.getByTestId("request").textContent).toBe("second");
  });

  it("keeps an edited request when the selection resolves afterwards", () => {
    const { rerender } = render(<Harness defaultRequest="first" />);
    act(() => screen.getByText("edit").click());
    expect(screen.getByTestId("request").textContent).toBe("edited by hand");
    rerender(<Harness defaultRequest="second" />);
    expect(screen.getByTestId("request").textContent).toBe("edited by hand");
  });
});
