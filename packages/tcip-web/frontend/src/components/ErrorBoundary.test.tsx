import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";

import { ErrorBoundary } from "@/components/ErrorBoundary";

const Boom = () => {
  throw new Error("boom");
};

describe("ErrorBoundary", () => {
  it("renders a fallback when a child throws, and recovers when resetKey changes", () => {
    // React logs the caught render error to console.error; silence it for the test.
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const { rerender } = render(
      <ErrorBoundary resetKey="a">
        <Boom />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();

    rerender(
      <ErrorBoundary resetKey="b">
        <div>healthy</div>
      </ErrorBoundary>,
    );
    expect(screen.getByText("healthy")).toBeInTheDocument();
    spy.mockRestore();
  });
});
