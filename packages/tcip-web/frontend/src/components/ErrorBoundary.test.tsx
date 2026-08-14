import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";

import { ErrorBoundary } from "@/components/ErrorBoundary";

const Boom = () => {
  throw new Error("boom");
};

afterEach(cleanup);

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

  it("reports the caught error and the crashing subtree's stack, not only the fallback", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    render(
      <ErrorBoundary resetKey="a">
        <Boom />
      </ErrorBoundary>,
    );
    const reported = spy.mock.calls.filter((c) => c[0] === "ErrorBoundary caught:");
    expect(reported).toHaveLength(1);
    expect((reported[0][1] as Error).message).toBe("boom");
    expect(String(reported[0][2])).toContain("Boom");
    spy.mockRestore();
  });

  it("keeps the fallback up when a re-render leaves resetKey unchanged", () => {
    const spy = vi.spyOn(console, "error").mockImplementation(() => {});
    const { rerender } = render(
      <ErrorBoundary resetKey="a">
        <Boom />
      </ErrorBoundary>,
    );
    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();

    rerender(
      <ErrorBoundary resetKey="a">
        <div>healthy</div>
      </ErrorBoundary>,
    );
    expect(screen.queryByText("healthy")).not.toBeInTheDocument();
    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
    spy.mockRestore();
  });
});
