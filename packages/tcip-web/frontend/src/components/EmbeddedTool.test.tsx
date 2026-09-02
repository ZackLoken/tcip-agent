import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";

import { EmbeddedTool } from "@/components/EmbeddedTool";

afterEach(cleanup);

describe("EmbeddedTool", () => {
  it("renders nothing until the caller has something to show", () => {
    const { container } = render(<EmbeddedTool title="TensorBoard" url={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows a launch in flight", () => {
    render(<EmbeddedTool title="TensorBoard" url={null} loading />);
    expect(screen.getByText("Starting…")).toBeInTheDocument();
    expect(screen.queryByTitle("TensorBoard")).not.toBeInTheDocument();
  });

  it("offers a retry on a failure", () => {
    const onRetry = vi.fn();
    render(
      <EmbeddedTool title="Ray dashboard" url={null} error="No cluster is up." onRetry={onRetry} />,
    );
    expect(screen.getByText("No cluster is up.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /try again/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("frames the tool and links out to it full-screen", () => {
    render(<EmbeddedTool title="TensorBoard" url="http://localhost:6006" />);
    const frame = screen.getByTitle("TensorBoard");
    expect(frame).toHaveAttribute("src", "http://localhost:6006");
    const link = screen.getByRole("link", { name: /open in a new tab/i });
    expect(link).toHaveAttribute("href", "http://localhost:6006");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("names Try again and Open in a new tab with the tool's own title", () => {
    const onRetry = vi.fn();
    const { unmount } = render(
      <EmbeddedTool
        title="Sweep TensorBoard"
        url={null}
        error="sweep not found"
        onRetry={onRetry}
      />,
    );
    expect(
      screen.getByRole("button", { name: "Try again: Sweep TensorBoard" }),
    ).toBeInTheDocument();
    unmount();

    render(<EmbeddedTool title="Ray dashboard" url="http://localhost:8265" />);
    expect(
      screen.getByRole("link", { name: "Open in a new tab: Ray dashboard" }),
    ).toBeInTheDocument();
  });

  it("carries its own title as a heading, not a plain span", () => {
    render(<EmbeddedTool title="TensorBoard" url="http://localhost:6006" />);
    expect(screen.getByRole("heading", { level: 2, name: "TensorBoard" })).toBeInTheDocument();
  });

  it("puts a failure sentence in a polite status region so a keyboard user hears it", () => {
    render(<EmbeddedTool title="TensorBoard" url={null} error="This run produced no logs." />);
    expect(screen.getByRole("status")).toHaveTextContent("This run produced no logs.");
  });
});
