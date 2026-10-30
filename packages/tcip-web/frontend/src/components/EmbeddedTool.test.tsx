import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EmbeddedTool } from "@/components/EmbeddedTool";
import { TENSORBOARD_SERVINGS } from "@/test/tensorboard";

afterEach(cleanup);

const [{ url: BOARD }] = Object.values(TENSORBOARD_SERVINGS);

/** A failure sentence of this test's own: the component shows whatever its caller composed. */
const FAILURE = "the failure this test hands the tool";

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
    render(<EmbeddedTool title="TensorBoard" url={null} error={FAILURE} onRetry={onRetry} />);
    expect(screen.getByText(FAILURE)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /try again/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("frames the tool and links out to it full-screen", () => {
    render(<EmbeddedTool title="TensorBoard" url={BOARD} />);
    const frame = screen.getByTitle("TensorBoard");
    expect(frame).toHaveAttribute("src", BOARD);
    const link = screen.getByRole("link", { name: /open in a new tab/i });
    expect(link).toHaveAttribute("href", BOARD);
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("names Try again and Open in a new tab with the tool's own title", () => {
    const onRetry = vi.fn();
    const { unmount } = render(
      <EmbeddedTool title="Sweep TensorBoard" url={null} error={FAILURE} onRetry={onRetry} />,
    );
    expect(
      screen.getByRole("button", { name: "Try again: Sweep TensorBoard" }),
    ).toBeInTheDocument();
    unmount();

    render(<EmbeddedTool title="Sweep TensorBoard" url={BOARD} />);
    expect(
      screen.getByRole("link", { name: "Open in a new tab: Sweep TensorBoard" }),
    ).toBeInTheDocument();
  });

  it("carries its own title as a heading, not a plain span", () => {
    render(<EmbeddedTool title="TensorBoard" url={BOARD} />);
    expect(screen.getByRole("heading", { level: 2, name: "TensorBoard" })).toBeInTheDocument();
  });

  it("puts a failure sentence in a polite status region so a keyboard user hears it", () => {
    render(<EmbeddedTool title="TensorBoard" url={null} error={FAILURE} />);
    expect(screen.getByRole("status")).toHaveTextContent(FAILURE);
  });
});
