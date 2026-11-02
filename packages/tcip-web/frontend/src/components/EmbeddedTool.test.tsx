import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { EmbeddedTool } from "@/components/EmbeddedTool";
import { PENDING } from "@/hooks/useEmbeddedToolRetry";
import { TENSORBOARD_FAILURES, TENSORBOARD_SERVINGS } from "@/test/tensorboard";

afterEach(cleanup);

const [BOARD] = TENSORBOARD_SERVINGS;
const [FAILURE, DYING] = TENSORBOARD_FAILURES;

describe("EmbeddedTool", () => {
  it("shows a launch in flight", () => {
    render(<EmbeddedTool title="TensorBoard" outcome={PENDING} onRetry={vi.fn()} />);
    expect(screen.getByText("Starting…")).toBeInTheDocument();
    expect(screen.queryByTitle("TensorBoard")).not.toBeInTheDocument();
  });

  it("offers a retry on a failure", () => {
    const onRetry = vi.fn();
    render(<EmbeddedTool title="TensorBoard" outcome={FAILURE} onRetry={onRetry} />);
    expect(screen.getByText(FAILURE.error)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /try again/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });

  it("shows a failed launch's child output after its reason", () => {
    render(<EmbeddedTool title="TensorBoard" outcome={DYING} onRetry={vi.fn()} />);
    expect(screen.getByRole("status")).toHaveTextContent(`${DYING.error}: ${DYING.output}`);
  });

  it("says a request that reached no answer could not reach the tool, and offers a retry", () => {
    render(
      <EmbeddedTool
        title="TensorBoard"
        outcome={{ kind: "unreached", message: "connection refused" }}
        onRetry={vi.fn()}
      />,
    );
    expect(screen.getByRole("status")).toHaveTextContent(
      "TensorBoard could not be reached: connection refused",
    );
    expect(screen.getByRole("button", { name: "Try again: TensorBoard" })).toBeInTheDocument();
  });

  it("frames the tool and links out to it full-screen", () => {
    render(<EmbeddedTool title="TensorBoard" outcome={BOARD} onRetry={vi.fn()} />);
    const frame = screen.getByTitle("TensorBoard");
    expect(frame).toHaveAttribute("src", BOARD.url);
    const link = screen.getByRole("link", { name: /open in a new tab/i });
    expect(link).toHaveAttribute("href", BOARD.url);
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("names Try again and Open in a new tab with the tool's own title", () => {
    const { unmount } = render(
      <EmbeddedTool title="Sweep TensorBoard" outcome={FAILURE} onRetry={vi.fn()} />,
    );
    expect(
      screen.getByRole("button", { name: "Try again: Sweep TensorBoard" }),
    ).toBeInTheDocument();
    unmount();

    render(<EmbeddedTool title="Sweep TensorBoard" outcome={BOARD} onRetry={vi.fn()} />);
    expect(
      screen.getByRole("link", { name: "Open in a new tab: Sweep TensorBoard" }),
    ).toBeInTheDocument();
  });

  it("carries its own title as a heading, not a plain span", () => {
    render(<EmbeddedTool title="TensorBoard" outcome={BOARD} onRetry={vi.fn()} />);
    expect(screen.getByRole("heading", { level: 2, name: "TensorBoard" })).toBeInTheDocument();
  });

  it("puts a failure sentence in a polite status region so a keyboard user hears it", () => {
    render(<EmbeddedTool title="TensorBoard" outcome={FAILURE} onRetry={vi.fn()} />);
    expect(screen.getByRole("status")).toHaveTextContent(FAILURE.error);
  });
});
