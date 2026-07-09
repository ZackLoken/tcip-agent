import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { ChatPopup } from "@/components/ChatPopup";

// The chat socket is a thin WS wrapper; stub it so tests don't open real sockets.
vi.mock("@/api/chat", async () => {
  const actual = await vi.importActual<typeof import("@/api/chat")>("@/api/chat");
  return {
    ...actual,
    chatApi: {
      status: vi.fn(),
      createSession: vi.fn().mockResolvedValue({ session_id: "chat_test" }),
      messages: vi.fn().mockResolvedValue({ messages: [] }),
      send: vi.fn().mockResolvedValue({ status: "accepted" }),
      interrupt: vi.fn().mockResolvedValue({ status: "accepted" }),
      permission: vi.fn().mockResolvedValue({ status: "ok" }),
    },
    ChatSocket: vi.fn().mockImplementation(() => ({ connect: vi.fn(), close: vi.fn() })),
  };
});

import { chatApi } from "@/api/chat";

afterEach(cleanup);
beforeEach(() => {
  vi.mocked(chatApi.status).mockResolvedValue({ available: true });
  vi.mocked(chatApi.createSession).mockResolvedValue({ session_id: "chat_test" });
});

describe("ChatPopup", () => {
  it("renders only the launcher until opened", () => {
    render(<ChatPopup />);
    expect(screen.getByLabelText("Toggle agent chat")).toBeInTheDocument();
    expect(screen.queryByText("Agent chat")).not.toBeInTheDocument();
  });

  it("opens the panel and, when a backend is available, shows a composer", async () => {
    render(<ChatPopup />);
    fireEvent.click(screen.getByLabelText("Toggle agent chat"));
    expect(screen.getByText("Agent chat")).toBeInTheDocument();
    expect(await screen.findByPlaceholderText("Ask the agent…")).toBeInTheDocument();
    await waitFor(() => expect(chatApi.createSession).toHaveBeenCalled());
  });

  it("shows the unconfigured message when no backend is available", async () => {
    vi.mocked(chatApi.status).mockResolvedValue({
      available: false,
      reason: "Claude Code is not available.",
    });
    render(<ChatPopup />);
    fireEvent.click(screen.getByLabelText("Toggle agent chat"));
    expect(await screen.findByText(/Claude Code is not available/)).toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Ask the agent…")).not.toBeInTheDocument();
    // No session is created when the backend is unavailable.
    expect(chatApi.createSession).not.toHaveBeenCalled();
  });

  it("sends a message through the chat API", async () => {
    render(<ChatPopup />);
    fireEvent.click(screen.getByLabelText("Toggle agent chat"));
    const box = await screen.findByPlaceholderText("Ask the agent…");
    await waitFor(() => expect(chatApi.createSession).toHaveBeenCalled());
    fireEvent.change(box, { target: { value: "why did mAP drop?" } });
    fireEvent.click(screen.getByText("Send"));
    await waitFor(() =>
      expect(chatApi.send).toHaveBeenCalledWith("chat_test", "why did mAP drop?", true),
    );
  });
});
