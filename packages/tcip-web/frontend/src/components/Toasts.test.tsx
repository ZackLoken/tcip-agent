import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";

import { Toasts } from "@/components/Toasts";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
});

afterEach(cleanup);

describe("Toasts repeats", () => {
  it("collapses an identical repeated message into one toast with a count", () => {
    render(<Toasts />);
    act(() => useStore.getState().pushToast("Cancel failed: network error"));
    act(() => useStore.getState().pushToast("Cancel failed: network error"));
    act(() => useStore.getState().pushToast("Cancel failed: network error"));

    expect(screen.getAllByText(/Cancel failed: network error/)).toHaveLength(1);
    expect(screen.getByText(/\(×3\)/)).toBeInTheDocument();
  });

  it("keeps a different message, or a different level, as its own toast", () => {
    render(<Toasts />);
    act(() => useStore.getState().pushToast("Cancel failed: network error"));
    act(() => useStore.getState().pushToast("Cancel failed: network error", "info"));
    act(() => useStore.getState().pushToast("Relaunch failed: network error"));

    expect(screen.getAllByRole("status")).toHaveLength(1);
    expect(screen.getAllByRole("alert")).toHaveLength(2);
  });
});

describe("Toasts live region politeness", () => {
  it("marks an error toast assertive and a non-error toast polite", () => {
    render(<Toasts />);
    act(() => useStore.getState().pushToast("something failed", "error"));
    act(() => useStore.getState().pushToast("opened the project", "info"));

    const alert = screen.getByRole("alert");
    expect(alert).toHaveAttribute("aria-live", "assertive");
    const status = screen.getByRole("status");
    expect(status).toHaveAttribute("aria-live", "polite");
  });
});
