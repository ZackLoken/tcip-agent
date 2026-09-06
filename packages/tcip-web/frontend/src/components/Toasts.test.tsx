import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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

describe("Toasts channel replacement", () => {
  it("replaces the standing toast on a channel under a fresh id, restarting its timer", () => {
    vi.useFakeTimers();
    try {
      render(<Toasts />);
      act(() => useStore.getState().pushToast("First refusal", "error", "cut"));
      const firstId = useStore.getState().toasts[0].id;

      act(() => vi.advanceTimersByTime(4000));
      act(() => useStore.getState().pushToast("Second refusal", "error", "cut"));

      const toasts = useStore.getState().toasts;
      expect(toasts).toHaveLength(1);
      expect(toasts[0].message).toBe("Second refusal");
      expect(toasts[0].id).not.toBe(firstId);

      // The replaced toast's own timer restarted: it survives past the first toast's original
      // six-second deadline, dismissing only six seconds after the replacement landed.
      act(() => vi.advanceTimersByTime(4000));
      expect(useStore.getState().toasts).toHaveLength(1);
      act(() => vi.advanceTimersByTime(2001));
      expect(useStore.getState().toasts).toHaveLength(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("carries the count over when an identical message repeats on the same channel", () => {
    render(<Toasts />);
    act(() => useStore.getState().pushToast("Refused", "error", "cut"));
    act(() => useStore.getState().pushToast("Refused", "error", "cut"));
    act(() => useStore.getState().pushToast("Refused", "error", "cut"));

    expect(screen.getAllByText(/Refused/)).toHaveLength(1);
    expect(screen.getByText(/\(×3\)/)).toBeInTheDocument();
  });

  it("leaves an unchannelled push behaving as today: identical text collapses, otherwise appends", () => {
    render(<Toasts />);
    act(() => useStore.getState().pushToast("Cancel failed: network error"));
    act(() => useStore.getState().pushToast("Cancel failed: network error"));
    act(() => useStore.getState().pushToast("Relaunch failed: network error"));

    expect(useStore.getState().toasts).toHaveLength(2);
    expect(screen.getByText(/\(×2\)/)).toBeInTheDocument();
  });
});

describe("Toasts dismiss button naming", () => {
  it("carries each toast's own message so two Dismiss buttons never share one name", () => {
    render(<Toasts />);
    act(() => useStore.getState().pushToast("Cancel failed: network error"));
    act(() => useStore.getState().pushToast("Relaunch failed: network error"));

    expect(
      screen.getByRole("button", { name: "Dismiss: Cancel failed: network error" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Dismiss: Relaunch failed: network error" }),
    ).toBeInTheDocument();
  });

  it("truncates a long message to its first words", () => {
    render(<Toasts />);
    act(() =>
      useStore.getState().pushToast("Opened scratch-project-r2, but it has no dated images yet."),
    );

    expect(
      screen.getByRole("button", { name: "Dismiss: Opened scratch-project-r2, but it has no…" }),
    ).toBeInTheDocument();
  });
});
