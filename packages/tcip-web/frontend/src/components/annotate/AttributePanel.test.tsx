import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { StructuredRefusalError } from "@/api/http";
import { subjectsApi } from "@/api/subjects";
import { ATTR_TYPES } from "@/api/types.generated";
import { AttributePanel } from "@/components/annotate/AttributePanel";
import type { ReviewItem } from "@/lib/reviewItems";
import { useStore } from "@/store";
import { openTestProject } from "@/test/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  openTestProject({ dataset_root: "C:/data", date: "undated" });
  act(() => useStore.getState().setActiveSubject("bud"));
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// Runs outside any act() wrapper so each fireEvent's own act() flushes before the next query;
// only the final submit (fillAttributeDraft's caller wraps that one) kicks off the async save.
function fillAttributeDraft(name: string, values: string) {
  fireEvent.click(screen.getByText("Add attribute"));
  fireEvent.change(screen.getByLabelText("attribute name"), { target: { value: name } });
  fireEvent.change(screen.getByLabelText("attribute values"), {
    target: { value: values },
  });
}

/** A box of the active subject on the canvas, focused. */
function focusedBox(): ReviewItem {
  act(() =>
    useStore
      .getState()
      .add("boxes", { x1: 10, y1: 10, x2: 50, y2: 50, subject: "bud", attributes: {} }),
  );
  return {
    kind: "annotation",
    shape: "box",
    ref: 0,
    subject: "bud",
    bbox: [10, 10, 50, 50],
    match: null,
    reviewed: false,
    score: null,
    at: [30, 30],
    flags: [],
  };
}

const landed = {
  status: "ok" as const,
  n_subjects: 1,
  subjects_path: "C:/data/subjects.json",
  version: "v2",
};

describe("AttributePanel attribute kinds", () => {
  it("offers exactly the kinds the backend's registry declares", () => {
    render(<AttributePanel focused={null} />);
    fireEvent.click(screen.getByText("Add attribute"));
    const options = screen.getByLabelText("attribute type").querySelectorAll("option");
    expect(Array.from(options, (o) => o.value)).toEqual([...ATTR_TYPES]);
    expect(screen.getByRole("combobox", { name: "attribute type" })).toBeInTheDocument();
  });
});

describe("AttributePanel presence", () => {
  it("is absent with no active subject, no focused shape and no image-level rating", () => {
    act(() => useStore.getState().setActiveSubject(null));
    render(<AttributePanel focused={null} />);
    expect(screen.queryByTestId("subject-attributes")).toBeNull();
  });

  it("shows the active subject's attribute block and the focused shape's subject", () => {
    act(() =>
      useStore.getState().setRegistry({
        bud: { attributes: { ripeness: { type: "ordinal", values: ["green", "ripe"] } } },
      }),
    );
    render(<AttributePanel focused={focusedBox()} />);
    expect(screen.getByText("bud attributes")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add attribute" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "ripeness" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add a ripeness value" })).toBeInTheDocument();
  });
});

describe("AttributePanel authoring", () => {
  it("posts the grown registry with the loaded version, installs it and counts the unset shapes", async () => {
    act(() => useStore.getState().setRegistry({ bud: {} }));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue(landed);
    render(<AttributePanel focused={focusedBox()} />);
    fillAttributeDraft("size", "small\nlarge");
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("Add"))));

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0]).toEqual({
      bud: { attributes: { size: { type: "categorical", values: ["small", "large"] } } },
    });
    expect(saveSpy.mock.calls[0][2]).toBeNull();
    expect(useStore.getState().registry.version).toBe("v2");
    expect(screen.getByText("size: 1 of 1 bud shapes unset")).toBeInTheDocument();
  });

  it("refuses a name the subject already declares, inline, and posts nothing", async () => {
    act(() =>
      useStore
        .getState()
        .setRegistry(
          { bud: { attributes: { size: { type: "categorical", values: ["small"] } } } },
          "v1",
        ),
    );
    const saveSpy = vi.spyOn(subjectsApi, "save");
    render(<AttributePanel focused={null} />);
    fillAttributeDraft("size", "large");
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("Add"))));

    expect(screen.getByText("size is already declared.")).toBeInTheDocument();
    expect(saveSpy).not.toHaveBeenCalled();
  });
});

describe("AttributePanel registry-growing on a lost audit line", () => {
  it("adopts the committed registry and toasts the gap message on the audit-gap 409", async () => {
    const committed = {
      status: "ok",
      n_subjects: 1,
      subjects_path: "C:/data/subjects.json",
      version: "v2",
    };
    const message = "replace_registry completed and its audit entry could not be written";
    vi.spyOn(subjectsApi, "save").mockRejectedValue(
      new StructuredRefusalError(
        { error: "audit_entry_not_written", message, committed },
        409,
        message,
      ),
    );

    render(<AttributePanel focused={null} />);
    fillAttributeDraft("opening", "closed\nopen");
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("Add"))));

    // The committed registry is adopted as though the save had answered 200, and the gap
    // itself is surfaced as a toast rather than swallowed.
    expect(useStore.getState().registry.version).toBe("v2");
    expect(useStore.getState().toasts.some((t) => t.message === message)).toBe(true);
  });

  it("reloads the registry from the server on an ordinary refusal, unlike the audit-gap case", async () => {
    vi.spyOn(subjectsApi, "save").mockRejectedValue(new Error("409 stale version"));
    vi.spyOn(subjectsApi, "load").mockResolvedValue({
      subjects: { bud: {} },
      discovered: [],
      version: "v3",
      unreadable: [],
    });

    render(<AttributePanel focused={null} />);
    fillAttributeDraft("opening", "closed\nopen");
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("Add"))));

    expect(useStore.getState().registry.subjects).toEqual({ bud: {} });
    expect(useStore.getState().registry.version).toBe("v3");
  });
});
