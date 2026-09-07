import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";

import { classesApi } from "@/api/classes";
import { StructuredRefusalError } from "@/api/http";
import { AttributePanel } from "@/components/annotate/AttributePanel";
import { useStore } from "@/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.setState((s) => ({
    gui: {
      ...s.gui,
      dataset: { ...s.gui.dataset, project_root: "C:/proj", dataset_root: "C:/data" },
    },
  }));
  act(() => useStore.getState().setActiveSubject("bud"));
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

// Runs outside any act() wrapper so each fireEvent's own act() flushes before the next query;
// only the final submit (fillAttributeDraft's caller wraps that one) kicks off the async save.
function fillAttributeDraft(name: string, values: string) {
  fireEvent.click(screen.getByText("+ Attribute"));
  fireEvent.change(screen.getByPlaceholderText("attribute name"), { target: { value: name } });
  fireEvent.change(screen.getByPlaceholderText("one value per line"), {
    target: { value: values },
  });
}

describe("AttributePanel registry-growing on a lost audit line", () => {
  it("adopts the committed registry and toasts the gap message on the audit-gap 409", async () => {
    const committed = {
      status: "ok",
      n_subjects: 1,
      classes_path: "C:/data/classes.json",
      version: "v2",
      schema_change_sweep: { newly_stamped: {}, predating_vocabulary: {}, warning: null },
    };
    const message = "gui_save_classes completed and its audit entry could not be written";
    vi.spyOn(classesApi, "save").mockRejectedValue(
      new StructuredRefusalError(
        { error: "audit_entry_not_written", message, committed },
        409,
        message,
      ),
    );

    render(<AttributePanel selectedBoxIdx={null} locked={false} />);
    fillAttributeDraft("opening", "closed\nopen");
    await act(async () => {
      fireEvent.click(screen.getByText("Add"));
    });

    // The committed registry is adopted as though the save had answered 200, and the gap
    // itself is surfaced as a toast rather than swallowed.
    expect(useStore.getState().registry.version).toBe("v2");
    expect(useStore.getState().toasts.some((t) => t.message === message)).toBe(true);
  });

  it("reloads the registry from the server on an ordinary refusal, unlike the audit-gap case", async () => {
    vi.spyOn(classesApi, "save").mockRejectedValue(new Error("409 stale version"));
    vi.spyOn(classesApi, "load").mockResolvedValue({
      subjects: { bud: {} },
      version: "v3",
      unreadable: [],
    });

    render(<AttributePanel selectedBoxIdx={null} locked={false} />);
    fillAttributeDraft("opening", "closed\nopen");
    await act(async () => {
      fireEvent.click(screen.getByText("Add"));
    });

    expect(useStore.getState().registry.subjects).toEqual({ bud: {} });
    expect(useStore.getState().registry.version).toBe("v3");
  });
});
