import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { subjectsApi } from "@/api/subjects";
import { ATTR_TYPES } from "@/api/types.generated";
import { AttributePanel } from "@/components/annotate/AttributePanel";
import { NO_MATCHES, reviewItems, type ReviewItem } from "@/lib/reviewItems";
import { useStore } from "@/store";
import { BARE_REGISTRY, LABEL_ANSWERS, SERVED_ATTRIBUTES } from "@/test/labels";
import { thrown } from "@/test/refusals";
import { openTestProject, selected } from "@/test/store";

const initialStoreState = useStore.getState();

/** The subject the registry the load route served declares. */
const [SUBJECT] = Object.keys(LABEL_ANSWERS.registry_loaded.subjects ?? {});

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  openTestProject(selected({ count: 0 }));
  act(() => useStore.getState().setActiveSubject(SUBJECT));
});

afterEach(cleanup);

/** Type `text` into the input labeled `label` and press Enter. */
function enter(label: string, text: string) {
  const input = screen.getByLabelText(label);
  fireEvent.change(input, { target: { value: text } });
  fireEvent.keyDown(input, { key: "Enter" });
}

// Runs outside any act() wrapper so each fireEvent's own act() flushes before the next query;
// only the final submit (fillAttributeDraft's caller wraps that one) kicks off the async save.
function fillAttributeDraft(name: string, values: string[]) {
  fireEvent.click(screen.getByText("Add attribute"));
  fireEvent.change(screen.getByLabelText("attribute name"), { target: { value: name } });
  for (const value of values) enter("attribute value", value);
}

/** A box of the active subject on the canvas, focused. */
function focusedBox(): ReviewItem {
  act(() =>
    useStore
      .getState()
      .add("boxes", { x1: 10, y1: 10, x2: 50, y2: 50, subject: SUBJECT, attributes: {} }),
  );
  const { canvas } = useStore.getState();
  const [item] = reviewItems({
    canvas,
    proposals: [],
    matches: NO_MATCHES,
    reviewing: false,
    flags: canvas.flags,
  });
  return item;
}

const landed = LABEL_ANSWERS.registry_saved;

/** The served subject as declared before any attribute was, and as served with its attribute. */
const BARE = BARE_REGISTRY[SUBJECT];
const DECLARING = (LABEL_ANSWERS.registry_loaded.subjects ?? {})[SUBJECT];

/** The served subject's one declared attribute, its name and its values. */
const [[DECLARED, { values: DECLARED_VALUES }]] = Object.entries(SERVED_ATTRIBUTES);
const ADD_A_VALUE = new RegExp(`^Add an? ${DECLARED} value$`);

describe("AttributePanel attribute kinds", () => {
  it("offers exactly the kinds the backend's registry declares", () => {
    act(() => useStore.getState().setRegistry({ [SUBJECT]: BARE }));
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
    act(() => useStore.getState().setRegistry({ [SUBJECT]: DECLARING }));
    render(<AttributePanel focused={focusedBox()} />);
    expect(screen.getByText(`${SUBJECT} attributes`)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add attribute" })).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: DECLARED })).toBeInTheDocument();
    // One beside the focused shape's select, one in the subject's declared attributes.
    expect(screen.getAllByRole("button", { name: ADD_A_VALUE })).toHaveLength(2);
  });
});

describe("AttributePanel authoring", () => {
  it("posts the grown registry with the loaded version, installs it and counts the unset shapes", async () => {
    act(() => useStore.getState().setRegistry({ [SUBJECT]: BARE }));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue(landed);
    render(<AttributePanel focused={focusedBox()} />);
    fillAttributeDraft("size", ["small", "large"]);
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("Add"))));

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].subjects).toEqual({
      [SUBJECT]: {
        ...BARE,
        attributes: { size: { type: "categorical", values: ["small", "large"] } },
      },
    });
    expect(saveSpy.mock.calls[0][0].version).toBeNull();
    expect(useStore.getState().registry.version).toBe(landed.version);
    expect(screen.getByText(`size: 1 of 1 ${SUBJECT} shapes unset`)).toBeInTheDocument();
  });

  it("declares every value entered, the one still in the box included, as one attribute", async () => {
    act(() => useStore.getState().setRegistry({ [SUBJECT]: BARE }));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue(landed);
    render(<AttributePanel focused={null} />);
    fillAttributeDraft("opening", ["closed", "half"]);
    fireEvent.change(screen.getByLabelText("attribute value"), { target: { value: "open" } });

    expect(screen.getByText("closed")).toBeInTheDocument();
    expect(screen.getByText("half")).toBeInTheDocument();
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("Add"))));

    expect(saveSpy.mock.calls[0][0].subjects).toEqual({
      [SUBJECT]: {
        ...BARE,
        attributes: { opening: { type: "categorical", values: ["closed", "half", "open"] } },
      },
    });
  });

  it("adds several values to a declared attribute with no shape focused", async () => {
    act(() => useStore.getState().setRegistry({ [SUBJECT]: DECLARING }));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue(landed);
    render(<AttributePanel focused={null} />);
    fireEvent.click(screen.getByRole("button", { name: ADD_A_VALUE }));
    await act(() => Promise.resolve(enter(`new ${DECLARED} value`, "half")));
    await act(() => Promise.resolve(enter(`new ${DECLARED} value`, "wide")));

    expect(saveSpy).toHaveBeenCalledTimes(2);
    expect(saveSpy.mock.calls[1][0].subjects).toEqual({
      [SUBJECT]: {
        ...BARE,
        attributes: {
          [DECLARED]: {
            ...SERVED_ATTRIBUTES[DECLARED],
            values: [...DECLARED_VALUES, "half", "wide"],
          },
        },
      },
    });
  });

  it("refuses a name the subject already declares, inline, and posts nothing", async () => {
    act(() =>
      useStore
        .getState()
        .setRegistry({ [SUBJECT]: DECLARING }, LABEL_ANSWERS.registry_loaded.version),
    );
    const saveSpy = vi.spyOn(subjectsApi, "save");
    render(<AttributePanel focused={null} />);
    fillAttributeDraft(DECLARED, ["large"]);
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("Add"))));

    expect(screen.getByText(`${DECLARED} is already declared.`)).toBeInTheDocument();
    expect(saveSpy).not.toHaveBeenCalled();
  });
});

describe("AttributePanel registry-growing on a lost audit line", () => {
  beforeEach(() => act(() => useStore.getState().setRegistry({ [SUBJECT]: BARE })));

  it("adopts the committed registry and toasts the gap message on the audit-gap 409", async () => {
    const gap = LABEL_ANSWERS.registry_audit_gap;
    vi.spyOn(subjectsApi, "save").mockRejectedValue(await thrown(gap));

    render(<AttributePanel focused={null} />);
    fillAttributeDraft("opening", ["closed", "open"]);
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("Add"))));

    // The committed registry is adopted as though the save had answered 200, and the gap
    // itself is surfaced as a toast rather than swallowed.
    const { committed, message } = gap.body.detail;
    expect(useStore.getState().registry.version).toBe(committed.version);
    expect(useStore.getState().toasts.some((t) => t.message === message)).toBe(true);
  });

  it("reloads the registry from the server on an ordinary refusal, unlike the audit-gap case", async () => {
    vi.spyOn(subjectsApi, "save").mockRejectedValue(await thrown(LABEL_ANSWERS.registry_stale));
    const loaded = LABEL_ANSWERS.registry_loaded;
    vi.spyOn(subjectsApi, "load").mockResolvedValue(loaded);

    render(<AttributePanel focused={null} />);
    fillAttributeDraft("opening", ["closed", "open"]);
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("Add"))));

    expect(useStore.getState().registry.subjects).toEqual(loaded.subjects);
    expect(useStore.getState().registry.version).toBe(loaded.version);
  });
});
