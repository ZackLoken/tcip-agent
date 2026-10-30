import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.

import { api } from "@/api/client";
import { resultsApi } from "@/api/inference";
import { subjectsApi } from "@/api/subjects";
import type { ImageBands, ScorerEntry } from "@/api/types.generated";
import { AnnotateToolbar } from "@/components/AnnotateToolbar";
import { defaultBandSelection, type BandSelection } from "@/lib/bandSelection";
import { NO_MATCHES, reviewItems, type ReviewItem } from "@/lib/reviewItems";
import { useStore } from "@/store";
import type { AnnotateView, SubjectState } from "@/store/types";
import { ANNOTATE_ANSWERS, registryOf, SECOND_SUBJECT } from "@/test/annotate";
import { BAND_ANSWERS } from "@/test/bands";
import { REGISTERED_MODELS } from "@/test/deliveryRecords";
import { LABEL_ANSWERS, SERVED_ATTRIBUTES } from "@/test/labels";
import { QUEUE_ANSWERS } from "@/test/queue";
import { thrown } from "@/test/refusals";
import { openTestProject, SELECTED_SUBJECT, selected } from "@/test/store";

const initialStoreState = useStore.getState();

const { registry_saved: REGISTRY_SAVED, registry_loaded: REGISTRY_LOADED } = LABEL_ANSWERS;
const LOADED_VERSION = REGISTRY_LOADED.version;

/** The subjects the load route served, and the one subject they declare, by name. */
const SERVED = REGISTRY_LOADED.subjects ?? {};
const [SERVED_NAME] = Object.keys(SERVED);

/** The subject menu's entry of the served subject, and the pill naming it or nothing. */
const SERVED_PILL = new RegExp(`^${SERVED_NAME}`);
const SUBJECT_PILL = new RegExp(`${SERVED_NAME}|select subject`);

const modeButton = (name: "Box" | "Polygon" | "Point") =>
  screen.getByRole("button", { name: new RegExp(`^${name}$`) });

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  // The toolbar's nav hook persists the settled index; nothing here should reach the backend.
  vi.spyOn(api.dataset, "nav").mockResolvedValue(null);
});

afterEach(cleanup);

function renderToolbar(
  bandsInfo?: ImageBands | null,
  bandSelection?: BandSelection | null,
  extra?: {
    subjectState?: SubjectState | null;
    onComplete?: (next: boolean) => void;
    onCancelSave?: () => void;
    proposing?: boolean;
    onPhrase?: (phrase: string) => void;
    onPropose?: () => void;
    view?: AnnotateView;
    onView?: (next: AnnotateView) => void;
    predictionsAvailable?: boolean;
    focused?: ReviewItem | null;
  },
) {
  render(
    <AnnotateToolbar
      view={extra?.view ?? { proposals: true, predictions: true }}
      onView={extra?.onView ?? (() => {})}
      predictionsAvailable={extra?.predictionsAvailable ?? false}
      focused={extra?.focused ?? null}
      onSave={() => {}}
      onCancelSave={extra?.onCancelSave ?? (() => {})}
      saveDisabled={false}
      dirty={false}
      bandsInfo={bandsInfo}
      bandSelection={bandSelection}
      onBandSelectionChange={() => {}}
      subjectState={extra?.subjectState ?? null}
      onComplete={extra?.onComplete ?? (() => {})}
      proposing={extra?.proposing ?? false}
      onPhrase={extra?.onPhrase ?? (() => {})}
      onPropose={extra?.onPropose ?? (() => {})}
    />,
  );
}

function openEdit() {
  const btn = screen.getByRole("button", { name: "Edit" });
  if (btn.getAttribute("aria-expanded") !== "true") fireEvent.click(btn);
}

const FOUR_BANDS = BAND_ANSWERS.four_spectral;

const PLAIN = BAND_ANSWERS.plain;

describe("AnnotateToolbar save state", () => {
  it("shows a save in flight, and the button then cancels it rather than saving again", () => {
    const onCancelSave = vi.fn();
    renderToolbar(null, null, { onCancelSave });
    const save = () => screen.getByRole("button", { name: /^Saved$|^Saving$/ });
    expect(save()).toHaveTextContent("Saved");
    act(() => void useStore.getState().holdForSave("x"));
    expect(save()).toHaveTextContent("Saving");
    fireEvent.click(save());
    expect(onCancelSave).toHaveBeenCalledOnce();
  });
});

describe("AnnotateToolbar engine prompt", () => {
  const promptToggle = () => screen.getByRole("button", { name: /^(Prompt|Proposing)$/ });

  it("arms the next point or box as a prompt, and disarmed shows no phrase", () => {
    renderToolbar();
    expect(promptToggle()).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByRole("textbox", { name: "Phrase" })).toBeNull();
    fireEvent.click(promptToggle());
    expect(useStore.getState().annotateUi.prompt).toBe(true);
    expect(screen.getByRole("textbox", { name: "Phrase" })).toBeInTheDocument();
  });

  it("sends the phrase alone, with no threshold, tiling or merge control", () => {
    const onPhrase = vi.fn();
    renderToolbar(null, null, { onPhrase });
    fireEvent.click(promptToggle());
    fireEvent.change(screen.getByRole("textbox", { name: "Phrase" }), {
      target: { value: "open bud" },
    });
    expect(screen.queryByRole("spinbutton")).toBeNull();
    expect(screen.queryByRole("button", { name: "Tiling" })).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Find" }));

    expect(onPhrase).toHaveBeenCalledWith("open bud");
  });

  it("counts the held prompts on Propose and sends them together once asked", () => {
    const onPropose = vi.fn();
    renderToolbar(null, null, { onPropose });
    fireEvent.click(promptToggle());
    expect(screen.getByRole("button", { name: "Propose" })).toBeDisabled();
    act(() => {
      useStore.getState().addPromptPoint([10, 10]);
      useStore.getState().addPromptBox([20, 20, 40, 40]);
    });
    fireEvent.click(screen.getByRole("button", { name: "Propose 2" }));

    expect(onPropose).toHaveBeenCalledTimes(1);
  });

  it("disarming drops the held prompts", () => {
    renderToolbar();
    fireEvent.click(promptToggle());
    act(() => useStore.getState().addPromptPoint([10, 10]));
    fireEvent.click(promptToggle());

    expect(useStore.getState().annotateUi.pendingPrompts).toEqual({
      points: [],
      boxes: [],
      found: [],
    });
  });

  it("while a prompt is in flight, says so and refuses another send", () => {
    const onPhrase = vi.fn();
    renderToolbar(null, null, { onPhrase, proposing: true });
    fireEvent.click(promptToggle());
    fireEvent.change(screen.getByRole("textbox", { name: "Phrase" }), {
      target: { value: "open bud" },
    });

    expect(promptToggle()).toHaveTextContent("Proposing");
    expect(screen.getByRole("button", { name: "Find" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Find" }));
    expect(onPhrase).not.toHaveBeenCalled();
  });
});

describe("AnnotateToolbar draw mode", () => {
  it("offers all three geometry kinds, with the active one pressed", () => {
    renderToolbar();
    expect(modeButton("Box")).toHaveAttribute("aria-pressed", "true"); // the default mode
    expect(modeButton("Polygon")).toHaveAttribute("aria-pressed", "false");
    expect(modeButton("Point")).toHaveAttribute("aria-pressed", "false");
  });

  it("switching to Point sets the store's mode and moves the pressed state", () => {
    renderToolbar();
    fireEvent.click(modeButton("Point"));
    expect(useStore.getState().gui.mode).toBe("point");
    expect(modeButton("Point")).toHaveAttribute("aria-pressed", "true");
    expect(modeButton("Box")).toHaveAttribute("aria-pressed", "false");

    // ...and back out again, so Point is a mode like the others rather than a trap.
    fireEvent.click(modeButton("Box"));
    expect(useStore.getState().gui.mode).toBe("box");
    expect(modeButton("Point")).toHaveAttribute("aria-pressed", "false");
  });

  it("counts points in the subject pill alongside boxes, polygons and ratings", () => {
    renderToolbar();
    fireEvent.click(modeButton("Point"));
    act(() => {
      const s = useStore.getState();
      s.setActiveSubject("tip");
      s.add("points", { x: 1, y: 2, subject: "tip", attributes: {} });
      s.add("points", { x: 3, y: 4, subject: "tip", attributes: {} });
      s.add("points", { x: 5, y: 6, subject: "other", attributes: {} });
    });
    // The pill shows the active subject's own count: a placed point is an annotation like any other.
    expect(screen.getByText("(2)")).toBeInTheDocument();
  });

  it("Snap and Stream stay polygon-only in point mode (they have no meaning for a point)", () => {
    renderToolbar();
    fireEvent.click(modeButton("Point"));
    openEdit();
    expect(screen.getByRole("menuitemcheckbox", { name: "Snap" })).toBeDisabled();
    expect(screen.getByRole("menuitemcheckbox", { name: "Stream" })).toBeDisabled();
  });

  it("the Edit menu holds the five editing tools and nothing else", () => {
    renderToolbar();
    openEdit();
    const items = screen.getByRole("menu").querySelectorAll("button");
    expect(Array.from(items, (item) => item.textContent)).toEqual([
      "Snap",
      "Stream",
      "Cut",
      "Undo",
      "Redo",
    ]);
    expect(screen.queryByRole("button", { name: "Editor" })).toBeNull();
  });

  it("the View menu toggles labels, proposals and predictions, the last only with a bucket", () => {
    const onView = vi.fn();
    renderToolbar(null, null, { onView });
    fireEvent.click(screen.getByRole("button", { name: "View" }));
    const items = screen.getByRole("menu").querySelectorAll("button");
    expect(Array.from(items, (item) => item.textContent)).toEqual([
      "Labels",
      "Proposals",
      "Predictions",
    ]);
    expect(screen.getByRole("menuitemcheckbox", { name: "Predictions" })).toBeDisabled();

    fireEvent.click(screen.getByRole("menuitemcheckbox", { name: "Labels" }));
    expect(useStore.getState().annotateUi.visible).toBe(false);
    fireEvent.click(screen.getByRole("menuitemcheckbox", { name: "Proposals" }));
    expect(onView).toHaveBeenCalledWith({ proposals: false, predictions: true });

    cleanup();
    renderToolbar(null, null, { onView, predictionsAvailable: true });
    fireEvent.click(screen.getByRole("button", { name: "View" }));
    fireEvent.click(screen.getByRole("menuitemcheckbox", { name: "Predictions" }));
    expect(onView).toHaveBeenLastCalledWith({ proposals: true, predictions: false });
  });

  it("groups the tools and prompt, the editing menus and Save, and the image controls", () => {
    renderToolbar();
    const row = screen.getByRole("group", { name: "Tool" }).parentElement!;
    const order = Array.from(row.children, (child) =>
      child.getAttribute("aria-hidden") === "true" ? "|" : (child.textContent ?? ""),
    );
    expect(order.filter((o) => o === "|")).toHaveLength(2);
    expect(order.indexOf("|")).toBeGreaterThan(order.findIndex((o) => o.includes("Point")));
    expect(order.lastIndexOf("|")).toBeLessThan(order.findIndex((o) => o.includes("Complete")));
  });

  it("renders Point immediately to the left of Box (Point, Box, Polygon)", () => {
    renderToolbar();
    const group = screen.getByRole("group", { name: "Tool" });
    const labels = Array.from(group.querySelectorAll("button")).map((b) => b.textContent);
    expect(labels).toEqual(["Point", "Box", "Polygon"]);
  });

  it("Box and Polygon describe themselves, the same as Point does", () => {
    renderToolbar();
    expect(screen.getByRole("button", { name: "Box" })).toHaveAttribute(
      "title",
      expect.stringMatching(/box/i),
    );
    expect(screen.getByRole("button", { name: "Polygon" })).toHaveAttribute(
      "title",
      expect.stringMatching(/polygon/i),
    );
  });

  it("the image-position textbox has an accessible name, not just a visible counter beside it", () => {
    renderToolbar();
    expect(screen.getByRole("textbox", { name: "Image position" })).toBeInTheDocument();
  });
});

describe("AnnotateToolbar Cut item", () => {
  it("states the disabled reason outside polygon mode", () => {
    renderToolbar();
    openEdit();
    const cutItem = screen.getByRole("menuitemcheckbox", { name: "Cut" });
    expect(cutItem).toBeDisabled();
    expect(cutItem).toHaveAttribute("title", "Cut: in polygon mode only");
  });

  it("states the how-to once enabled, in polygon mode", () => {
    renderToolbar();
    fireEvent.click(modeButton("Polygon"));
    openEdit();
    const cutItem = screen.getByRole("menuitemcheckbox", { name: "Cut" });
    expect(cutItem).not.toBeDisabled();
    expect(cutItem).toHaveAttribute(
      "title",
      "Click two points on either side of the selected polygon to split it (x)",
    );
  });
});

describe("AnnotateToolbar subject authoring", () => {
  function seedDataset() {
    openTestProject(selected({ count: 0 }));
  }

  function openSubjectMenu() {
    fireEvent.click(screen.getByRole("button", { name: /select subject/ }));
  }

  /** Answer the name prompt, then the description prompt, each with the next of `answers`. */
  function answerPrompt(...answers: (string | null)[]) {
    const spy = vi.spyOn(window, "prompt");
    for (const answer of answers) spy.mockReturnValueOnce(answer);
    return spy;
  }

  it("registers a typed name with its surrounding whitespace stripped and its description", async () => {
    // An untrimmed registry key would open a second subject that reads as the first.
    seedDataset();
    act(() => useStore.getState().setRegistry(SERVED));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue(REGISTRY_SAVED);
    answerPrompt("  husk  ", "one husk");
    renderToolbar();

    openSubjectMenu();
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("New subject"))));

    expect(Object.keys(useStore.getState().registry.subjects)).toEqual([SERVED_NAME, "husk"]);
    expect(useStore.getState().gui.active_subject).toBe("husk");
    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].subjects).toEqual({
      ...SERVED,
      husk: { description: "one husk" },
    });
  });

  it("adds nothing when the description prompt is dismissed", async () => {
    seedDataset();
    act(() => useStore.getState().setRegistry(SERVED));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue(REGISTRY_SAVED);
    answerPrompt("husk", null);
    renderToolbar();

    openSubjectMenu();
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("New subject"))));

    expect(Object.keys(useStore.getState().registry.subjects)).toEqual([SERVED_NAME]);
    expect(saveSpy).not.toHaveBeenCalled();
  });

  it("selects an existing subject rather than resetting its attribute definitions", async () => {
    seedDataset();
    const both = { ...SERVED, ...registryOf(SECOND_SUBJECT) };
    act(() => useStore.getState().setRegistry(both));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue(REGISTRY_SAVED);
    answerPrompt(SERVED_NAME);
    renderToolbar();

    openSubjectMenu();
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("New subject"))));

    expect(useStore.getState().registry.subjects).toEqual(both);
    expect(useStore.getState().gui.active_subject).toBe(SERVED_NAME);
    expect(saveSpy).not.toHaveBeenCalled();
  });

  it("adds nothing when the prompt is dismissed or answered with only whitespace", async () => {
    seedDataset();
    act(() => useStore.getState().setRegistry(SERVED));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue(REGISTRY_SAVED);
    const promptSpy = answerPrompt(null);
    renderToolbar();

    openSubjectMenu();
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("New subject"))));
    expect(Object.keys(useStore.getState().registry.subjects)).toEqual([SERVED_NAME]);

    promptSpy.mockReturnValue("   ");
    openSubjectMenu();
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("New subject"))));

    expect(Object.keys(useStore.getState().registry.subjects)).toEqual([SERVED_NAME]);
    expect(useStore.getState().gui.active_subject).toBeNull();
    expect(saveSpy).not.toHaveBeenCalled();
  });

  it("posts the loaded registry version and stores the version the save returns", async () => {
    seedDataset();
    act(() => useStore.getState().setRegistry(SERVED, LOADED_VERSION));
    const saveSpy = vi.spyOn(subjectsApi, "save").mockResolvedValue(REGISTRY_SAVED);
    answerPrompt("husk", "one husk");
    renderToolbar();

    openSubjectMenu();
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("New subject"))));

    expect(saveSpy.mock.calls[0][0].version).toBe(LOADED_VERSION);
    expect(useStore.getState().registry.version).toBe(REGISTRY_SAVED.version);
  });

  it("reloads the registry from the server when the save is refused", async () => {
    seedDataset();
    act(() => useStore.getState().setRegistry(SERVED, LOADED_VERSION));
    vi.spyOn(subjectsApi, "save").mockRejectedValue(await thrown(LABEL_ANSWERS.registry_stale));
    vi.spyOn(subjectsApi, "load").mockResolvedValue(REGISTRY_LOADED);
    answerPrompt("husk", "one husk");
    renderToolbar();

    openSubjectMenu();
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("New subject"))));

    // The refused optimistic add is discarded in favor of what the server actually holds.
    expect(useStore.getState().registry.subjects).toEqual(REGISTRY_LOADED.subjects);
    expect(useStore.getState().registry.version).toBe(REGISTRY_LOADED.version);
  });

  it("reverts the optimistically set active subject when the save is refused", async () => {
    seedDataset();
    act(() => {
      useStore.getState().setRegistry(SERVED, LOADED_VERSION);
      useStore.getState().setActiveSubject(SERVED_NAME);
    });
    vi.spyOn(subjectsApi, "save").mockRejectedValue(await thrown(LABEL_ANSWERS.registry_stale));
    vi.spyOn(subjectsApi, "load").mockResolvedValue(REGISTRY_LOADED);
    answerPrompt("husk", "one husk");
    renderToolbar();

    // The served subject is already active, so the pill reads its name rather than the default
    // placeholder.
    fireEvent.click(screen.getByRole("button", { name: SUBJECT_PILL }));
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("New subject"))));

    // "husk" was set optimistically as active; the refusal must not leave it active.
    expect(useStore.getState().gui.active_subject).toBe(SERVED_NAME);
  });

  it("adopts the committed registry and keeps the active subject when the save's audit line is lost", async () => {
    seedDataset();
    act(() => {
      useStore.getState().setRegistry(SERVED, LOADED_VERSION);
      useStore.getState().setActiveSubject(SERVED_NAME);
    });
    const gap = LABEL_ANSWERS.registry_audit_gap;
    vi.spyOn(subjectsApi, "save").mockRejectedValue(await thrown(gap));
    answerPrompt("husk", "one husk");
    renderToolbar();

    // The served subject is already active, so the pill reads its name rather than the default
    // placeholder.
    fireEvent.click(screen.getByRole("button", { name: SUBJECT_PILL }));
    await act(() => Promise.resolve(fireEvent.click(screen.getByText("New subject"))));

    // The committed body is adopted as though the save had answered 200, and the optimistic
    // active subject is kept rather than reverted (unlike an ordinary refusal above).
    expect(useStore.getState().registry.version).toBe(gap.body.detail.committed.version);
    expect(useStore.getState().gui.active_subject).toBe("husk");
  });
});

describe("AnnotateToolbar subject menu attributes", () => {
  it("holds the active subject's attributes under the subject list, absent with no subject", () => {
    openTestProject(selected({ count: 0 }));
    act(() => useStore.getState().setRegistry(SERVED));
    renderToolbar();
    fireEvent.click(screen.getByRole("button", { name: /select subject/ }));
    expect(screen.queryByTestId("subject-attributes")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: SERVED_PILL }));
    fireEvent.click(screen.getByRole("button", { name: SERVED_PILL }));
    expect(screen.getByTestId("subject-attributes")).toBeInTheDocument();
    expect(screen.getByText(`${SERVED_NAME} attributes`)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Add attribute" })).toBeInTheDocument();
  });

  it("edits the focused box's attributes inside the menu", () => {
    const [[declared, { values }]] = Object.entries(SERVED_ATTRIBUTES);
    openTestProject(selected({ count: 0 }));
    act(() => {
      useStore.getState().setRegistry(SERVED);
      useStore.getState().setActiveSubject(SERVED_NAME);
      useStore
        .getState()
        .add("boxes", { x1: 0, y1: 0, x2: 5, y2: 5, subject: SERVED_NAME, attributes: {} });
    });
    const { canvas } = useStore.getState();
    const [focused] = reviewItems({
      canvas,
      proposals: [],
      matches: NO_MATCHES,
      reviewing: false,
      flags: canvas.flags,
    });
    renderToolbar(null, null, { focused });
    fireEvent.click(screen.getByRole("button", { name: SERVED_PILL }));
    fireEvent.change(screen.getByRole("combobox", { name: declared }), {
      target: { value: values[0] },
    });
    expect(useStore.getState().canvas.boxes[0].attributes).toEqual({ [declared]: values[0] });
  });
});

describe("AnnotateToolbar band picker (progressive disclosure)", () => {
  it("is absent with no bandsInfo at all (a project with no channel-count fact yet)", () => {
    renderToolbar(null, null);
    expect(screen.queryByLabelText("R band")).not.toBeInTheDocument();
  });

  it("is hidden for a frame the server displays plain", () => {
    renderToolbar(PLAIN, defaultBandSelection(PLAIN));
    expect(screen.queryByLabelText("R band")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Band")).not.toBeInTheDocument();
  });

  it("is shown for a >3-band (multispectral) dataset", () => {
    renderToolbar(FOUR_BANDS, defaultBandSelection(FOUR_BANDS));
    expect(screen.getByLabelText("R band")).toBeInTheDocument();
    expect(screen.getByLabelText("G band")).toBeInTheDocument();
    expect(screen.getByLabelText("B band")).toBeInTheDocument();
  });
});

function seedSubject(subject: string) {
  openTestProject(selected({ subject, count: 1 }));
}

describe("AnnotateToolbar image stepping", () => {
  it("leaves the focused item behind on the image it left", () => {
    openTestProject(selected({ count: 2 }));
    useStore
      .getState()
      .setFocus({ kind: "proposal", index: 3, bucket: ANNOTATE_ANSWERS.buckets.review });
    expect(useStore.getState().canvas.focus).not.toBeNull();
    renderToolbar();

    fireEvent.click(screen.getByRole("button", { name: "Next image" }));

    expect(useStore.getState().gui.dataset?.current_image_index).toBe(1);
    expect(useStore.getState().canvas.focus).toBeNull();
  });
});

describe("AnnotateToolbar image filter", () => {
  it("names each filter with its count and steps through the kept images only", () => {
    // The produced states: the first negative, the second partial, the fifth complete.
    const { states } = ANNOTATE_ANSWERS.marked_states;
    const held = Object.values(states);
    const counted = (kept: (state: SubjectState) => boolean) => held.filter(kept).length;
    openTestProject(selected({ subject: SELECTED_SUBJECT }));
    act(() => useStore.getState().setImageStates(states));
    renderToolbar();
    const filter = screen.getByRole("combobox", { name: "Images" });
    expect(Array.from(filter.querySelectorAll("option"), (o) => o.textContent)).toEqual([
      `all (${held.length})`,
      `unannotated (${counted((s) => s === "unannotated")})`,
      `in progress (${counted((s) => s === "partial")})`,
      `complete (${counted((s) => s === "complete" || s === "negative")})`,
      "rank a review queue",
    ]);

    const kept = held.flatMap((s, i) => (s === "complete" || s === "negative" ? [i] : []));
    fireEvent.change(filter, { target: { value: "complete" } });
    expect(screen.getByRole("textbox", { name: "Image position" })).toHaveValue("1");
    fireEvent.click(screen.getByRole("button", { name: "Next image" }));
    expect(useStore.getState().gui.dataset?.current_image_index).toBe(kept[1]);
    expect(screen.getByRole("button", { name: "Next image" })).toBeDisabled();
  });
});

describe("AnnotateToolbar review queue", () => {
  const { completed: COMPLETED, failed: FAILED, launched: LAUNCHED } = QUEUE_ANSWERS;
  const [MODEL] = REGISTERED_MODELS.models;
  const RANKED = COMPLETED.ranked!;
  const QUEUE_SELECTION = QUEUE_ANSWERS.selection.selection;
  const imagesSelect = () => screen.getByRole("combobox", { name: "Images" });
  const at = () => useStore.getState().gui.dataset?.current_image_index;
  /** The value names a served scorer's values schema lists under its properties, in order. */
  const valueNames = (scorer: ScorerEntry) =>
    Object.keys(scorer.values_schema.properties as object);

  /** Open the launch from the Images select and wait for its scorers to load. */
  async function openLaunch() {
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue(REGISTERED_MODELS);
    vi.spyOn(api.annotate, "scorers").mockResolvedValue(QUEUE_ANSWERS.scorers);
    fireEvent.change(imagesSelect(), { target: { value: "rank_queue" } });
    await screen.findByRole("option", { name: "combined" });
    await screen.findByRole("option", { name: MODEL.entry.name });
  }

  it("launches with the values the person states, none prefilled, and walks the completed queue in rank order", async () => {
    openTestProject(QUEUE_SELECTION);
    const launch = vi.spyOn(api.annotate, "launchQueue").mockResolvedValue(LAUNCHED);
    vi.spyOn(api.annotate, "queueJob")
      .mockResolvedValueOnce(QUEUE_ANSWERS.running)
      .mockResolvedValue(COMPLETED);
    renderToolbar();
    await openLaunch();

    const rank = screen.getByRole("button", { name: "Rank" });
    expect(screen.getByLabelText("Budget")).toHaveValue(null);
    expect(screen.getByLabelText("Reason")).toHaveValue("");
    expect(rank).toBeDisabled();

    fireEvent.change(screen.getByRole("combobox", { name: "Scorer" }), {
      target: { value: "combined" },
    });
    for (const name of valueNames(
      QUEUE_ANSWERS.scorers.scorers.find((s) => s.name === "combined")!,
    )) {
      expect(screen.getByLabelText(name)).toHaveValue("");
    }
    fireEvent.change(screen.getByRole("combobox", { name: "Model" }), {
      target: { value: MODEL.entry.checkpoint_path },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Scorer" }), {
      target: { value: RANKED.scorer.name },
    });
    fireEvent.change(screen.getByLabelText("Budget"), { target: { value: String(RANKED.budget) } });
    expect(rank).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Reason"), { target: { value: RANKED.rationale } });
    fireEvent.click(rank);

    await waitFor(() => expect(launch).toHaveBeenCalledOnce());
    expect(launch.mock.calls[0][0]).toEqual({
      checkpoint_path: MODEL.entry.checkpoint_path,
      scorer: RANKED.scorer,
      budget: RANKED.budget,
      rationale: RANKED.rationale,
    });

    await waitFor(() => expect(useStore.getState().annotateUi.reviewQueue?.ranked).toBeTruthy(), {
      timeout: 5000,
    });
    const names = QUEUE_SELECTION.images.map((image) => image.name);
    const walked: string[] = [];
    const position = screen.getByRole("textbox", { name: "Image position" });
    fireEvent.change(position, { target: { value: "1" } });
    fireEvent.keyDown(position, { key: "Enter" });
    walked.push(names[at()!]);
    while (!screen.getByRole("button", { name: "Next image" }).hasAttribute("disabled")) {
      fireEvent.click(screen.getByRole("button", { name: "Next image" }));
      walked.push(names[at()!]);
    }
    expect(walked).toEqual(RANKED.queue.map((entry) => entry.image));
  });

  /** Open the launch over the typed catalog, choose its model and the typed scorer, and state
   *  `typed` for each of its parameters with the budget and reason; return the Rank button. */
  async function stateTyped(typed: Record<string, string>) {
    renderToolbar();
    vi.spyOn(resultsApi, "registeredModels").mockResolvedValue(REGISTERED_MODELS);
    vi.spyOn(api.annotate, "scorers").mockResolvedValue(QUEUE_ANSWERS.typed_scorers);
    fireEvent.change(imagesSelect(), { target: { value: "rank_queue" } });
    await screen.findByRole("option", { name: TYPED.name });
    await screen.findByRole("option", { name: MODEL.entry.name });
    fireEvent.change(screen.getByRole("combobox", { name: "Model" }), {
      target: { value: MODEL.entry.checkpoint_path },
    });
    fireEvent.change(screen.getByRole("combobox", { name: "Scorer" }), {
      target: { value: TYPED.name },
    });
    for (const [name, text] of Object.entries(typed)) {
      fireEvent.change(screen.getByLabelText(new RegExp(`^${name}`)), { target: { value: text } });
    }
    fireEvent.change(screen.getByLabelText("Budget"), { target: { value: String(RANKED.budget) } });
    fireEvent.change(screen.getByLabelText("Reason"), { target: { value: RANKED.rationale } });
    return screen.getByRole("button", { name: "Rank" });
  }

  // The typed sample factory's parameters, by the JSON type each schema admits: a boolean, an
  // array of numbers and a number or null.
  const TYPED = QUEUE_ANSWERS.typed_scorers.scorers.find((s) => s.name === "typed")!;
  const [FLAG, WEIGHTS, CAP] = valueNames(TYPED);

  it("states each value as the JSON type its parameter's schema admits", async () => {
    openTestProject(QUEUE_SELECTION);
    const launch = vi.spyOn(api.annotate, "launchQueue").mockResolvedValue(LAUNCHED);
    vi.spyOn(api.annotate, "queueJob").mockResolvedValue(COMPLETED);

    const rank = await stateTyped({ [FLAG]: "false", [WEIGHTS]: "[1, 2.5]", [CAP]: "0.5" });
    fireEvent.click(rank);

    await waitFor(() => expect(launch).toHaveBeenCalledOnce());
    expect(launch.mock.calls[0][0].scorer).toEqual({
      name: TYPED.name,
      [FLAG]: false,
      [WEIGHTS]: [1, 2.5],
      [CAP]: 0.5,
    });
  });

  it("shows a refused launch's refusal as the launch route wrote it, holding no job", async () => {
    openTestProject(QUEUE_SELECTION);
    const refused = await thrown(QUEUE_ANSWERS.refused_launch);
    vi.spyOn(api.annotate, "launchQueue").mockRejectedValue(refused);

    const rank = await stateTyped({ [CAP]: "half" });
    fireEvent.click(rank);

    expect((await screen.findByRole("alert")).textContent).toBe(refused.message);
    expect(useStore.getState().annotateUi.reviewQueue).toBeNull();
  });

  it("walks every image of a capture the queue did not rank, though it completes there", async () => {
    openTestProject(QUEUE_SELECTION);
    useStore.getState().setReviewQueue(LAUNCHED);
    useStore.getState().setImageFilter("queue");
    let complete = () => {};
    vi.spyOn(api.annotate, "queueJob").mockImplementation(
      () =>
        new Promise((resolve) => {
          complete = () => resolve(COMPLETED);
        }),
    );
    renderToolbar();

    const later = QUEUE_ANSWERS.later_selection.selection;
    act(() => openTestProject(later));
    await waitFor(() => expect(api.annotate.queueJob).toHaveBeenCalled(), { timeout: 5000 });
    act(() => complete());
    await waitFor(() => expect(useStore.getState().annotateUi.reviewQueue?.ranked).toBeTruthy());

    expect(imagesSelect()).toHaveValue("all");
    expect(screen.getByRole("textbox", { name: "Image position" })).toHaveValue("1");
  });

  it("shows a failed job's error as the backend wrote it and holds no ordering", async () => {
    openTestProject(QUEUE_SELECTION);
    useStore.getState().setReviewQueue(LAUNCHED);
    vi.spyOn(api.annotate, "queueJob").mockResolvedValue(FAILED);
    renderToolbar();

    await waitFor(
      () => expect(useStore.getState().toasts.map((t) => t.message)).toContain(FAILED.error),
      { timeout: 5000 },
    );
    expect(useStore.getState().annotateUi.imageFilter).toBe("all");
    expect(screen.queryByRole("option", { name: /^review queue/ })).toBeNull();
  });

  it("offers a held queue only on the capture it ranked", () => {
    useStore.getState().setReviewQueue(COMPLETED);
    useStore.getState().setImageFilter("queue");
    openTestProject(QUEUE_ANSWERS.later_selection.selection);
    renderToolbar();

    expect(screen.queryByRole("option", { name: /^review queue/ })).toBeNull();
    expect(imagesSelect()).toHaveValue("all");

    cleanup();
    openTestProject(QUEUE_SELECTION);
    useStore.getState().setImageFilter("queue");
    renderToolbar();
    expect(
      screen.getByRole("option", { name: `review queue (${RANKED.queue.length})` }),
    ).toBeInTheDocument();
  });
});

describe("AnnotateToolbar Complete toggle", () => {
  it("reads checked from the served state, with or without annotations", () => {
    seedSubject(SELECTED_SUBJECT);
    renderToolbar(null, null, { subjectState: "negative" });
    expect(screen.getByRole("checkbox", { name: /Complete/ })).toBeChecked();
    cleanup();
    renderToolbar(null, null, { subjectState: "partial" });
    expect(screen.getByRole("checkbox", { name: /Complete/ })).not.toBeChecked();
  });

  it("asks the tab to mark or withdraw, and is disabled until the document loads", () => {
    seedSubject(SELECTED_SUBJECT);
    const onComplete = vi.fn();
    renderToolbar(null, null, { subjectState: "partial", onComplete });
    fireEvent.click(screen.getByRole("checkbox", { name: /Complete/ }));
    expect(onComplete).toHaveBeenCalledWith(true);
    cleanup();
    renderToolbar(null, null, { subjectState: null, onComplete });
    expect(screen.getByRole("checkbox", { name: /Complete/ })).toBeDisabled();
  });
});
