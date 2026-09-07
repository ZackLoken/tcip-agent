import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { ProjectPicker, localTime } from "@/components/ProjectPicker";
import { useStore } from "@/store";

vi.mock("@/api/client", () => {
  return {
    api: {
      projects: {
        list: vi.fn(),
        setActive: vi.fn().mockResolvedValue({ name: "x", path: "/x" }),
        removalPreview: vi.fn(),
        remove: vi.fn(),
        releaseBinding: vi.fn(),
      },
      dataset: {
        select: vi.fn(),
        tree: vi.fn(),
      },
    },
  };
});

import { api } from "@/api/client";
import type { ProjectSummary } from "@/api/client";

const PROJECTS: ProjectSummary[] = [
  {
    name: "crop_a_subject_a_valley-farm",
    path: "/ws/crop_a_subject_a_valley-farm",
    created: 1_700_000_000,
    modified: 1_700_000_500,
    dates: ["2026-02-11", "2026-03-01"],
    subjects: ["subject_a", "bush"],
    models: ["baseline"],
    // subject_a labelled (+ baseline predicted) on 02-11; bush labelled on 03-01.
    subjects_by_date: { "2026-02-11": ["subject_a"], "2026-03-01": ["bush"] },
    models_by_date: { "2026-02-11": ["baseline"], "2026-03-01": [] },
    image_count: 42,
    is_active: false,
    site: "north orchard",
    site_problem: null,
    label_problem: null,
    removal_refusal: null,
    removal_releasable: false,
    dependency_warnings: [],
    dependency_problem: null,
  },
  {
    name: "crop_b_burr_site-b",
    path: "/ws/crop_b_burr_site-b",
    created: 1_700_000_000,
    modified: 1_700_000_100,
    dates: ["2026-03-05"],
    subjects: [],
    models: [],
    subjects_by_date: { "2026-03-05": [] },
    models_by_date: { "2026-03-05": [] },
    image_count: 7,
    is_active: false,
    site: null,
    site_problem:
      "No site recorded yet: record it with initialize_project(<path>, site=<site>) or " +
      "tcip write-project-site, for /ws/crop_b_burr_site-b",
    label_problem: null,
    removal_refusal: null,
    removal_releasable: false,
    dependency_warnings: [],
    dependency_problem: null,
  },
];

afterEach(cleanup);
beforeEach(() => {
  // Reset store dataset + the module-level auto-open guard between tests by reloading
  // the module is overkill; instead each test controls `active` so auto-open is inert.
  useStore.getState().clearDataset();
  vi.mocked(api.dataset.select).mockReset();
  vi.mocked(api.projects.setActive).mockReset();
  vi.mocked(api.projects.setActive).mockResolvedValue({ name: "x", path: "/x" });
  vi.mocked(api.projects.removalPreview).mockReset();
  vi.mocked(api.projects.remove).mockReset();
  vi.mocked(api.projects.releaseBinding).mockReset();
});

const NO_REFUSAL_PREVIEW = {
  external_roots: [],
  dependent_projects: [],
  refusal: null,
  releasable: false,
};

describe("ProjectPicker", () => {
  it("lists workspace projects with stats", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    expect(await screen.findByText("crop_a_subject_a_valley-farm")).toBeInTheDocument();
    expect(screen.getByText("crop_b_burr_site-b")).toBeInTheDocument();
    expect(screen.getByText("42 image(s)")).toBeInTheDocument();
  });

  it("counts each card's dates, subjects and models from that project's own summary", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    // The three counts differ within the first card, so a stat reading off the wrong list shows.
    expect(await screen.findByText("2 dates")).toBeInTheDocument();
    expect(screen.getByText("2 subjects")).toBeInTheDocument();
    expect(screen.getByText("1 model")).toBeInTheDocument();
    expect(screen.getByText("1 date")).toBeInTheDocument();
    expect(screen.getByText("0 subjects")).toBeInTheDocument();
    expect(screen.getByText("0 models")).toBeInTheDocument();
  });

  it("renders each card's site under its name", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    expect(await screen.findByText("north orchard")).toBeInTheDocument();
  });

  it("renders the site problem text in place of a site the project has none of", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    expect(
      await screen.findByText(
        "No site recorded yet: record it with initialize_project(<path>, site=<site>) or " +
          "tcip write-project-site, for /ws/crop_b_burr_site-b",
      ),
    ).toBeInTheDocument();
  });

  it("renders a card's label_problem beside its site", async () => {
    const withLabelProblem = [
      {
        ...PROJECTS[0],
        label_problem:
          "/ws/crop_a_subject_a_valley-farm/annotations/2026-02-11/IMG_0000.json does not decode as JSON",
      },
      PROJECTS[1],
    ];
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: withLabelProblem,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    expect(
      await screen.findByText(
        "/ws/crop_a_subject_a_valley-farm/annotations/2026-02-11/IMG_0000.json does not decode as JSON",
      ),
    ).toBeInTheDocument();
  });

  it("shows an empty state when there are no projects", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: [],
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);
    expect(await screen.findByText("No projects yet")).toBeInTheDocument();
  });

  it("opens a project via /dataset/select with project root = dataset root", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      generation: 1,
      selection: {
        project_root: "/ws/crop_a_subject_a_valley-farm",
        dataset_root: "/ws/crop_a_subject_a_valley-farm",
        subject: "subject_a",
        date: "2026-03-01",
        image_list: [],
        current_image_index: 0,
        annotations_dir: null,
        predictions_dir: null,
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    render(<ProjectPicker />);
    fireEvent.click(await screen.findByText("crop_a_subject_a_valley-farm"));
    // Default date is the most recent ISO date.
    fireEvent.click(screen.getByText("Open project"));

    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    expect(arg.project_root).toBe("/ws/crop_a_subject_a_valley-farm");
    expect(arg.dataset_root).toBe(arg.project_root);
    expect(arg.date).toBe("2026-03-01");
    // Store now holds the selection (dataset ready).
    await waitFor(() =>
      expect(useStore.getState().gui.dataset.dataset_root).toBe("/ws/crop_a_subject_a_valley-farm"),
    );
    // Opening from the picker adopts the project (writes the active-project marker).
    expect(api.projects.setActive).toHaveBeenCalledWith("crop_a_subject_a_valley-farm");
  });

  it("still opens the project when the marker write is rejected, and surfaces a toast", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      generation: 1,
      selection: {
        project_root: "/ws/crop_a_subject_a_valley-farm",
        dataset_root: "/ws/crop_a_subject_a_valley-farm",
        subject: "subject_a",
        date: "2026-03-01",
        image_list: [],
        current_image_index: 0,
        annotations_dir: null,
        predictions_dir: null,
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
    vi.mocked(api.projects.setActive).mockRejectedValue(new Error("locked"));
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<ProjectPicker />);
    fireEvent.click(await screen.findByText("crop_a_subject_a_valley-farm"));
    fireEvent.click(screen.getByText("Open project"));

    await waitFor(() =>
      expect(useStore.getState().gui.dataset.dataset_root).toBe("/ws/crop_a_subject_a_valley-farm"),
    );
    expect(pushToast).toHaveBeenCalledWith(expect.stringContaining("crop_a_subject_a_valley-farm"));
  });

  it("filters the subject options to the selected date's labelled subjects", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);
    fireEvent.click(await screen.findByText("crop_a_subject_a_valley-farm"));

    // Default date is the most recent ISO date (2026-03-01), where only 'bush' is labelled.
    const subjectSelect = screen.getByLabelText("Subject") as HTMLSelectElement;
    let opts = Array.from(subjectSelect.options).map((o) => o.value);
    expect(opts).toContain("bush");
    expect(opts).not.toContain("subject_a");

    // Switch to 2026-02-11, where only 'subject_a' is labelled.
    fireEvent.change(screen.getByLabelText("Date"), { target: { value: "2026-02-11" } });
    opts = Array.from(subjectSelect.options).map((o) => o.value);
    expect(opts).toContain("subject_a");
    expect(opts).not.toContain("bush");
  });

  it("names the unset subject and model options only when they carry the glyph, not the state word", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    // 2026-02-11 has options for both selects, so each renders its glyph branch.
    fireEvent.click(await screen.findByText("crop_a_subject_a_valley-farm"));
    fireEvent.change(screen.getByLabelText("Date"), { target: { value: "2026-02-11" } });

    const subjectSelect = screen.getByLabelText("Subject") as HTMLSelectElement;
    expect(
      within(subjectSelect).getByRole("option", { name: "no subject chosen" }),
    ).toBeInTheDocument();
    const modelSelect = screen.getByLabelText("Model") as HTMLSelectElement;
    expect(
      within(modelSelect).getByRole("option", { name: "no model chosen" }),
    ).toBeInTheDocument();

    // crop_b's only date has neither, so the state word stands as its own accessible name.
    fireEvent.click(screen.getByText("crop_b_burr_site-b"));
    const emptySubjectSelect = screen.getByLabelText("Subject") as HTMLSelectElement;
    expect(
      within(emptySubjectSelect).getByRole("option", { name: "no labels" }),
    ).toBeInTheDocument();
    const emptyModelSelect = screen.getByLabelText("Model") as HTMLSelectElement;
    expect(within(emptyModelSelect).getByRole("option", { name: "no preds" })).toBeInTheDocument();
  });

  it("auto-opens the active project on first load without writing the marker", async () => {
    // autoOpenAttempted is module-scoped, already tripped by earlier tests in this file;
    // reset the module registry to exercise a fresh first load.
    vi.resetModules();
    const { api: freshApi } = await import("@/api/client");
    const { ProjectPicker: FreshProjectPicker } = await import("@/components/ProjectPicker");

    vi.mocked(freshApi.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: PROJECTS[0].name,
      active_path: PROJECTS[0].path,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(freshApi.dataset.select).mockResolvedValue({
      status: "ok",
      generation: 1,
      selection: {
        project_root: PROJECTS[0].path,
        dataset_root: PROJECTS[0].path,
        subject: "bush",
        date: "2026-03-01",
        image_list: [],
        current_image_index: 0,
        annotations_dir: null,
        predictions_dir: null,
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    render(<FreshProjectPicker />);

    await waitFor(() => expect(freshApi.dataset.select).toHaveBeenCalledTimes(1));
    expect(freshApi.projects.setActive).not.toHaveBeenCalled();
  });

  it("attempts auto-open at most once per page load, even when the first mount does not survive its fetch", async () => {
    vi.resetModules();
    const { api: freshApi } = await import("@/api/client");
    const { ProjectPicker: FreshProjectPicker } = await import("@/components/ProjectPicker");

    // The first mount's fetch never resolves before it unmounts, mirroring every load where the
    // app opens the project itself ahead of the picker settling.
    vi.mocked(freshApi.projects.list).mockReturnValueOnce(new Promise(() => {}));
    const { unmount } = render(<FreshProjectPicker />);
    unmount();

    vi.mocked(freshApi.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: PROJECTS[0].name,
      active_path: PROJECTS[0].path,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(freshApi.dataset.select).mockResolvedValue({
      status: "ok",
      generation: 1,
      selection: {
        project_root: PROJECTS[0].path,
        dataset_root: PROJECTS[0].path,
        subject: "bush",
        date: "2026-03-01",
        image_list: [],
        current_image_index: 0,
        annotations_dir: null,
        predictions_dir: null,
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    // A second mount in the same module, e.g. the footer's Switch Project, must not re-open it.
    render(<FreshProjectPicker />);

    await screen.findByText(PROJECTS[0].name);
    expect(freshApi.dataset.select).not.toHaveBeenCalled();
  });

  it("selects a card through a real button named by the project name alone", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    const card = await screen.findByRole("button", { name: "crop_a_subject_a_valley-farm" });
    expect(card).toHaveAttribute("aria-pressed", "false");
    fireEvent.click(card);
    expect(card).toHaveAttribute("aria-pressed", "true");
  });

  // jsdom does not lay out CSS, so this proves the aria-describedby wiring reaches the
  // block's text, not that display: contents keeps its children out of the box model.
  it("describes the card with its site and counts, without folding them into its name", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    const card = await screen.findByRole("button", { name: "crop_a_subject_a_valley-farm" });
    expect(card).toHaveAccessibleName("crop_a_subject_a_valley-farm");
    expect(card).toHaveAccessibleDescription(/north orchard/);
    expect(card).toHaveAccessibleDescription(/42 image\(s\)/);
  });

  it("puts the active badge in the card's description, not its name", async () => {
    const active: ProjectSummary = { ...PROJECTS[0], is_active: true };
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: [active, PROJECTS[1]],
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    const card = await screen.findByRole("button", { name: "crop_a_subject_a_valley-farm" });
    expect(card).toHaveAccessibleName("crop_a_subject_a_valley-farm");
    expect(card).toHaveAccessibleDescription(/active/);
    expect(card.getAttribute("aria-describedby")).not.toBeNull();
    const description = document.getElementById(card.getAttribute("aria-describedby")!);
    expect(description).not.toHaveTextContent("crop_a_subject_a_valley-farm");
  });

  it("names and describes a card correctly when the project name carries a space", async () => {
    const spaced: ProjectSummary = {
      ...PROJECTS[0],
      name: "crop a subject a valley farm",
    };
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: [spaced, PROJECTS[1]],
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    const card = await screen.findByRole("button", { name: "crop a subject a valley farm" });
    expect(card).toHaveAccessibleName("crop a subject a valley farm");
    expect(card).toHaveAccessibleDescription(/north orchard/);
  });

  it("gives the Open project button the full width of its action row", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);
    fireEvent.click(await screen.findByText("crop_a_subject_a_valley-farm"));

    expect(screen.getByText("Open project")).toHaveClass("flex-1");
  });

  it("keeps the selected panel's controls outside any role=button nesting", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);
    fireEvent.click(await screen.findByText("crop_a_subject_a_valley-farm"));

    const card = screen.getByRole("button", { name: "crop_a_subject_a_valley-farm" });
    const dateSelect = screen.getByLabelText("Date");
    const openButton = screen.getByText("Open project");
    expect(card.contains(dateSelect)).toBe(false);
    expect(card.contains(openButton)).toBe(false);
    expect(screen.queryAllByRole("button", { name: "crop_a_subject_a_valley-farm" })).toHaveLength(
      1,
    );
  });

  it("has no advanced folder-open escape hatch (project creation is agent-driven)", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);
    await screen.findByText(PROJECTS[0].name);
    expect(screen.queryByText(/open a folder outside the workspace/i)).not.toBeInTheDocument();
  });
});

async function selectFirstCard() {
  fireEvent.click(await screen.findByText(PROJECTS[0].name));
}

describe("ProjectPicker removal", () => {
  it.each([
    "sample plot is the project this workspace opens by default; choose a different default " +
      "first, or release it as the default",
    "sample plot is the project this backend started on; restart the backend first, and " +
      "start it without a state root naming this project",
    "sample plot is the project the GUI has open; open a different project first, or release " +
      "it as the open project",
  ])("disables Remove with the backend's own reason (%s)", async (reasonText) => {
    const projectsWithReason = [
      { ...PROJECTS[0], removal_refusal: reasonText, removal_releasable: false },
      PROJECTS[1],
    ];
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: projectsWithReason,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);
    await selectFirstCard();

    const remove = screen.getByRole("button", { name: "Remove…" });
    expect(remove).toBeDisabled();
    expect(remove).toHaveAttribute("aria-describedby");
    const reason = document.getElementById(remove.getAttribute("aria-describedby")!);
    expect(reason).toHaveTextContent(reasonText);
  });

  it("keeps Remove enabled, with the reason still described beside it, when the refusal is releasable", async () => {
    const reasonText =
      "sample plot is the project this workspace opens by default; choose a different " +
      "default first, or release it as the default";
    const projectsWithReason = [
      { ...PROJECTS[0], removal_refusal: reasonText, removal_releasable: true },
      PROJECTS[1],
    ];
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: projectsWithReason,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);
    await selectFirstCard();

    const remove = screen.getByRole("button", { name: "Remove…" });
    expect(remove).not.toBeDisabled();
    expect(remove).toHaveAttribute("aria-describedby");
    const reason = document.getElementById(remove.getAttribute("aria-describedby")!);
    expect(reason).toHaveTextContent(reasonText);
  });

  it("enables Remove for a card the backend names no reason for", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);
    await selectFirstCard();

    expect(screen.getByRole("button", { name: "Remove…" })).not.toBeDisabled();
  });

  it("renders the refusal, dependents and external roots before the name field", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue({
      external_roots: [{ path: "/data/external", layouts: ["root"], present: true }],
      dependent_projects: [{ project: "other_project", dataset_id: "ds1" }],
      refusal: "a live run is in progress",
      releasable: false,
    });
    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));

    const dialog = await screen.findByRole("dialog");
    const refusal = await within(dialog).findByText(/a live run is in progress/);
    const dependents = within(dialog).getByText(/other_project/);
    const external = within(dialog).getByText(/external/);
    const nameField = within(dialog).getByLabelText(/type the project name to confirm/i);
    // All three precede the name field in document order (dialog body reads top to bottom).
    expect(
      refusal.compareDocumentPosition(nameField) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      dependents.compareDocumentPosition(nameField) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      external.compareDocumentPosition(nameField) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(nameField).not.toBeDisabled();
  });

  it("keeps the name field enabled and focused with a refusal present, and keeps confirm disabled", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue({
      external_roots: [],
      dependent_projects: [],
      refusal: "a live run is in progress",
      releasable: false,
    });
    render(<ProjectPicker />);
    await selectFirstCard();
    const removeButton = screen.getByRole("button", { name: "Remove…" });
    removeButton.focus();
    fireEvent.click(removeButton);

    const dialog = await screen.findByRole("dialog");
    await within(dialog).findByText(/a live run is in progress/);
    const nameField = within(dialog).getByLabelText(/type the project name to confirm/i);
    expect(nameField).not.toBeDisabled();
    expect(nameField).toHaveFocus();
    expect(within(dialog).getByRole("button", { name: /^Remove$/ })).toBeDisabled();
  });

  it("renders a failed preview as a warning outside the refusal slot, and keeps confirm disabled", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockRejectedValue(new Error("backend unreachable"));
    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));

    const dialog = await screen.findByRole("dialog");
    const warning = await within(dialog).findByText(
      /dependents and refusals could not be checked: backend unreachable\. Close and try again\./,
    );
    expect(within(dialog).queryByText(/^backend unreachable$/)).not.toBeInTheDocument();
    expect(dialog).not.toHaveAttribute("aria-busy");
    const nameField = within(dialog).getByLabelText(/type the project name to confirm/i);
    expect(nameField).not.toBeDisabled();
    fireEvent.change(nameField, { target: { value: PROJECTS[0].name } });
    expect(within(dialog).getByRole("button", { name: /^Remove$/ })).toBeDisabled();
    expect(warning).toBeInTheDocument();
  });

  it("keeps the confirm control disabled until the typed name matches exactly", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue(NO_REFUSAL_PREVIEW);
    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));

    const dialog = await screen.findByRole("dialog");
    const nameField = await within(dialog).findByLabelText(/type the project name to confirm/i);
    const confirm = within(dialog).getByRole("button", { name: /^Remove$/ });
    await waitFor(() => expect(confirm).toBeDisabled());

    fireEvent.change(nameField, { target: { value: PROJECTS[0].name.slice(0, -1) } });
    expect(confirm).toBeDisabled();

    fireEvent.change(nameField, { target: { value: PROJECTS[0].name } });
    expect(confirm).not.toBeDisabled();
  });

  it("keeps confirm disabled while the preview is still pending, even with the exact name typed", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    let resolvePreview: (value: typeof NO_REFUSAL_PREVIEW) => void = () => {};
    vi.mocked(api.projects.removalPreview).mockReturnValue(
      new Promise((resolve) => {
        resolvePreview = resolve;
      }),
    );
    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));

    const dialog = await screen.findByRole("dialog");
    expect(dialog).toHaveAttribute("aria-busy", "true");
    const nameField = within(dialog).getByLabelText(/type the project name to confirm/i);
    fireEvent.change(nameField, { target: { value: PROJECTS[0].name } });
    const confirm = within(dialog).getByRole("button", { name: /^Remove$/ });
    expect(confirm).toBeDisabled();

    resolvePreview(NO_REFUSAL_PREVIEW);
    await waitFor(() => expect(confirm).not.toBeDisabled());
    expect(dialog).not.toHaveAttribute("aria-busy");
  });

  it("scopes the live region to the preview block, not the name field or the buttons", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue({
      external_roots: [{ path: "/data/external", layouts: ["root"], present: true }],
      dependent_projects: [{ project: "other_project", dataset_id: "ds1" }],
      refusal: null,
      releasable: false,
    });
    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));

    const dialog = await screen.findByRole("dialog");
    const dependents = await within(dialog).findByText(/other_project/);
    const liveRegion = dialog.querySelector('[aria-live="polite"]') as HTMLElement;
    expect(liveRegion).toContainElement(dependents);
    const nameField = within(dialog).getByLabelText(/type the project name to confirm/i);
    expect(liveRegion.contains(nameField)).toBe(false);
    const confirm = within(dialog).getByRole("button", { name: /^Remove$/ });
    expect(liveRegion.contains(confirm)).toBe(false);
  });

  it("closes on Escape without sending a request", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue(NO_REFUSAL_PREVIEW);
    render(<ProjectPicker />);
    await selectFirstCard();
    const removeButton = screen.getByRole("button", { name: "Remove…" });
    // fireEvent.click does not move focus the way a real browser click does; a real click on
    // this button focuses it first, which is what the dialog's own opener-tracking relies on.
    removeButton.focus();
    fireEvent.click(removeButton);
    await screen.findByRole("dialog");

    fireEvent.keyDown(window, { key: "Escape" });

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(removeButton).toHaveFocus();
    expect(api.projects.remove).not.toHaveBeenCalled();
  });

  it("renders the request's own refusal in the dialog's refusal slot when the submit itself is refused", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue(NO_REFUSAL_PREVIEW);
    vi.mocked(api.projects.remove).mockRejectedValue(
      new Error("another request already marked it for removal"),
    );
    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));
    const dialog = await screen.findByRole("dialog");
    const nameField = await within(dialog).findByLabelText(/type the project name to confirm/i);
    fireEvent.change(nameField, { target: { value: PROJECTS[0].name } });
    fireEvent.click(within(dialog).getByRole("button", { name: /^Remove$/ }));

    await within(dialog).findByText(/another request already marked it for removal/);
    expect(api.projects.remove).toHaveBeenCalled();
  });

  it("re-fetches the listing behind the dialog on a refused submit", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue(NO_REFUSAL_PREVIEW);
    vi.mocked(api.projects.remove).mockRejectedValue(
      new Error("another request already marked it for removal"),
    );
    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));
    const dialog = await screen.findByRole("dialog");
    const nameField = await within(dialog).findByLabelText(/type the project name to confirm/i);
    fireEvent.change(nameField, { target: { value: PROJECTS[0].name } });
    const listCallsBefore = vi.mocked(api.projects.list).mock.calls.length;
    fireEvent.click(within(dialog).getByRole("button", { name: /^Remove$/ }));

    await within(dialog).findByText(/another request already marked it for removal/);
    await waitFor(() =>
      expect(vi.mocked(api.projects.list).mock.calls.length).toBeGreaterThan(listCallsBefore),
    );
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  it("renders a release button only when the preview says releasable, with its own sentence, calls the release route, toasts, and re-fetches the preview and the listing", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview)
      .mockResolvedValueOnce({
        external_roots: [],
        dependent_projects: [],
        refusal:
          "sample plot is the project this workspace opens by default; choose a " +
          "different default first, or release it as the default",
        releasable: true,
      })
      .mockResolvedValueOnce({
        external_roots: [],
        dependent_projects: [],
        refusal: null,
        releasable: false,
      });
    vi.mocked(api.projects.releaseBinding).mockResolvedValue({
      name: PROJECTS[0].name,
      marker_cleared: true,
      canvas_binding_released: false,
      refusal: null,
      releasable: false,
    });
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));
    const dialog = await screen.findByRole("dialog");

    const releaseButton = await within(dialog).findByRole("button", {
      name: `Release ${PROJECTS[0].name}`,
    });
    await within(dialog).findByText(
      /Stops it opening by default, and forgets it as the GUI's open project if the GUI has it open; nothing else changes\./,
    );

    fireEvent.click(releaseButton);

    await waitFor(() =>
      expect(api.projects.releaseBinding).toHaveBeenCalledWith(
        PROJECTS[0].name,
        expect.any(String),
      ),
    );
    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        `Released ${PROJECTS[0].name} as the default.`,
        "success",
      ),
    );
    await waitFor(() => expect(api.projects.removalPreview).toHaveBeenCalledTimes(2));
    await waitFor(() => expect(api.projects.list).toHaveBeenCalledTimes(2));
  });

  it.each([
    [true, false, `Released ${PROJECTS[0].name} as the default.`, "success"],
    [false, true, `Released ${PROJECTS[0].name} as the open project.`, "success"],
    [
      true, true,
      `Released ${PROJECTS[0].name} as the default and as the open project.`,
      "success",
    ],
    [
      false, false,
      `Nothing to release: ${PROJECTS[0].name} is not the default and the GUI does not have ` +
        "it open.",
      "info",
    ],
  ] as const)(
    "composes the release toast from marker_cleared=%s canvas_binding_released=%s",
    async (markerCleared, canvasBindingReleased, expectedMessage, expectedLevel) => {
      vi.mocked(api.projects.list).mockResolvedValue({
        workspace: "/ws",
        active: null,
        active_path: null,
        projects: PROJECTS,
        pending_removal: [],
        removal_startup_outcomes: [],
      });
      vi.mocked(api.projects.removalPreview).mockResolvedValue({
        external_roots: [],
        dependent_projects: [],
        refusal: "sample plot is the project this workspace opens by default",
        releasable: true,
      });
      vi.mocked(api.projects.releaseBinding).mockResolvedValue({
        name: PROJECTS[0].name,
        marker_cleared: markerCleared,
        canvas_binding_released: canvasBindingReleased,
        refusal: null,
        releasable: false,
      });
      const pushToast = vi.spyOn(useStore.getState(), "pushToast");

      render(<ProjectPicker />);
      await selectFirstCard();
      fireEvent.click(screen.getByRole("button", { name: "Remove…" }));
      const dialog = await screen.findByRole("dialog");
      const releaseButton = await within(dialog).findByRole("button", {
        name: `Release ${PROJECTS[0].name}`,
      });
      fireEvent.click(releaseButton);

      await waitFor(() =>
        expect(pushToast).toHaveBeenCalledWith(expectedMessage, expectedLevel),
      );
    },
  );

  it("renders no release button when the preview says not releasable", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue({
      external_roots: [],
      dependent_projects: [],
      refusal: "a live run is in progress",
      releasable: false,
    });
    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));
    const dialog = await screen.findByRole("dialog");
    await within(dialog).findByText(/a live run is in progress/);
    expect(within(dialog).queryByRole("button", { name: /^Release / })).not.toBeInTheDocument();
  });

  it("ends the toast with the no-project-open sentence when recorded_in_open_project is false", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue(NO_REFUSAL_PREVIEW);
    vi.mocked(api.projects.remove).mockResolvedValue({
      name: PROJECTS[0].name,
      archive_path: "/ws/.removed/x.zip",
      holding_dir: "/ws/.removed/x",
      external_roots: [],
      dependent_projects: [],
      completes: "at the next backend start, or tcip complete-removals",
      audit_scope: "/ws/target",
      recorded_in_open_project: false,
      audit_note:
        "its own line and the route's own line are both in its own log; the archive door's " +
        "own line is under the root $TCIP_STATE_ROOT names.",
    });
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));
    const dialog = await screen.findByRole("dialog");
    const nameField = await within(dialog).findByLabelText(/type the project name to confirm/i);
    fireEvent.change(nameField, { target: { value: PROJECTS[0].name } });
    fireEvent.click(within(dialog).getByRole("button", { name: /^Remove$/ }));

    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.stringContaining(
          `This backend has no project open, so the request is recorded in ${PROJECTS[0].name}'s own log.`,
        ),
        "success",
      ),
    );
  });

  it("renders the card's dependency-warning sentences for both present states and the registry problem", async () => {
    const withWarnings: ProjectSummary[] = [
      {
        ...PROJECTS[0],
        dependency_warnings: [
          {
            dataset_id: "ds1",
            dataset_path: "/ws/target/a",
            target: "sample_plot_target",
            present: true,
            archive_path: "/ws/.removed/sample_plot_target-x.zip",
            holding_dir: "/ws/.removed/sample_plot_target-x",
          },
          {
            dataset_id: "ds2",
            dataset_path: "/ws/target/b",
            target: "sample_plot_gone",
            present: false,
          },
        ],
      },
      { ...PROJECTS[1], dependency_problem: "its dataset registry could not be read: not json" },
    ];
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: withWarnings,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    await screen.findByText(
      /Depends on sample_plot_target, which is pending removal; its images move to the workspace's holding directory at the next backend start\. Register the dataset again from where they are then to clear this\./,
    );
    await screen.findByText(
      /Depends on sample_plot_gone, which is no longer in the workspace\. Register the dataset again from where its images now are to clear this\./,
    );
    expect(screen.queryByText(/dataset ds1/)).not.toBeInTheDocument();
    expect(screen.queryByText(/dataset ds2/)).not.toBeInTheDocument();
    await screen.findByText("its dataset registry could not be read: not json");
  });

  it("renders the card's no-id registry problem verbatim, with no card-side wrapping", async () => {
    const withProblem: ProjectSummary[] = [
      {
        ...PROJECTS[0],
        dependency_problem:
          "its dataset registry has an entry with a path and no id: /ws/target/x",
      },
      PROJECTS[1],
    ];
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: withProblem,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    await screen.findByText(
      "its dataset registry has an entry with a path and no id: /ws/target/x",
    );
  });

  it("renders the dialog's no-id dependent line verbatim after the project's name", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue({
      external_roots: [],
      dependent_projects: [
        {
          project: "other_project",
          unreadable: "its dataset registry has an entry with a path and no id: /ws/target/x",
        },
      ],
      refusal: null,
      releasable: false,
    });
    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));

    const dialog = await screen.findByRole("dialog");
    await within(dialog).findByText(
      "other_project: its dataset registry has an entry with a path and no id: /ws/target/x",
    );
  });

  it("groups a dependent's own multiple datasets under one target into one card sentence", async () => {
    const withWarnings: ProjectSummary[] = [
      {
        ...PROJECTS[0],
        dependency_warnings: [
          {
            dataset_id: "ds1",
            dataset_path: "/ws/target/a",
            target: "sample_plot_target",
            present: true,
          },
          {
            dataset_id: "ds2",
            dataset_path: "/ws/target/b",
            target: "sample_plot_target",
            present: true,
          },
        ],
      },
      PROJECTS[1],
    ];
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: withWarnings,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    await screen.findByText(
      /Depends on sample_plot_target \(2 datasets\), which is pending removal; its images move to the workspace's holding directory at the next backend start\. Register the datasets again from where they are then to clear this\./,
    );
    expect(screen.queryAllByText(/Depends on sample_plot_target/)).toHaveLength(1);
    expect(screen.queryByText(/Register the dataset again/)).not.toBeInTheDocument();
  });

  it("keeps the singular remedy for a group of one dataset, plural for more than one, in the absent case too", async () => {
    const withWarnings: ProjectSummary[] = [
      {
        ...PROJECTS[0],
        dependency_warnings: [
          {
            dataset_id: "ds1",
            dataset_path: "/ws/target/a",
            target: "sample_plot_gone",
            present: false,
          },
          {
            dataset_id: "ds2",
            dataset_path: "/ws/target/b",
            target: "sample_plot_gone",
            present: false,
          },
        ],
      },
      PROJECTS[1],
    ];
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: withWarnings,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    await screen.findByText(
      /Depends on sample_plot_gone \(2 datasets\), which is no longer in the workspace\. Register the datasets again from where its images now are to clear this\./,
    );
  });

  it("on success, toasts at the success level, forgets the recent entry, refetches, and lists the project under pending removal", async () => {
    vi.mocked(api.projects.list)
      .mockResolvedValueOnce({
        workspace: "/ws",
        active: null,
        active_path: null,
        projects: PROJECTS,
        pending_removal: [],
        removal_startup_outcomes: [],
      })
      .mockResolvedValueOnce({
        workspace: "/ws",
        active: null,
        active_path: null,
        projects: [PROJECTS[1]],
        pending_removal: [
          {
            name: PROJECTS[0].name,
            requested_at: "20260304T120000Z",
            archive_path: "/ws/.removed/x.zip",
            holding_dir: "/ws/.removed/x",
          },
        ],
        removal_startup_outcomes: [],
      });
    vi.mocked(api.projects.removalPreview).mockResolvedValue(NO_REFUSAL_PREVIEW);
    vi.mocked(api.projects.remove).mockResolvedValue({
      name: PROJECTS[0].name,
      archive_path: "/ws/.removed/x.zip",
      holding_dir: "/ws/.removed/x",
      external_roots: [],
      dependent_projects: [],
      completes: "at the next backend start, or tcip complete-removals",
      audit_scope: "/ws/other",
      recorded_in_open_project: true,
      audit_note:
        "its own line is in its own log; the route's own line and the archive " +
        "door's own line are in the open project's log.",
    });
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");
    localStorage.setItem(
      "tcip.recent_projects",
      JSON.stringify([{ name: PROJECTS[0].name, path: PROJECTS[0].path }]),
    );

    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));
    const dialog = await screen.findByRole("dialog");
    const nameField = await within(dialog).findByLabelText(/type the project name to confirm/i);
    fireEvent.change(nameField, { target: { value: PROJECTS[0].name } });
    fireEvent.click(within(dialog).getByRole("button", { name: /^Remove$/ }));

    await waitFor(() =>
      expect(api.projects.remove).toHaveBeenCalledWith({
        name: PROJECTS[0].name,
        confirm_name: PROJECTS[0].name,
        user: expect.any(String),
      }),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());
    expect(pushToast).toHaveBeenCalledWith(expect.stringContaining("x.zip"), "success");
    expect(JSON.parse(localStorage.getItem("tcip.recent_projects") ?? "[]")).toEqual([]);
    await screen.findByText(new RegExp(`Pending removal: ${PROJECTS[0].name}`));
    // The card and its own "Remove..." button (the dialog's recorded opener) are both gone from
    // this refetched listing: focus must land on the annotator field, not the body.
    await waitFor(() => expect(screen.getByLabelText("Annotator")).toHaveFocus());
  });

  it("names the dependent count in the success toast when the response lists dependents", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue(NO_REFUSAL_PREVIEW);
    vi.mocked(api.projects.remove).mockResolvedValue({
      name: PROJECTS[0].name,
      archive_path: "/ws/.removed/x.zip",
      holding_dir: "/ws/.removed/x",
      external_roots: [],
      dependent_projects: [
        { project: "dependent_one", dataset_id: "ds1" },
        { project: "dependent_two", dataset_id: "ds2" },
      ],
      completes: "at the next backend start, or tcip complete-removals",
      audit_scope: "/ws/other",
      recorded_in_open_project: true,
      audit_note: "its own line is in its own log.",
    });
    const pushToast = vi.spyOn(useStore.getState(), "pushToast");

    render(<ProjectPicker />);
    await selectFirstCard();
    fireEvent.click(screen.getByRole("button", { name: "Remove…" }));
    const dialog = await screen.findByRole("dialog");
    const nameField = await within(dialog).findByLabelText(/type the project name to confirm/i);
    fireEvent.change(nameField, { target: { value: PROJECTS[0].name } });
    fireEvent.click(within(dialog).getByRole("button", { name: /^Remove$/ }));

    await waitFor(() =>
      expect(pushToast).toHaveBeenCalledWith(
        expect.stringContaining("2 project(s) depend on it; see their cards."),
        "success",
      ),
    );
  });

  it("returns focus to the Remove... control after a release followed by Cancel", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview)
      .mockResolvedValueOnce({
        external_roots: [],
        dependent_projects: [],
        refusal: "sample plot is the project this workspace opens by default",
        releasable: true,
      })
      .mockResolvedValueOnce({
        external_roots: [],
        dependent_projects: [],
        refusal: null,
        releasable: false,
      });
    vi.mocked(api.projects.releaseBinding).mockResolvedValue({
      name: PROJECTS[0].name,
      marker_cleared: true,
      canvas_binding_released: false,
      refusal: null,
      releasable: false,
    });

    render(<ProjectPicker />);
    await selectFirstCard();
    const removeButton = screen.getByRole("button", { name: "Remove…" });
    removeButton.focus();
    fireEvent.click(removeButton);
    const dialog = await screen.findByRole("dialog");
    const releaseButton = await within(dialog).findByRole("button", {
      name: `Release ${PROJECTS[0].name}`,
    });
    fireEvent.click(releaseButton);
    await waitFor(() => expect(api.projects.releaseBinding).toHaveBeenCalled());
    await waitFor(() => expect(api.projects.removalPreview).toHaveBeenCalledTimes(2));

    fireEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(removeButton).toHaveFocus();
  });

  it("lists the last start's removal outcomes", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      removal_startup_outcomes: [
        { name: "moved_one", moved_to: "/ws/.removed/moved_one-x", archive_path: "/ws/a.zip" },
        { name: "blocked_one", blocked_by: "a held database", archive_path: "/ws/b.zip" },
      ],
    });
    render(<ProjectPicker />);
    await screen.findByText(PROJECTS[0].name);

    expect(screen.getByText(/moved_one.*moved to/)).toBeInTheDocument();
    expect(screen.getByText(/blocked_one.*blocked/)).toBeInTheDocument();
  });

  it("names a held-handle block distinctly from a filesystem-boundary block by errno", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [
        {
          name: "held_one",
          requested_at: "20260304T120000Z",
          archive_path: "/ws/a.zip",
          holding_dir: "/ws/.removed/held_one-x",
        },
      ],
      removal_startup_outcomes: [
        { name: "held_one", blocked_by: "in use", blocked_errno: 13, archive_path: "/ws/a.zip" },
        {
          name: "xdev_one",
          blocked_by: "cross-device",
          blocked_errno: 18,
          archive_path: "/ws/b.zip",
        },
      ],
    });
    render(<ProjectPicker />);
    await screen.findByText(PROJECTS[0].name);

    expect(screen.getByText(/held_one.*another process still holds its files/)).toBeInTheDocument();
    expect(screen.getByText(/xdev_one.*cannot move across filesystems/)).toBeInTheDocument();
    // held_one has a blocked outcome from this start: the plain pending line is not shown too.
    expect(screen.queryByText(/^Pending removal: held_one/)).not.toBeInTheDocument();
  });
});

describe("localTime", () => {
  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("renders a compact UTC stamp in the timezone the test fixes", () => {
    vi.stubEnv("TZ", "UTC");
    const expected = new Date(Date.UTC(2026, 2, 4, 12, 0, 0)).toLocaleString();
    expect(localTime("20260304T120000Z")).toBe(expected);
  });

  it("falls back to the raw stamp when it does not parse", () => {
    expect(localTime("not-a-compact-utc-stamp")).toBe("not-a-compact-utc-stamp");
  });
});
