import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import { ProjectPicker } from "@/components/ProjectPicker";
import { useStore } from "@/store";

vi.mock("@/api/client", () => {
  return {
    api: {
      projects: {
        list: vi.fn(),
        setActive: vi.fn().mockResolvedValue({ name: "x", path: "/x" }),
        removalPreview: vi.fn(),
        remove: vi.fn(),
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
});

const NO_OPEN_PROJECT = { marker: null, platform_root: null, canvas_binding: null };
const NO_REFUSAL_PREVIEW = { external_roots: [], dependent_projects: [], refusal: null };

describe("ProjectPicker", () => {
  it("lists workspace projects with stats", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);

    const card = await screen.findByRole("button", { name: "crop_a_subject_a_valley-farm" });
    expect(card).toHaveAccessibleName("crop_a_subject_a_valley-farm");
    expect(card).toHaveAccessibleDescription(/north orchard/);
    expect(card).toHaveAccessibleDescription(/42 image\(s\)/);
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
      open_project_names: { marker: null, platform_root: null, canvas_binding: null },
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
    "sample plot is the project this workspace opens by default; choose a different default first",
    "sample plot is the project this backend started on; restart on a different project first",
    "sample plot is the project the GUI has open; open a different project first",
  ])("disables Remove with the backend's own reason (%s)", async (reasonText) => {
    const projectsWithReason = [{ ...PROJECTS[0], removal_refusal: reasonText }, PROJECTS[1]];
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: projectsWithReason,
      pending_removal: [],
      open_project_names: NO_OPEN_PROJECT,
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

  it("disables every card's Remove control when no project is open in the backend", async () => {
    const reasonText =
      "no project is open in this backend; open one first so the request is recorded in its log";
    const projectsWithReason = PROJECTS.map((p) => ({ ...p, removal_refusal: reasonText }));
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: projectsWithReason,
      pending_removal: [],
      open_project_names: NO_OPEN_PROJECT,
      removal_startup_outcomes: [],
    });
    render(<ProjectPicker />);
    await selectFirstCard();

    expect(screen.getByRole("button", { name: "Remove…" })).toBeDisabled();
  });

  it("enables Remove for a card the backend names no reason for", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      open_project_names: NO_OPEN_PROJECT,
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
      open_project_names: NO_OPEN_PROJECT,
      removal_startup_outcomes: [],
    });
    vi.mocked(api.projects.removalPreview).mockResolvedValue({
      external_roots: [{ path: "/data/external", layouts: ["root"], present: true }],
      dependent_projects: [{ project: "other_project", dataset_id: "ds1" }],
      refusal: "a live run is in progress",
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
    expect(nameField).toBeDisabled();
  });

  it("keeps the confirm control disabled until the typed name matches exactly", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      open_project_names: NO_OPEN_PROJECT,
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
      open_project_names: NO_OPEN_PROJECT,
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

  it("closes on Escape without sending a request", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      open_project_names: NO_OPEN_PROJECT,
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
      open_project_names: NO_OPEN_PROJECT,
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

  it("on success, toasts at the success level, forgets the recent entry, refetches, and lists the project under pending removal", async () => {
    vi.mocked(api.projects.list)
      .mockResolvedValueOnce({
        workspace: "/ws",
        active: null,
        active_path: null,
        projects: PROJECTS,
        pending_removal: [],
        open_project_names: NO_OPEN_PROJECT,
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
        open_project_names: NO_OPEN_PROJECT,
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
  });

  it("lists the last start's removal outcomes", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
      pending_removal: [],
      open_project_names: NO_OPEN_PROJECT,
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
      open_project_names: NO_OPEN_PROJECT,
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
