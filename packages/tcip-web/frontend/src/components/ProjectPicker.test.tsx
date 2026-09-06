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
});

describe("ProjectPicker", () => {
  it("lists workspace projects with stats", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
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

  it("has no advanced folder-open escape hatch (project creation is agent-driven)", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      active_path: null,
      projects: PROJECTS,
    });
    render(<ProjectPicker />);
    await screen.findByText(PROJECTS[0].name);
    expect(screen.queryByText(/open a folder outside the workspace/i)).not.toBeInTheDocument();
  });
});
