import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

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
        listImages: vi.fn(),
      },
    },
  };
});

import { api } from "@/api/client";
import type { ProjectSummary } from "@/api/client";

const PROJECTS: ProjectSummary[] = [
  {
    name: "hazelnut_catkin_valley-farm",
    path: "/ws/hazelnut_catkin_valley-farm",
    created: 1_700_000_000,
    modified: 1_700_000_500,
    dates: ["2026-02-11", "2026-03-01"],
    subjects: ["catkin", "bush"],
    models: ["baseline"],
    // catkin labelled (+ baseline predicted) on 02-11; bush labelled on 03-01.
    subjects_by_date: { "2026-02-11": ["catkin"], "2026-03-01": ["bush"] },
    models_by_date: { "2026-02-11": ["baseline"], "2026-03-01": [] },
    image_count: 42,
    is_active: false,
  },
  {
    name: "chestnut_burr_site-b",
    path: "/ws/chestnut_burr_site-b",
    created: 1_700_000_000,
    modified: 1_700_000_100,
    dates: ["2026-03-05"],
    subjects: [],
    models: [],
    subjects_by_date: { "2026-03-05": [] },
    models_by_date: { "2026-03-05": [] },
    image_count: 7,
    is_active: false,
  },
];

afterEach(cleanup);
beforeEach(() => {
  // Reset store dataset + the module-level auto-open guard between tests by reloading
  // the module is overkill; instead each test controls `active` so auto-open is inert.
  useStore.getState().clearDataset();
  vi.mocked(api.dataset.select).mockReset();
  vi.mocked(api.projects.setActive).mockResolvedValue({ name: "x", path: "/x" });
});

describe("ProjectPicker", () => {
  it("lists workspace projects with stats", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      projects: PROJECTS,
    });
    render(<ProjectPicker />);

    expect(await screen.findByText("hazelnut_catkin_valley-farm")).toBeInTheDocument();
    expect(screen.getByText("chestnut_burr_site-b")).toBeInTheDocument();
    expect(screen.getByText("42 image(s)")).toBeInTheDocument();
  });

  it("counts each card's dates, subjects and models from that project's own summary", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
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

  it("shows an empty state when there are no projects", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      projects: [],
    });
    render(<ProjectPicker />);
    expect(await screen.findByText("No projects yet")).toBeInTheDocument();
  });

  it("opens a project via /dataset/select with project root = dataset root", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      projects: PROJECTS,
    });
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      selection: {
        project_root: "/ws/hazelnut_catkin_valley-farm",
        dataset_root: "/ws/hazelnut_catkin_valley-farm",
        subject: "catkin",
        date: "2026-03-01",
        image_list: [],
        current_image_index: 0,
        annotations_dir: null,
        predictions_dir: null,
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);

    render(<ProjectPicker />);
    fireEvent.click(await screen.findByText("hazelnut_catkin_valley-farm"));
    // Default date is the most recent ISO date.
    fireEvent.click(screen.getByText("Open project"));

    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    const arg = vi.mocked(api.dataset.select).mock.calls[0][0];
    expect(arg.project_root).toBe("/ws/hazelnut_catkin_valley-farm");
    expect(arg.dataset_root).toBe(arg.project_root);
    expect(arg.date).toBe("2026-03-01");
    // Store now holds the selection (dataset ready).
    await waitFor(() =>
      expect(useStore.getState().gui.dataset.dataset_root).toBe("/ws/hazelnut_catkin_valley-farm"),
    );
  });

  it("filters the subject options to the selected date's labelled subjects", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      projects: PROJECTS,
    });
    render(<ProjectPicker />);
    fireEvent.click(await screen.findByText("hazelnut_catkin_valley-farm"));

    // Default date is the most recent ISO date (2026-03-01), where only 'bush' is labelled.
    const subjectSelect = screen.getByLabelText("Subject") as HTMLSelectElement;
    let opts = Array.from(subjectSelect.options).map((o) => o.value);
    expect(opts).toContain("bush");
    expect(opts).not.toContain("catkin");

    // Switch to 2026-02-11, where only 'catkin' is labelled.
    fireEvent.change(screen.getByLabelText("Date"), { target: { value: "2026-02-11" } });
    opts = Array.from(subjectSelect.options).map((o) => o.value);
    expect(opts).toContain("catkin");
    expect(opts).not.toContain("bush");
  });

  it("has no advanced folder-open escape hatch (project creation is agent-driven)", async () => {
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/ws",
      active: null,
      projects: PROJECTS,
    });
    render(<ProjectPicker />);
    await screen.findByText(PROJECTS[0].name);
    expect(screen.queryByText(/open a folder outside the workspace/i)).not.toBeInTheDocument();
  });
});
