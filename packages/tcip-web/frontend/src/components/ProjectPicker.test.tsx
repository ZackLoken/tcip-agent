import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { ProjectPicker } from "@/components/ProjectPicker";
import { useStore } from "@/store";

vi.mock("@/api/client", () => {
  return {
    api: {
      projects: {
        list: vi.fn(),
        open: vi.fn(),
        remove: vi.fn(),
        rename: vi.fn(),
      },
      dataset: {
        select: vi.fn(),
      },
    },
  };
});

import { api } from "@/api/client";
import type { ProjectSummary } from "@/api/client";

const PROJECTS: ProjectSummary[] = [
  {
    id: "a1b2c3d4e5f6",
    display_name: "Valley farm",
    site: "north orchard",
    record_problem: null,
    path: "/ws/valley",
    created: 1_700_000_000,
    modified: 1_700_000_500,
    dates: ["2026-02-11", "2026-03-01"],
    subjects: ["subject_a", "bush"],
    models: ["baseline"],
    // subject_a labeled (+ baseline predicted) on 02-11; bush labeled on 03-01.
    subjects_by_date: { "2026-02-11": ["subject_a"], "2026-03-01": ["bush"] },
    models_by_date: { "2026-02-11": ["baseline"], "2026-03-01": [] },
    image_count: 42,
    is_open: false,
    label_problem: null,
  },
  {
    id: null,
    display_name: null,
    site: null,
    record_problem: "/ws/stray is not a project yet: create it with initialize_project",
    path: "/ws/stray",
    created: 1_700_000_000,
    modified: 1_700_000_100,
    dates: [],
    subjects: [],
    models: [],
    subjects_by_date: {},
    models_by_date: {},
    image_count: 0,
    is_open: false,
    label_problem: null,
  },
];

function listing(overrides: Partial<Awaited<ReturnType<typeof api.projects.list>>> = {}) {
  return {
    workspace: "/ws",
    open_id: null,
    last_opened_problem: null,
    projects: PROJECTS,
    ...overrides,
  };
}

function selection(root: string) {
  return {
    status: "ok",
    selection: {
      dataset_root: root,
      subject: "subject_a",
      date: "2026-03-01",
      model_name: null,
      image_list: [],
      current_image_index: 0,
      images_dir: null,
      annotations_dir: null,
      predictions_dir: null,
      label_paths: {},
      prediction_paths: {},
    },
  };
}

afterEach(cleanup);
beforeEach(() => {
  useStore.getState().clearDataset();
  useStore.setState({ openProject: null });
  vi.mocked(api.projects.list).mockReset();
  vi.mocked(api.projects.open).mockReset();
  vi.mocked(api.projects.remove).mockReset();
  vi.mocked(api.projects.rename).mockReset();
  vi.mocked(api.dataset.select).mockReset();
});

describe("ProjectPicker", () => {
  it("lists each project by the display name its own record holds, with its stats", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    render(<ProjectPicker />);

    expect(await screen.findByText("Valley farm")).toBeInTheDocument();
    expect(screen.getByText("north orchard")).toBeInTheDocument();
    expect(screen.getByText("42 image(s)")).toBeInTheDocument();
    expect(screen.getByText("2 dates")).toBeInTheDocument();
  });

  it("lists a directory whose record does not read by its path and the reason, with no open", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    render(<ProjectPicker />);

    expect(
      await screen.findByText("/ws/stray is not a project yet: create it with initialize_project"),
    ).toBeInTheDocument();
    expect(screen.getByText("/ws/stray")).toBeInTheDocument();
  });

  it("names a last-opened project the workspace no longer holds", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(
      listing({ last_opened_problem: "no project in the workspace /ws has id 'gone'" }),
    );
    render(<ProjectPicker />);

    expect(
      await screen.findByText(/no project in the workspace \/ws has id 'gone'/),
    ).toBeInTheDocument();
  });

  it("opens a project by its id, then selects a dataset inside its own directory", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    vi.mocked(api.projects.open).mockResolvedValue({
      id: "a1b2c3d4e5f6",
      display_name: "Valley farm",
      path: "/ws/valley",
    });
    vi.mocked(api.dataset.select).mockResolvedValue(selection("/ws/valley"));
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText("Valley farm"));
    fireEvent.click(screen.getByText("Open project"));

    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(api.projects.open).toHaveBeenCalledWith("a1b2c3d4e5f6");
    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      dataset_root: "/ws/valley",
      date: "2026-03-01",
    });
    expect(useStore.getState().openProject).toEqual({ id: "a1b2c3d4e5f6", path: "/ws/valley" });
  });

  it("removes a project by its id once its display name is typed", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    vi.mocked(api.projects.remove).mockResolvedValue({
      archive_path: "/ws/.removed/valley-20260929T000000Z.zip",
      moved_to: "/ws/.removed/valley-20260929T000000Z",
    });
    useStore.setState({ user: "grower" });
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText("Valley farm"));
    fireEvent.click(screen.getByText("Remove…"));
    const confirm = screen.getByRole("button", { name: "Remove" });
    expect(confirm).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Type the project name to confirm"), {
      target: { value: "Valley farm" },
    });
    fireEvent.click(confirm);

    await waitFor(() => expect(api.projects.remove).toHaveBeenCalledTimes(1));
    expect(api.projects.remove).toHaveBeenCalledWith({
      id: "a1b2c3d4e5f6",
      confirm_name: "Valley farm",
      user: "grower",
    });
  });

  it("renames a project's display name by its id", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    vi.mocked(api.projects.rename).mockResolvedValue({
      id: "a1b2c3d4e5f6",
      display_name: "Valley farm east",
      previous_display_name: "Valley farm",
    });
    useStore.setState({ user: "grower" });
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText("Valley farm"));
    fireEvent.click(screen.getByText("Rename…"));
    fireEvent.change(screen.getByLabelText("New name"), {
      target: { value: "Valley farm east" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rename" }));

    await waitFor(() => expect(api.projects.rename).toHaveBeenCalledTimes(1));
    expect(api.projects.rename).toHaveBeenCalledWith({
      id: "a1b2c3d4e5f6",
      display_name: "Valley farm east",
      user: "grower",
    });
  });
});
