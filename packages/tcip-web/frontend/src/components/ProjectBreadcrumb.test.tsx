import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { ProjectBreadcrumb } from "@/components/ProjectBreadcrumb";
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
      },
    },
  };
});

import { api } from "@/api/client";
import type { ProjectSummary } from "@/api/client";

// The current-project row marker (the breadcrumb's active-row glyph, U+25CF).
const MARKER = String.fromCharCode(0x25cf);
const CURRENT_ROW = MARKER + " alpha";

function summary(name: string): ProjectSummary {
  return {
    name,
    path: `/w/${name}`,
    created: 1,
    modified: 2,
    dates: ["2026-01-01"],
    subjects: ["subject_a"],
    models: [],
    subjects_by_date: { "2026-01-01": ["subject_a"] },
    models_by_date: { "2026-01-01": [] },
    image_count: 1,
    is_active: false,
    site: "north orchard",
    site_problem: null,
    label_problem: null,
  };
}

function openOn(name: string) {
  useStore.setState((st) => ({
    gui: {
      ...st.gui,
      dataset: {
        ...st.gui.dataset,
        project_root: `/w/${name}`,
        dataset_root: `/w/${name}`,
        subject: "subject_a",
        date: "2026-01-01",
        image_list: [],
        current_image_index: 0,
        annotations_dir: null,
        predictions_dir: null,
      },
    },
  }));
}

afterEach(cleanup);
beforeEach(() => {
  localStorage.removeItem("tcip.recent_projects");
  vi.mocked(api.projects.list).mockReset();
  vi.mocked(api.dataset.select).mockReset();
  vi.mocked(api.projects.setActive).mockReset();
  vi.mocked(api.projects.setActive).mockResolvedValue({ name: "x", path: "/x" });
  vi.mocked(api.projects.list).mockResolvedValue({
    workspace: "/w",
    active: null,
    active_path: null,
    projects: [summary("alpha"), summary("beta")],
  });
  openOn("alpha");
});

describe("recent-projects menu", () => {
  it("lists the just-opened project, marked as current", async () => {
    localStorage.setItem(
      "tcip.recent_projects",
      JSON.stringify([{ name: "alpha", path: "/w/alpha" }]),
    );
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    expect(await screen.findByText(CURRENT_ROW)).toBeInTheDocument();
    expect(screen.queryByText(/no recent projects/i)).not.toBeInTheDocument();
  });

  it("clicking the current project just closes the menu, opening nothing", async () => {
    localStorage.setItem(
      "tcip.recent_projects",
      JSON.stringify([{ name: "alpha", path: "/w/alpha" }]),
    );
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    fireEvent.click(await screen.findByText(CURRENT_ROW));
    expect(screen.queryByText(CURRENT_ROW)).not.toBeInTheDocument();
    expect(api.dataset.select).not.toHaveBeenCalled();
  });

  it("another recent project opens through the dataset select", async () => {
    localStorage.setItem(
      "tcip.recent_projects",
      JSON.stringify([
        { name: "alpha", path: "/w/alpha" },
        { name: "beta", path: "/w/beta" },
      ]),
    );
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      generation: 1,
      selection: {
        project_root: "/w/beta",
        dataset_root: "/w/beta",
        subject: "subject_a",
        date: "2026-01-01",
        image_list: [],
        current_image_index: 0,
        annotations_dir: null,
        predictions_dir: null,
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    fireEvent.click(await screen.findByText("beta"));
    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.dataset.select).mock.calls[0][0].project_root).toBe("/w/beta");
    // Opening a recent project is a human-initiated adoption: the marker gets written.
    expect(api.projects.setActive).toHaveBeenCalledWith("beta");
  });
});

describe("switching date", () => {
  it("does not write the active-project marker (not an adoption)", async () => {
    vi.mocked(api.dataset.select).mockResolvedValue({
      status: "ok",
      generation: 1,
      selection: {
        project_root: "/w/alpha",
        dataset_root: "/w/alpha",
        subject: "subject_a",
        date: "2026-01-01",
        image_list: [],
        current_image_index: 0,
        annotations_dir: null,
        predictions_dir: null,
      },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
    } as any);
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/w",
      active: null,
      active_path: null,
      projects: [{ ...summary("alpha"), dates: ["2026-01-01", "2026-02-02"] }],
    });
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Switch date"));
    fireEvent.click(await screen.findByText("2026-02-02"));
    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(api.projects.setActive).not.toHaveBeenCalled();
  });
});
