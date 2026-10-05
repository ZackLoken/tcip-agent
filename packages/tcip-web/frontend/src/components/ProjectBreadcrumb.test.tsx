import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { ProjectBreadcrumb } from "@/components/ProjectBreadcrumb";
import { useStore } from "@/store";

vi.mock("@/api/client", () => {
  return {
    api: {
      projects: {
        list: vi.fn(),
        open: vi.fn(),
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
const CURRENT_ROW = MARKER + " Alpha block";

function summary(id: string, displayName: string): ProjectSummary {
  return {
    id,
    display_name: displayName,
    site: "north orchard",
    record_problem: null,
    path: `/w/${id}`,
    created: 1,
    modified: 2,
    dates: ["2026-01-01"],
    subjects: ["subject_a"],
    subjects_by_date: { "2026-01-01": ["subject_a"] },
    buckets_by_date: { "2026-01-01": [] },
    image_count: 1,
    is_open: false,
    label_problem: null,
  };
}

function selection(root: string) {
  return {
    status: "ok",
    selection: {
      dataset_root: root,
      subject: "subject_a",
      date: "2026-01-01",
      image_list: [],
      current_image_index: 0,
      images_dir: null,
      bucket: null,
    },
  };
}

function openOn(id: string, date: string | null = "2026-01-01") {
  useStore.setState((st) => ({
    gui: {
      ...st.gui,
      dataset: {
        ...st.gui.dataset,
        dataset_root: `/w/${id}`,
        subject: "subject_a",
        date,
        image_list: [],
        current_image_index: 0,
        bucket: null,
      },
    },
    openProject: { id, path: `/w/${id}` },
  }));
}

afterEach(cleanup);
beforeEach(() => {
  localStorage.removeItem("tcip.recent_projects");
  vi.mocked(api.projects.list).mockReset();
  vi.mocked(api.projects.open).mockReset();
  vi.mocked(api.dataset.select).mockReset();
  vi.mocked(api.projects.list).mockResolvedValue({
    workspace: "/w",
    open_id: "a1",
    last_opened_problem: null,
    projects: [summary("a1", "Alpha block"), summary("b2", "Beta block")],
  });
  openOn("a1");
});

describe("recent-projects menu", () => {
  it("lists the open project, marked as current", async () => {
    localStorage.setItem("tcip.recent_projects", JSON.stringify(["a1"]));
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    expect(await screen.findByText(CURRENT_ROW)).toBeInTheDocument();
    expect(screen.queryByText(/no recent projects/i)).not.toBeInTheDocument();
  });

  it("clicking the current project just closes the menu, opening nothing", async () => {
    localStorage.setItem("tcip.recent_projects", JSON.stringify(["a1"]));
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    fireEvent.click(await screen.findByText(CURRENT_ROW));
    expect(screen.queryByText(CURRENT_ROW)).not.toBeInTheDocument();
    expect(api.projects.open).not.toHaveBeenCalled();
  });

  it("another recent project opens by its id, then selects inside its own directory", async () => {
    localStorage.setItem("tcip.recent_projects", JSON.stringify(["a1", "b2"]));
    vi.mocked(api.projects.open).mockResolvedValue({
      id: "b2",
      display_name: "Beta block",
      path: "/w/b2",
    });
    vi.mocked(api.dataset.select).mockResolvedValue(selection("/w/b2"));
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    fireEvent.click(await screen.findByText("Beta block"));
    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(api.projects.open).toHaveBeenCalledWith("b2");
    expect(vi.mocked(api.dataset.select).mock.calls[0][0].dataset_root).toBe("/w/b2");
    expect(useStore.getState().openProject).toEqual({ id: "b2", path: "/w/b2" });
  });
});

describe("footer open state", () => {
  it("names the open project by its display name, and shows no dated images when it has no date", async () => {
    openOn("a1", null);
    render(<ProjectBreadcrumb />);
    expect(await screen.findByText("Alpha block")).toBeInTheDocument();
    expect(screen.getByText("no dated images")).toBeInTheDocument();
    expect(screen.queryByTitle("Switch date")).not.toBeInTheDocument();
  });

  it("carries the no-date explanation as visually hidden text in the reading order, not only a title", () => {
    openOn("a1", null);
    render(<ProjectBreadcrumb />);
    expect(
      screen.getByText(
        "This project has no dated images yet; ask the agent to ingest images first.",
      ),
    ).toBeInTheDocument();
  });

  it("reads no project open only once nothing is open", () => {
    useStore.setState((st) => ({
      gui: { ...st.gui, dataset: { ...st.gui.dataset, dataset_root: null, date: null } },
      openProject: null,
    }));
    render(<ProjectBreadcrumb />);
    expect(screen.getByText("no project open")).toBeInTheDocument();
  });
});

describe("switching date", () => {
  it("reopens the same project on the chosen date", async () => {
    vi.mocked(api.projects.open).mockResolvedValue({
      id: "a1",
      display_name: "Alpha block",
      path: "/w/a1",
    });
    vi.mocked(api.dataset.select).mockResolvedValue(selection("/w/a1"));
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/w",
      open_id: "a1",
      last_opened_problem: null,
      projects: [{ ...summary("a1", "Alpha block"), dates: ["2026-01-01", "2026-02-02"] }],
    });
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Switch date"));
    fireEvent.click(await screen.findByText("2026-02-02"));
    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(api.projects.open).toHaveBeenCalledWith("a1");
    expect(vi.mocked(api.dataset.select).mock.calls[0][0].date).toBe("2026-02-02");
  });
});
