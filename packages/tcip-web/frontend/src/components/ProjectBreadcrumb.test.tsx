import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api, type ProjectSummary } from "@/api/client";
import type { OpenAnswer } from "@/api/types.generated";
import { ProjectBreadcrumb } from "@/components/ProjectBreadcrumb";
import { openWithDefaults } from "@/lib/openProject";
import { useStore } from "@/store";
import { INVARIANCES } from "@/test/projectRecords";
import { openAnswer, selectAnswer, selected, stateEnvelope, testProject } from "@/test/store";

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

// The current-project row marker (the breadcrumb's active-row glyph, U+25CF).
const MARKER = String.fromCharCode(0x25cf);
const CURRENT_ROW = MARKER + " Alpha block";

function summary(id: string, displayName: string): ProjectSummary {
  return {
    id,
    display_name: displayName,
    site: "north orchard",
    invariances: INVARIANCES,
    record_problem: null,
    path: `/w/${id}`,
    created: 1,
    modified: 2,
    dates: ["2026-01-01"],
    subjects: ["subject_a"],
    subjects_by_date: { "2026-01-01": ["subject_a"] },
    buckets_by_date: { "2026-01-01": [] },
    image_count: 1,
    label_problem: null,
  };
}

/** The open route's answer for the listed project `id`: the produced answer with that project's
 *  id, directory and display name. */
function opened(id: string, displayName: string): OpenAnswer {
  return openAnswer({ id, path: `/w/${id}`, display_name: displayName });
}

const SELECT_ANSWER = selectAnswer(selected({ subject: "subject_a" }));

/** Open the project `id` on the produced capture of photographs, or on none. */
function openOn(id: string, captured = true) {
  useStore.setState((st) => ({
    gui: { ...st.gui, dataset: captured ? selected({ subject: "subject_a" }) : null },
    openProject: testProject({ id, path: `/w/${id}` }),
  }));
}

afterEach(cleanup);
beforeEach(() => {
  localStorage.removeItem("tcip.recent_projects");
  useStore.getState().setUser("grower");
  useStore.getState().patchOpenStatus({ opening: null, openError: null });
  vi.mocked(api.projects.list).mockReset();
  vi.mocked(api.projects.open).mockReset();
  vi.mocked(api.dataset.select).mockReset();
  vi.mocked(api.projects.list).mockResolvedValue({
    workspace: "/w",
    open_id: "a1",
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
    vi.mocked(api.projects.open).mockResolvedValue(opened("b2", "Beta block"));
    vi.mocked(api.dataset.select).mockResolvedValue(SELECT_ANSWER);
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    fireEvent.click(await screen.findByText("Beta block"));
    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(api.projects.open).toHaveBeenCalledWith({ id: "b2", user: "grower" });
    expect(vi.mocked(api.dataset.select).mock.calls[0][0].dataset_root).toBe("/w/b2");
    expect(useStore.getState().openProject).toEqual(opened("b2", "Beta block").project);
  });
});

describe("recent-projects menu through the one opening transition", () => {
  it("starts no open while another open is in flight, and disables the menu", async () => {
    useStore.getState().patchOpenStatus({ opening: { requestId: 99, projectId: "zz" } });
    render(<ProjectBreadcrumb />);

    expect(screen.getByTitle("Recent projects")).toBeDisabled();
    await openWithDefaults({ ...summary("b2", "Beta block"), id: "b2" });
    expect(api.projects.open).not.toHaveBeenCalled();
  });

  it("opens a recent project from the listing it already holds, without listing again", async () => {
    localStorage.setItem("tcip.recent_projects", JSON.stringify(["a1", "b2"]));
    let listingsBeforeOpen = -1;
    vi.mocked(api.projects.open).mockImplementation(() => {
      listingsBeforeOpen = vi.mocked(api.projects.list).mock.calls.length;
      return Promise.resolve(opened("b2", "Beta block"));
    });
    vi.mocked(api.dataset.select).mockResolvedValue(SELECT_ANSWER);
    render(<ProjectBreadcrumb />);
    await waitFor(() => expect(api.projects.list).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByTitle("Recent projects"));
    fireEvent.click(await screen.findByText("Beta block"));

    await waitFor(() =>
      expect(useStore.getState().openProject).toEqual(opened("b2", "Beta block").project),
    );
    expect(listingsBeforeOpen).toBe(1);
  });

  it("records and toasts a failed open for the project it was started for", async () => {
    localStorage.setItem("tcip.recent_projects", JSON.stringify(["a1", "b2"]));
    vi.mocked(api.projects.open).mockRejectedValue(new Error("beta refused"));
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    fireEvent.click(await screen.findByText("Beta block"));

    await waitFor(() => expect(useStore.getState().openError?.message).toContain("beta refused"));
    expect(useStore.getState().openError?.projectId).toBe("b2");
    expect(useStore.getState().toasts.map((t) => t.message)).toContainEqual(
      expect.stringContaining("beta refused"),
    );
  });

  it("drops a date switch's result once a snapshot adopted another project", async () => {
    const open = Promise.withResolvers<OpenAnswer>();
    vi.mocked(api.projects.open).mockReturnValue(open.promise);
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/w",
      open_id: "a1",
      projects: [{ ...summary("a1", "Alpha block"), dates: ["2026-01-01", "2026-02-02"] }],
    });
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Switch date"));
    fireEvent.click(await screen.findByText("2026-02-02"));
    await waitFor(() => expect(api.projects.open).toHaveBeenCalled());
    act(() =>
      useStore
        .getState()
        .mergeSnapshot(stateEnvelope({ version: 1, project: opened("b2", "Beta block").project })),
    );

    await act(() => Promise.resolve(open.resolve(opened("a1", "Alpha block"))));

    expect(api.dataset.select).not.toHaveBeenCalled();
    expect(useStore.getState().openProject).toEqual(opened("b2", "Beta block").project);
  });
});

describe("footer open state", () => {
  it("names the open project by its display name, and shows no dated images when it has no date", async () => {
    openOn("a1", false);
    render(<ProjectBreadcrumb />);
    expect(await screen.findByText("Alpha block")).toBeInTheDocument();
    expect(screen.getByText("No dated images")).toBeInTheDocument();
    expect(screen.queryByTitle("Switch date")).not.toBeInTheDocument();
  });

  it("reads Loading in place of the project name until the listing answers", () => {
    openOn("a1", false);
    render(<ProjectBreadcrumb />);
    expect(screen.getByRole("button", { name: "Loading…" })).toBeInTheDocument();
  });

  it("reads no project open only once nothing is open", () => {
    useStore.setState((st) => ({ gui: { ...st.gui, dataset: null }, openProject: null }));
    render(<ProjectBreadcrumb />);
    expect(screen.getByText("no project open")).toBeInTheDocument();
  });
});

describe("switching date", () => {
  it("reopens the same project on the chosen date", async () => {
    vi.mocked(api.projects.open).mockResolvedValue(opened("a1", "Alpha block"));
    vi.mocked(api.dataset.select).mockResolvedValue(SELECT_ANSWER);
    vi.mocked(api.projects.list).mockResolvedValue({
      workspace: "/w",
      open_id: "a1",
      projects: [{ ...summary("a1", "Alpha block"), dates: ["2026-01-01", "2026-02-02"] }],
    });
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Switch date"));
    fireEvent.click(await screen.findByText("2026-02-02"));
    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(api.projects.open).toHaveBeenCalledWith({ id: "a1", user: "grower" });
    expect(vi.mocked(api.dataset.select).mock.calls[0][0].date).toBe("2026-02-02");
  });
});
