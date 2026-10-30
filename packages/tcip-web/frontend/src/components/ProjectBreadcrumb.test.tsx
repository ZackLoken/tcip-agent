import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import type { OpenAnswer } from "@/api/types.generated";
import { ProjectBreadcrumb } from "@/components/ProjectBreadcrumb";
import { type OpenableProject, openWithDefaults } from "@/lib/openProject";
import { useStore } from "@/store";
import { openedAs, PICKER, PROJECT_FIXTURES } from "@/test/projectRecords";
import {
  SELECTED_SUBJECT as SUBJECT,
  selected,
  stateEnvelope,
  SUBJECT_SELECT_ANSWER,
} from "@/test/store";

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

const { valley: VALLEY, ridge: RIDGE } = PICKER;

// The current-project row marker (the breadcrumb's active-row glyph, U+25CF).
const MARKER = String.fromCharCode(0x25cf);
const CURRENT_ROW = `${MARKER} ${VALLEY.record.display_name}`;

/** A date of Valley farm's other than the produced selection's, holding {@link SUBJECT} after
 *  another subject. */
const [{ name: LATER }] = VALLEY.tree.captures;

const SELECT_ANSWER = SUBJECT_SELECT_ANSWER;

/** Open the listed project `p` on the produced capture of photographs, or on none. */
function openOn(p: OpenableProject, captured = true) {
  useStore.setState((st) => ({
    gui: { ...st.gui, dataset: captured ? selected({ subject: SUBJECT }) : null },
    openProject: openedAs(p).project,
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
  vi.mocked(api.projects.list).mockResolvedValue(PROJECT_FIXTURES.picker_open);
  openOn(VALLEY);
});

/** Hold the listed projects `projects` as the recent ones, newest first. */
function recent(...projects: OpenableProject[]) {
  localStorage.setItem("tcip.recent_projects", JSON.stringify(projects.map((p) => p.record.id)));
}

describe("recent-projects menu", () => {
  it("lists the open project, marked as current", async () => {
    recent(VALLEY);
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    expect(await screen.findByText(CURRENT_ROW)).toBeInTheDocument();
    expect(screen.queryByText(/no recent projects/i)).not.toBeInTheDocument();
  });

  it("clicking the current project just closes the menu, opening nothing", async () => {
    recent(VALLEY);
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    fireEvent.click(await screen.findByText(CURRENT_ROW));
    expect(screen.queryByText(CURRENT_ROW)).not.toBeInTheDocument();
    expect(api.projects.open).not.toHaveBeenCalled();
  });

  it("another recent project opens by its id, then selects inside its own directory", async () => {
    recent(VALLEY, RIDGE);
    vi.mocked(api.projects.open).mockResolvedValue(openedAs(RIDGE));
    vi.mocked(api.dataset.select).mockResolvedValue(SELECT_ANSWER);
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    fireEvent.click(await screen.findByText(RIDGE.record.display_name));
    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(api.projects.open).toHaveBeenCalledWith({ id: RIDGE.record.id, user: "grower" });
    expect(vi.mocked(api.dataset.select).mock.calls[0][0].dataset_root).toBe(
      RIDGE.tree.dataset_root,
    );
    expect(useStore.getState().openProject).toEqual(openedAs(RIDGE).project);
  });
});

describe("recent-projects menu through the one opening transition", () => {
  it("starts no open while another open is in flight, and disables the menu", async () => {
    useStore.getState().patchOpenStatus({ opening: { requestId: 99, projectId: "zz" } });
    render(<ProjectBreadcrumb />);

    expect(screen.getByTitle("Recent projects")).toBeDisabled();
    await openWithDefaults(RIDGE);
    expect(api.projects.open).not.toHaveBeenCalled();
  });

  it("opens a recent project from the listing it already holds, without listing again", async () => {
    recent(VALLEY, RIDGE);
    let listingsBeforeOpen = -1;
    vi.mocked(api.projects.open).mockImplementation(() => {
      listingsBeforeOpen = vi.mocked(api.projects.list).mock.calls.length;
      return Promise.resolve(openedAs(RIDGE));
    });
    vi.mocked(api.dataset.select).mockResolvedValue(SELECT_ANSWER);
    render(<ProjectBreadcrumb />);
    await waitFor(() => expect(api.projects.list).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByTitle("Recent projects"));
    fireEvent.click(await screen.findByText(RIDGE.record.display_name));

    await waitFor(() => expect(useStore.getState().openProject).toEqual(openedAs(RIDGE).project));
    expect(listingsBeforeOpen).toBe(1);
  });

  it("records and toasts a failed open for the project it was started for", async () => {
    recent(VALLEY, RIDGE);
    vi.mocked(api.projects.open).mockRejectedValue(new Error("ridge refused"));
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Recent projects"));
    fireEvent.click(await screen.findByText(RIDGE.record.display_name));

    await waitFor(() => expect(useStore.getState().openError?.message).toContain("ridge refused"));
    expect(useStore.getState().openError?.projectId).toBe(RIDGE.record.id);
    expect(useStore.getState().toasts.map((t) => t.message)).toContainEqual(
      expect.stringContaining("ridge refused"),
    );
  });

  it("drops a date switch's result once a snapshot adopted another project", async () => {
    const open = Promise.withResolvers<OpenAnswer>();
    vi.mocked(api.projects.open).mockReturnValue(open.promise);
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Switch date"));
    fireEvent.click(await screen.findByText(LATER));
    await waitFor(() => expect(api.projects.open).toHaveBeenCalled());
    act(() =>
      useStore
        .getState()
        .mergeSnapshot(stateEnvelope({ version: 1, project: openedAs(RIDGE).project })),
    );

    await act(() => Promise.resolve(open.resolve(openedAs(VALLEY))));

    expect(api.dataset.select).not.toHaveBeenCalled();
    expect(useStore.getState().openProject).toEqual(openedAs(RIDGE).project);
  });
});

describe("footer open state", () => {
  it("names the open project by its display name, and shows no dated images when it has no date", async () => {
    openOn(VALLEY, false);
    render(<ProjectBreadcrumb />);
    expect(await screen.findByText(VALLEY.record.display_name)).toBeInTheDocument();
    expect(screen.getByText("No dated images")).toBeInTheDocument();
    expect(screen.queryByTitle("Switch date")).not.toBeInTheDocument();
  });

  it("reads Loading in place of the project name until the listing answers", () => {
    openOn(VALLEY, false);
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
    vi.mocked(api.projects.open).mockResolvedValue(openedAs(VALLEY));
    vi.mocked(api.dataset.select).mockResolvedValue(SELECT_ANSWER);
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Switch date"));
    fireEvent.click(await screen.findByText(LATER));
    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(api.projects.open).toHaveBeenCalledWith({ id: VALLEY.record.id, user: "grower" });
    expect(vi.mocked(api.dataset.select).mock.calls[0][0].date).toBe(LATER);
  });

  it("keeps the selected subject when the chosen date labels it, ahead of that date's first", async () => {
    vi.mocked(api.projects.open).mockResolvedValue(openedAs(VALLEY));
    vi.mocked(api.dataset.select).mockResolvedValue(SELECT_ANSWER);
    const [first, ...rest] = VALLEY.tree.subjects_by_date[LATER];
    expect(first).not.toBe(SUBJECT);
    expect(rest).toContain(SUBJECT);
    render(<ProjectBreadcrumb />);
    fireEvent.click(screen.getByTitle("Switch date"));
    fireEvent.click(await screen.findByText(LATER));
    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      date: LATER,
      subject: SUBJECT,
      bucket: null,
    });
  });
});
