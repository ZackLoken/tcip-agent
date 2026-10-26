import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api, type ProjectSummary } from "@/api/client";
import type { OpenAnswer, SnapshotProject } from "@/api/types.generated";
import { LISTING_REFRESH_MS, ProjectPicker } from "@/components/ProjectPicker";
import { startOpen } from "@/lib/openProject";
import { useStore } from "@/store";
import { INVARIANCES } from "@/test/projectRecords";
import { openAnswer, stateEnvelope } from "@/test/store";

type Selected = Awaited<ReturnType<typeof api.dataset.select>>;

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

const PROJECTS: ProjectSummary[] = [
  {
    id: "a1b2c3d4e5f6",
    display_name: "Valley farm",
    site: "north orchard",
    invariances: INVARIANCES,
    record_problem: null,
    path: "/ws/valley",
    created: 1_700_000_000,
    modified: 1_700_000_500,
    dates: ["2026-02-11", "2026-03-01"],
    subjects: ["subject_a", "bush"],
    // subject_a labeled (+ the bucket baseline published) on 02-11; bush labeled on 03-01.
    subjects_by_date: { "2026-02-11": ["subject_a"], "2026-03-01": ["bush"] },
    buckets_by_date: { "2026-02-11": ["baseline"], "2026-03-01": [] },
    image_count: 42,
    label_problem: null,
  },
  {
    id: null,
    display_name: null,
    site: null,
    invariances: null,
    record_problem: "/ws/stray is not a project yet: create it with initialize_project",
    path: "/ws/stray",
    created: 1_700_000_000,
    modified: 1_700_000_100,
    dates: [],
    subjects: [],
    subjects_by_date: {},
    buckets_by_date: {},
    image_count: 0,
    label_problem: null,
  },
];

function listing(overrides: Partial<Awaited<ReturnType<typeof api.projects.list>>> = {}) {
  return {
    workspace: "/ws",
    open_id: null,
    projects: PROJECTS,
    ...overrides,
  };
}

/** The open route's answer for the listed project `p`, whose record reads: the produced answer
 *  with that project's id, directory and display name. */
function openedAs(p: ProjectSummary): OpenAnswer {
  return openAnswer({ id: p.id!, path: p.path, display_name: p.display_name! });
}

function selection(root: string) {
  return {
    status: "ok",
    selection: {
      dataset_root: root,
      subject: "subject_a",
      date: "2026-03-01",
      image_list: [],
      current_image_index: 0,
      images_dir: null,
      bucket: null,
    },
  };
}

// The backend already holds the project open, as it does when the person switches projects.
function backendHoldsValleyOpen() {
  vi.mocked(api.projects.list).mockResolvedValue(listing({ open_id: "a1b2c3d4e5f6" }));
  vi.mocked(api.projects.open).mockResolvedValue(openedAs(PROJECTS[0]));
  vi.mocked(api.dataset.select).mockResolvedValue(selection("/ws/valley"));
}

const initialStoreState = useStore.getState();

afterEach(cleanup);
beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.getState().setUser("grower");
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

  it("opens a project by its id, then selects a dataset inside its own directory", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    vi.mocked(api.projects.open).mockResolvedValue(openedAs(PROJECTS[0]));
    vi.mocked(api.dataset.select).mockResolvedValue(selection("/ws/valley"));
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText("Valley farm"));
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(api.projects.open).toHaveBeenCalledWith({ id: "a1b2c3d4e5f6", user: "grower" });
    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      dataset_root: "/ws/valley",
      date: "2026-03-01",
    });
    expect(useStore.getState().openProject).toEqual(openedAs(PROJECTS[0]).project);
  });

  it("lists a project created while the picker is on screen without a reload", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      vi.mocked(api.projects.list)
        .mockResolvedValueOnce(listing({ projects: [] }))
        .mockResolvedValue(listing());
      render(<ProjectPicker />);
      expect(await screen.findByText("No projects.")).toBeInTheDocument();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(LISTING_REFRESH_MS);
      });

      expect(await screen.findByText("Valley farm")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("opens a project with no dated images and selects no dataset", async () => {
    const empty: ProjectSummary = {
      ...PROJECTS[0],
      id: "0f0e0d0c0b0a",
      display_name: "Empty block",
      path: "/ws/empty",
      dates: [],
      subjects: [],
      subjects_by_date: {},
      buckets_by_date: {},
      image_count: 0,
    };
    vi.mocked(api.projects.list).mockResolvedValue(listing({ projects: [empty] }));
    vi.mocked(api.projects.open).mockResolvedValue(openedAs(empty));
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText("Empty block"));
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    await waitFor(() => expect(useStore.getState().openProject).toEqual(openedAs(empty).project));
    expect(api.projects.open).toHaveBeenCalledWith({ id: "0f0e0d0c0b0a", user: "grower" });
    expect(api.dataset.select).not.toHaveBeenCalled();
  });

  it("refuses to open a project while the Annotator field holds no name, and says why at the field", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    useStore.getState().setUser("");
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText("Valley farm"));
    const field = screen.getByLabelText("Your name");
    fireEvent.change(field, { target: { value: "   " } });
    expect(screen.getByRole("button", { name: "Open" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Rename…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Remove…" })).toBeDisabled();
    expect(field).toHaveAccessibleDescription(/Enter your name/);
    fireEvent.keyDown(field, { key: "Enter" });
    expect(useStore.getState().user).toBe("");
  });

  it("keeps a typed name a draft until Enter commits it, and the open button then reads it", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    useStore.getState().setUser("");
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText("Valley farm"));
    const field = screen.getByLabelText("Your name");
    fireEvent.change(field, { target: { value: "j" } });
    expect(useStore.getState().user).toBe("");
    expect(screen.getByRole("button", { name: "Rename…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Open" })).toBeEnabled();

    fireEvent.keyDown(field, { key: "Enter" });
    expect(useStore.getState().user).toBe("j");
    expect(screen.getByRole("button", { name: "Rename…" })).toBeEnabled();
    expect(screen.queryByText(/Enter your name/)).not.toBeInTheDocument();
  });

  it("commits the typed name as typed and opens the project from the open button", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    vi.mocked(api.projects.open).mockResolvedValue(openedAs(PROJECTS[0]));
    vi.mocked(api.dataset.select).mockResolvedValue(selection("/ws/valley"));
    useStore.getState().setUser("");
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText("Valley farm"));
    fireEvent.change(screen.getByLabelText("Your name"), { target: { value: " jordan " } });
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    await waitFor(() =>
      expect(api.projects.open).toHaveBeenCalledWith({ id: "a1b2c3d4e5f6", user: " jordan " }),
    );
    expect(useStore.getState().user).toBe(" jordan ");
  });

  describe("an open in flight", () => {
    const HILL: ProjectSummary = {
      ...PROJECTS[0],
      id: "b1b2c3d4e5f6",
      display_name: "Hill farm",
      path: "/ws/hill",
    };
    const VALLEY_OPENED = openedAs(PROJECTS[0]);
    const HILL_OPENED = openedAs(HILL);

    function pendingOpen() {
      const open = Promise.withResolvers<OpenAnswer>();
      vi.mocked(api.projects.list).mockResolvedValue(listing({ projects: [PROJECTS[0], HILL] }));
      vi.mocked(api.projects.open).mockReturnValue(open.promise);
      return open.reject;
    }

    /** Have the dataset select pending until the test resolves the promise returned. */
    function pendingSelect() {
      const select = Promise.withResolvers<Selected>();
      vi.mocked(api.dataset.select).mockReturnValue(select.promise);
      return select;
    }

    /** Merge the backend's snapshot naming `project` at `version`. */
    const merge = (project: SnapshotProject, version = 1) =>
      act(() => useStore.getState().mergeSnapshot(stateEnvelope({ version, project })));

    async function startOpeningValley() {
      render(<ProjectPicker />);
      fireEvent.click(await screen.findByText("Valley farm"));
      fireEvent.click(screen.getByRole("button", { name: "Open" }));
      await waitFor(() => expect(useStore.getState().opening?.projectId).toBe("a1b2c3d4e5f6"));
    }

    it("records the failure for the project it was started for while that project is chosen", async () => {
      const reject = pendingOpen();
      await startOpeningValley();

      await act(() => Promise.resolve(reject(new Error("valley refused"))));

      expect(useStore.getState().openError?.projectId).toBe("a1b2c3d4e5f6");
      expect(useStore.getState().openError?.message).toContain("valley refused");
      expect(await screen.findByText(/valley refused/)).toBeInTheDocument();
      expect(useStore.getState().opening).toBeNull();
    });

    it("drops the failure once another project has been chosen", async () => {
      const reject = pendingOpen();
      await startOpeningValley();
      fireEvent.click(screen.getByText("Hill farm"));

      await act(() => Promise.resolve(reject(new Error("valley refused"))));

      expect(useStore.getState().openError).toBeNull();
      expect(screen.queryByText(/valley refused/)).not.toBeInTheDocument();
      expect(useStore.getState().opening).toBeNull();
    });

    it("drops the failure once a snapshot has adopted another project", async () => {
      const reject = pendingOpen();
      await startOpeningValley();
      merge(HILL_OPENED.project);

      await act(() => Promise.resolve(reject(new Error("valley refused"))));

      expect(useStore.getState().openError).toBeNull();
    });

    it("applies a success while the project is still the chosen one", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      vi.mocked(api.dataset.select).mockResolvedValue(selection("/ws/valley"));
      await startOpeningValley();

      await waitFor(() => expect(useStore.getState().openProject).toEqual(VALLEY_OPENED.project));
    });

    it("drops a success once another project has been chosen", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      const select = pendingSelect();
      await startOpeningValley();
      await waitFor(() => expect(api.dataset.select).toHaveBeenCalled());
      fireEvent.click(screen.getByText("Hill farm"));

      await act(() => Promise.resolve(select.resolve(selection("/ws/valley"))));

      expect(useStore.getState().openProject).toBeNull();
      expect(useStore.getState().opening).toBeNull();
    });

    it("drops a success arriving after a newer snapshot adopted another project", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      const select = pendingSelect();
      await startOpeningValley();
      await waitFor(() => expect(api.dataset.select).toHaveBeenCalled());
      merge(HILL_OPENED.project);

      await act(() => Promise.resolve(select.resolve(selection("/ws/valley"))));

      expect(useStore.getState().openProject).toEqual(HILL_OPENED.project);
    });

    it("drops a success once the project open before it has been adopted again", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      const select = pendingSelect();
      const hill = HILL_OPENED.project;
      useStore.setState({ openProject: hill });
      await startOpeningValley();
      await waitFor(() => expect(api.dataset.select).toHaveBeenCalled());
      merge(VALLEY_OPENED.project, 1);
      merge(hill, 2);

      await act(() => Promise.resolve(select.resolve(selection("/ws/valley"))));

      expect(useStore.getState().openProject).toEqual(hill);
    });

    it("selects no dataset for an open whose project was superseded before it answered", async () => {
      vi.mocked(api.projects.list).mockResolvedValue(listing({ projects: [PROJECTS[0], HILL] }));
      const open = Promise.withResolvers<OpenAnswer>();
      vi.mocked(api.projects.open).mockReturnValue(open.promise);
      await startOpeningValley();
      merge(HILL_OPENED.project);

      await act(() => Promise.resolve(open.resolve(VALLEY_OPENED)));

      expect(api.dataset.select).not.toHaveBeenCalled();
    });

    it("keeps a held open when the store rejects a stale snapshot of another project", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      const select = pendingSelect();
      merge(VALLEY_OPENED.project, 10);
      await startOpeningValley();
      await waitFor(() => expect(api.dataset.select).toHaveBeenCalled());
      merge(HILL_OPENED.project, 9);

      expect(useStore.getState().opening?.projectId).toBe("a1b2c3d4e5f6");
      await act(() => Promise.resolve(select.resolve(selection("/ws/valley"))));

      expect(useStore.getState().opening).toBeNull();
      expect(useStore.getState().openProject).toEqual(VALLEY_OPENED.project);
      expect(useStore.getState().gui.dataset.dataset_root).toBe("/ws/valley");
    });

    it("preselects the listed open project under the replayed mount effect of StrictMode", async () => {
      backendHoldsValleyOpen();
      render(
        <StrictMode>
          <ProjectPicker />
        </StrictMode>,
      );

      await screen.findByText("Valley farm");
      await waitFor(() => expect(useStore.getState().openChoice.projectId).toBe("a1b2c3d4e5f6"));
      expect(api.projects.open).not.toHaveBeenCalled();
      expect(useStore.getState().opening).toBeNull();
    });

    it("leaves a newer request for the same project held when the obsolete one ends", async () => {
      vi.mocked(api.projects.list).mockResolvedValue(listing({ projects: [PROJECTS[0], HILL] }));
      const obsolete = Promise.withResolvers<OpenAnswer>();
      vi.mocked(api.projects.open)
        .mockReturnValueOnce(obsolete.promise)
        .mockReturnValueOnce(new Promise(() => {}));
      await startOpeningValley();
      const first = useStore.getState().opening?.requestId;
      fireEvent.click(screen.getByText("Hill farm"));
      fireEvent.click(screen.getByText("Valley farm"));
      fireEvent.click(screen.getByRole("button", { name: "Open" }));
      await waitFor(() => expect(api.projects.open).toHaveBeenCalledTimes(2));
      const second = useStore.getState().opening?.requestId;
      expect(second).not.toBe(first);

      await act(() => Promise.resolve(obsolete.reject(new Error("first refused"))));

      expect(useStore.getState().opening).toEqual({
        requestId: second,
        projectId: "a1b2c3d4e5f6",
      });
      expect(useStore.getState().openError).toBeNull();
    });

    it("leaves one request when a second open is started beside one in flight", async () => {
      pendingOpen();
      render(<ProjectPicker />);
      fireEvent.click(await screen.findByText("Valley farm"));
      const [valley] = PROJECTS.filter((p): p is typeof p & { id: string } => p.id !== null);

      void startOpen(valley, "2026-03-01", "subject_a", "");
      void startOpen(valley, "2026-03-01", "subject_a", "");

      await waitFor(() => expect(api.projects.open).toHaveBeenCalled());
      expect(api.projects.open).toHaveBeenCalledTimes(1);
    });

    it("shows a failure only on the card of the project it belongs to", async () => {
      vi.mocked(api.projects.list).mockResolvedValue(listing({ projects: [PROJECTS[0], HILL] }));
      render(<ProjectPicker />);
      await screen.findByText("Hill farm");
      act(() => {
        const state = useStore.getState();
        state.patchOpenChoice({ projectId: "b1b2c3d4e5f6", date: "2026-03-01" });
        state.patchOpenStatus({
          openError: { projectId: "a1b2c3d4e5f6", message: "valley refused" },
        });
      });

      expect(screen.queryByText(/valley refused/)).not.toBeInTheDocument();

      act(() =>
        useStore
          .getState()
          .patchOpenStatus({ openError: { projectId: "b1b2c3d4e5f6", message: "hill refused" } }),
      );
      expect(screen.getByText(/hill refused/)).toBeInTheDocument();
    });
  });

  it("marks the project the listing names as open", async () => {
    const hill = {
      ...PROJECTS[0],
      id: "b1b2c3d4e5f6",
      display_name: "Hill farm",
      path: "/ws/hill",
    };
    vi.mocked(api.projects.list).mockResolvedValue(
      listing({ open_id: "a1b2c3d4e5f6", projects: [PROJECTS[0], hill] }),
    );
    render(<ProjectPicker />);

    await screen.findByText("Hill farm");
    expect(screen.getAllByText("open")).toHaveLength(1);
    expect(screen.getByText("open").closest(".tcip-panel")).toHaveTextContent("Valley farm");
  });

  it("preselects the backend's open project and opens nothing until the person does", async () => {
    backendHoldsValleyOpen();
    render(<ProjectPicker />);

    await screen.findByText("Valley farm");
    await waitFor(() => expect(useStore.getState().openChoice.projectId).toBe("a1b2c3d4e5f6"));
    expect(api.projects.open).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Open" })).toBeEnabled();
  });

  it("keeps the project the person chose over the preselection", async () => {
    backendHoldsValleyOpen();
    useStore.getState().patchOpenChoice({ projectId: "other0000000" });
    render(<ProjectPicker />);

    await screen.findByText("Valley farm");
    expect(api.projects.open).not.toHaveBeenCalled();
    expect(useStore.getState().openChoice.projectId).toBe("other0000000");
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
