import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import { GUI_STATE_DEFAULTS } from "@/api/types.generated";
import { ProjectPicker } from "@/components/ProjectPicker";
import { startOpen } from "@/lib/openProject";
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
      image_list: [],
      current_image_index: 0,
      images_dir: null,
      bucket: null,
    },
  };
}

// The backend already holds the project open, as it does when the app loads on a last-opened one.
function backendHoldsValleyOpen() {
  vi.mocked(api.projects.list).mockResolvedValue(listing({ open_id: "a1b2c3d4e5f6" }));
  vi.mocked(api.projects.open).mockResolvedValue({
    id: "a1b2c3d4e5f6",
    display_name: "Valley farm",
    path: "/ws/valley",
  });
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
    expect(api.projects.open).toHaveBeenCalledWith({ id: "a1b2c3d4e5f6", user: "grower" });
    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      dataset_root: "/ws/valley",
      date: "2026-03-01",
    });
    expect(useStore.getState().openProject).toEqual({ id: "a1b2c3d4e5f6", path: "/ws/valley" });
  });

  it("refuses to open a project while the Annotator field holds no name, and says why at the field", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    useStore.getState().setUser("");
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText("Valley farm"));
    const field = screen.getByLabelText("Annotator");
    fireEvent.change(field, { target: { value: "   " } });
    expect(screen.getByRole("button", { name: "Open project" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Rename…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Remove…" })).toBeDisabled();
    expect(field).toHaveAccessibleDescription(/Enter your name and press Enter/);
    fireEvent.keyDown(field, { key: "Enter" });
    expect(useStore.getState().user).toBe("");
  });

  it("keeps a typed name a draft until Enter commits it, and the open button then reads it", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    useStore.getState().setUser("");
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText("Valley farm"));
    const field = screen.getByLabelText("Annotator");
    fireEvent.change(field, { target: { value: "j" } });
    expect(useStore.getState().user).toBe("");
    expect(screen.getByRole("button", { name: "Rename…" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Open project" })).toBeEnabled();

    fireEvent.keyDown(field, { key: "Enter" });
    expect(useStore.getState().user).toBe("j");
    expect(screen.getByRole("button", { name: "Rename…" })).toBeEnabled();
    expect(screen.queryByText(/press Enter/)).not.toBeInTheDocument();
  });

  it("commits the typed name as typed and opens the project from the open button", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(listing());
    vi.mocked(api.projects.open).mockResolvedValue({
      id: "a1b2c3d4e5f6",
      display_name: "Valley farm",
      path: "/ws/valley",
    });
    vi.mocked(api.dataset.select).mockResolvedValue(selection("/ws/valley"));
    useStore.getState().setUser("");
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText("Valley farm"));
    fireEvent.change(screen.getByLabelText("Annotator"), { target: { value: " jordan " } });
    fireEvent.click(screen.getByRole("button", { name: "Open project" }));

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
    const VALLEY_OPENED = { id: "a1b2c3d4e5f6", display_name: "Valley farm", path: "/ws/valley" };

    function pendingOpen() {
      let reject!: (e: Error) => void;
      const promise = new Promise<Awaited<ReturnType<typeof api.projects.open>>>((_, rej) => {
        reject = rej;
      });
      vi.mocked(api.projects.list).mockResolvedValue(listing({ projects: [PROJECTS[0], HILL] }));
      vi.mocked(api.projects.open).mockReturnValue(promise);
      return reject;
    }

    async function startOpeningValley() {
      render(<ProjectPicker />);
      fireEvent.click(await screen.findByText("Valley farm"));
      fireEvent.click(screen.getByRole("button", { name: "Open project" }));
      await waitFor(() => expect(useStore.getState().opening?.projectId).toBe("a1b2c3d4e5f6"));
    }

    it("records the failure for the project it was started for while that project is chosen", async () => {
      const reject = pendingOpen();
      await startOpeningValley();

      await act(async () => reject(new Error("valley refused")));

      expect(useStore.getState().openError).toEqual({
        projectId: "a1b2c3d4e5f6",
        message: expect.stringContaining("valley refused"),
      });
      expect(await screen.findByText(/valley refused/)).toBeInTheDocument();
      expect(useStore.getState().opening).toBeNull();
    });

    it("drops the failure once another project has been chosen", async () => {
      const reject = pendingOpen();
      await startOpeningValley();
      fireEvent.click(screen.getByText("Hill farm"));

      await act(async () => reject(new Error("valley refused")));

      expect(useStore.getState().openError).toBeNull();
      expect(screen.queryByText(/valley refused/)).not.toBeInTheDocument();
      expect(useStore.getState().opening).toBeNull();
    });

    it("drops the failure once a snapshot has adopted another project", async () => {
      const reject = pendingOpen();
      await startOpeningValley();
      act(() =>
        useStore
          .getState()
          .mergeSnapshot(GUI_STATE_DEFAULTS, 1, { id: "b1b2c3d4e5f6", path: "/ws/hill" }, "e1"),
      );

      await act(async () => reject(new Error("valley refused")));

      expect(useStore.getState().openError).toBeNull();
    });

    it("applies a success while the project is still the chosen one", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      vi.mocked(api.dataset.select).mockResolvedValue(selection("/ws/valley"));
      await startOpeningValley();

      await waitFor(() =>
        expect(useStore.getState().openProject).toEqual({ id: "a1b2c3d4e5f6", path: "/ws/valley" }),
      );
    });

    it("drops a success once another project has been chosen", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      let release!: () => void;
      vi.mocked(api.dataset.select).mockReturnValue(
        new Promise((resolve) => {
          release = () => resolve(selection("/ws/valley"));
        }),
      );
      await startOpeningValley();
      await waitFor(() => expect(api.dataset.select).toHaveBeenCalled());
      fireEvent.click(screen.getByText("Hill farm"));

      await act(async () => release());

      expect(useStore.getState().openProject).toBeNull();
      expect(useStore.getState().opening).toBeNull();
    });

    it("drops a success arriving after a newer snapshot adopted another project", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      let release!: () => void;
      vi.mocked(api.dataset.select).mockReturnValue(
        new Promise((resolve) => {
          release = () => resolve(selection("/ws/valley"));
        }),
      );
      await startOpeningValley();
      await waitFor(() => expect(api.dataset.select).toHaveBeenCalled());
      act(() =>
        useStore
          .getState()
          .mergeSnapshot(GUI_STATE_DEFAULTS, 1, { id: "b1b2c3d4e5f6", path: "/ws/hill" }, "e1"),
      );

      await act(async () => release());

      expect(useStore.getState().openProject).toEqual({ id: "b1b2c3d4e5f6", path: "/ws/hill" });
    });

    it("drops a success once the project open before it has been adopted again", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      let release!: () => void;
      vi.mocked(api.dataset.select).mockReturnValue(
        new Promise((resolve) => {
          release = () => resolve(selection("/ws/valley"));
        }),
      );
      const hill = { id: "b1b2c3d4e5f6", path: "/ws/hill" };
      useStore.setState({ openProject: hill });
      await startOpeningValley();
      await waitFor(() => expect(api.dataset.select).toHaveBeenCalled());
      const merge = (project: { id: string; path: string }, version: number) =>
        act(() => useStore.getState().mergeSnapshot(GUI_STATE_DEFAULTS, version, project, "e1"));
      merge({ id: "a1b2c3d4e5f6", path: "/ws/valley" }, 1);
      merge(hill, 2);

      await act(async () => release());

      expect(useStore.getState().openProject).toEqual(hill);
    });

    it("selects no dataset for an open whose project was superseded before it answered", async () => {
      vi.mocked(api.projects.list).mockResolvedValue(listing({ projects: [PROJECTS[0], HILL] }));
      let answer!: () => void;
      vi.mocked(api.projects.open).mockReturnValue(
        new Promise((resolve) => {
          answer = () => resolve(VALLEY_OPENED);
        }),
      );
      await startOpeningValley();
      act(() =>
        useStore
          .getState()
          .mergeSnapshot(GUI_STATE_DEFAULTS, 1, { id: "b1b2c3d4e5f6", path: "/ws/hill" }, "e1"),
      );

      await act(async () => answer());

      expect(api.dataset.select).not.toHaveBeenCalled();
    });

    it("keeps a held open when the store rejects a stale snapshot of another project", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      let release!: () => void;
      vi.mocked(api.dataset.select).mockReturnValue(
        new Promise((resolve) => {
          release = () => resolve(selection("/ws/valley"));
        }),
      );
      act(() =>
        useStore
          .getState()
          .mergeSnapshot(GUI_STATE_DEFAULTS, 10, { id: "a1b2c3d4e5f6", path: "/ws/valley" }, "e1"),
      );
      await startOpeningValley();
      await waitFor(() => expect(api.dataset.select).toHaveBeenCalled());
      act(() =>
        useStore
          .getState()
          .mergeSnapshot(GUI_STATE_DEFAULTS, 9, { id: "b1b2c3d4e5f6", path: "/ws/hill" }, "e1"),
      );

      expect(useStore.getState().opening?.projectId).toBe("a1b2c3d4e5f6");
      await act(async () => release());

      expect(useStore.getState().opening).toBeNull();
      expect(useStore.getState().openProject).toEqual({ id: "a1b2c3d4e5f6", path: "/ws/valley" });
      expect(useStore.getState().gui.dataset.dataset_root).toBe("/ws/valley");
    });

    it("drops a first-load opening whose listing a snapshot of another project overtook", async () => {
      let list!: () => void;
      vi.mocked(api.projects.list).mockReturnValue(
        new Promise((resolve) => {
          list = () => resolve(listing({ open_id: "a1b2c3d4e5f6" }));
        }),
      );
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      render(<ProjectPicker />);
      expect(useStore.getState().opening).toEqual({
        requestId: expect.any(Number),
        projectId: null,
        accepted: null,
      });
      act(() =>
        useStore
          .getState()
          .mergeSnapshot(GUI_STATE_DEFAULTS, 1, { id: "b1b2c3d4e5f6", path: "/ws/hill" }, "e1"),
      );

      await act(async () => list());

      expect(api.projects.open).not.toHaveBeenCalled();
      expect(useStore.getState().openChoice.projectId).toBeNull();
      expect(useStore.getState().opening).toBeNull();
    });

    it("drops a first-load opening when an accepted snapshot closed the project the listing names", async () => {
      let list!: () => void;
      vi.mocked(api.projects.list).mockReturnValue(
        new Promise((resolve) => {
          list = () => resolve(listing({ open_id: "a1b2c3d4e5f6" }));
        }),
      );
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      vi.mocked(api.dataset.select).mockResolvedValue(selection("/ws/valley"));
      render(<ProjectPicker />);
      act(() =>
        useStore
          .getState()
          .mergeSnapshot(GUI_STATE_DEFAULTS, 1, { id: "a1b2c3d4e5f6", path: "/ws/valley" }, "e1"),
      );
      act(() => useStore.getState().mergeSnapshot(GUI_STATE_DEFAULTS, 2, null, "e1"));
      expect(useStore.getState().openProject).toBeNull();

      await act(async () => list());

      expect(api.projects.open).not.toHaveBeenCalled();
      expect(useStore.getState().openChoice.projectId).toBeNull();
      expect(useStore.getState().opening).toBeNull();
    });

    it("drops a first-load opening when an accepted snapshot says no project while none was open", async () => {
      let list!: () => void;
      vi.mocked(api.projects.list).mockReturnValue(
        new Promise((resolve) => {
          list = () => resolve(listing({ open_id: "a1b2c3d4e5f6" }));
        }),
      );
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      vi.mocked(api.dataset.select).mockResolvedValue(selection("/ws/valley"));
      render(<ProjectPicker />);
      expect(useStore.getState().openProject).toBeNull();
      act(() => useStore.getState().mergeSnapshot(GUI_STATE_DEFAULTS, 1, null, "e1"));
      expect(useStore.getState().wsVersion).toBe(1);

      await act(async () => list());

      expect(api.projects.open).not.toHaveBeenCalled();
      expect(useStore.getState().openChoice.projectId).toBeNull();
      expect(useStore.getState().opening).toBeNull();
    });

    it("continues a first-load opening when a snapshot adopted the project its listing names", async () => {
      let list!: () => void;
      vi.mocked(api.projects.list).mockReturnValue(
        new Promise((resolve) => {
          list = () => resolve(listing({ open_id: "a1b2c3d4e5f6" }));
        }),
      );
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      vi.mocked(api.dataset.select).mockResolvedValue(selection("/ws/valley"));
      render(<ProjectPicker />);
      act(() =>
        useStore
          .getState()
          .mergeSnapshot(GUI_STATE_DEFAULTS, 1, { id: "a1b2c3d4e5f6", path: "/ws/valley" }, "e1"),
      );
      expect(useStore.getState().opening?.projectId).toBeNull();

      await act(async () => list());

      await waitFor(() => expect(useStore.getState().gui.dataset.dataset_root).toBe("/ws/valley"));
      expect(useStore.getState().openChoice.projectId).toBe("a1b2c3d4e5f6");
      expect(api.projects.open).toHaveBeenCalledTimes(1);
      expect(useStore.getState().opening).toBeNull();
    });

    it("releases a first-load hold when the picker goes before its listing answers", async () => {
      let list!: () => void;
      vi.mocked(api.projects.list).mockReturnValue(
        new Promise((resolve) => {
          list = () => resolve(listing({ open_id: "a1b2c3d4e5f6" }));
        }),
      );
      render(<ProjectPicker />);
      expect(useStore.getState().opening).not.toBeNull();

      cleanup();
      expect(useStore.getState().opening).toBeNull();
      await act(async () => list());

      expect(api.projects.open).not.toHaveBeenCalled();
    });

    it("opens the listed project once under the replayed mount effect of StrictMode", async () => {
      let list!: () => void;
      vi.mocked(api.projects.list).mockReturnValue(
        new Promise((resolve) => {
          list = () => resolve(listing({ open_id: "a1b2c3d4e5f6" }));
        }),
      );
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      vi.mocked(api.dataset.select).mockResolvedValue(selection("/ws/valley"));
      render(
        <StrictMode>
          <ProjectPicker />
        </StrictMode>,
      );

      await act(async () => list());

      await waitFor(() => expect(useStore.getState().gui.dataset.dataset_root).toBe("/ws/valley"));
      expect(api.projects.open).toHaveBeenCalledTimes(1);
      expect(useStore.getState().openChoice.projectId).toBe("a1b2c3d4e5f6");
      expect(useStore.getState().opening).toBeNull();
    });

    it("keeps a first-load open running when the picker goes after handing it over", async () => {
      let answer!: () => void;
      backendHoldsValleyOpen();
      vi.mocked(api.projects.open).mockReturnValue(
        new Promise((resolve) => {
          answer = () => resolve(VALLEY_OPENED);
        }),
      );
      render(<ProjectPicker />);
      await waitFor(() => expect(useStore.getState().opening?.projectId).toBe("a1b2c3d4e5f6"));

      cleanup();
      expect(useStore.getState().opening?.projectId).toBe("a1b2c3d4e5f6");
      await act(async () => answer());

      await waitFor(() =>
        expect(useStore.getState().openProject).toEqual({ id: "a1b2c3d4e5f6", path: "/ws/valley" }),
      );
      expect(useStore.getState().gui.dataset.dataset_root).toBe("/ws/valley");
      expect(useStore.getState().opening).toBeNull();
    });

    it("leaves a newer request for the same project held when the obsolete one ends", async () => {
      vi.mocked(api.projects.list).mockResolvedValue(listing({ projects: [PROJECTS[0], HILL] }));
      let rejectFirst!: (e: Error) => void;
      vi.mocked(api.projects.open)
        .mockReturnValueOnce(
          new Promise((_, reject) => {
            rejectFirst = reject;
          }),
        )
        .mockReturnValueOnce(new Promise(() => {}));
      await startOpeningValley();
      const first = useStore.getState().opening?.requestId;
      fireEvent.click(screen.getByText("Hill farm"));
      fireEvent.click(screen.getByText("Valley farm"));
      fireEvent.click(screen.getByRole("button", { name: "Open project" }));
      await waitFor(() => expect(api.projects.open).toHaveBeenCalledTimes(2));
      const second = useStore.getState().opening?.requestId;
      expect(second).not.toBe(first);

      await act(async () => rejectFirst(new Error("first refused")));

      expect(useStore.getState().opening).toEqual({
        requestId: second,
        projectId: "a1b2c3d4e5f6",
        accepted: null,
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

    it("holds the first-load opening in flight so a second open cannot start beside it", async () => {
      backendHoldsValleyOpen();
      vi.mocked(api.projects.open).mockReturnValue(new Promise(() => {}));
      render(<ProjectPicker />);

      await waitFor(() => expect(useStore.getState().opening?.projectId).toBe("a1b2c3d4e5f6"));
      expect(await screen.findByRole("button", { name: "Opening…" })).toBeDisabled();
      expect(api.projects.open).toHaveBeenCalledTimes(1);
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

  it("does not reopen the backend's open project on first load while no name is entered", async () => {
    backendHoldsValleyOpen();
    useStore.getState().setUser("");
    render(<ProjectPicker />);

    await screen.findByText("Valley farm");
    expect(api.projects.open).not.toHaveBeenCalled();
    expect(useStore.getState().initialOpenAttempted).toBe(false);
    expect(useStore.getState().openChoice.projectId).toBeNull();
  });

  it("opens the backend's open project once a name is committed after a blank first mount", async () => {
    backendHoldsValleyOpen();
    useStore.getState().setUser("");
    render(<ProjectPicker />);
    await screen.findByText("Valley farm");
    cleanup();

    useStore.getState().setUser("jordan");
    render(<ProjectPicker />);

    await waitFor(() =>
      expect(api.projects.open).toHaveBeenCalledWith({ id: "a1b2c3d4e5f6", user: "jordan" }),
    );
    expect(useStore.getState().openChoice.projectId).toBe("a1b2c3d4e5f6");
  });

  it("opens the backend's open project on first load when a name is already committed", async () => {
    backendHoldsValleyOpen();
    render(<ProjectPicker />);

    await waitFor(() =>
      expect(api.projects.open).toHaveBeenCalledWith({ id: "a1b2c3d4e5f6", user: "grower" }),
    );
  });

  it("keeps the project the person chose over the first-load opening", async () => {
    backendHoldsValleyOpen();
    useStore.getState().patchOpenChoice({ projectId: "other0000000" });
    render(<ProjectPicker />);

    await screen.findByText("Valley farm");
    expect(api.projects.open).not.toHaveBeenCalled();
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
