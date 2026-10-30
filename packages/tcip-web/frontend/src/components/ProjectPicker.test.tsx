import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "@/api/client";
import type { OpenAnswer, SnapshotProject } from "@/api/types.generated";
import { LISTING_REFRESH_MS, ProjectPicker } from "@/components/ProjectPicker";
import { startOpen } from "@/lib/openProject";
import { useStore } from "@/store";
import { openedAs, PICKER, PROJECT_FIXTURES } from "@/test/projectRecords";
import { stateEnvelope, SUBJECT_SELECT_ANSWER } from "@/test/store";

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

const { valley: VALLEY, ridge: RIDGE, empty: EMPTY, stray: STRAY } = PICKER;
const { picker: LISTING, picker_open: VALLEY_OPEN, empty_listing: NO_PROJECTS } = PROJECT_FIXTURES;

/** The newest of Valley farm's dates holding labels. */
const [LANDED] = Object.entries(VALLEY.tree.subjects_by_date)
  .filter(([, subjects]) => subjects.length > 0)
  .map(([date]) => date)
  .reverse();

const SELECT_ANSWER = SUBJECT_SELECT_ANSWER;

// The backend already holds the project open, as it does when the person switches projects.
function backendHoldsValleyOpen() {
  vi.mocked(api.projects.list).mockResolvedValue(VALLEY_OPEN);
  vi.mocked(api.projects.open).mockResolvedValue(openedAs(VALLEY));
  vi.mocked(api.dataset.select).mockResolvedValue(SELECT_ANSWER);
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
    vi.mocked(api.projects.list).mockResolvedValue(LISTING);
    render(<ProjectPicker />);

    expect(await screen.findByText(VALLEY.record.display_name)).toBeInTheDocument();
    expect(screen.getAllByText(VALLEY.record.site).length).toBeGreaterThan(0);
    expect(screen.getByText(`${VALLEY.image_count} image(s)`)).toBeInTheDocument();
    expect(screen.getByText(`${VALLEY.tree.captures.length} dates`)).toBeInTheDocument();
  });

  it("lists a directory whose record does not read by its path and the reason, with no open", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(LISTING);
    render(<ProjectPicker />);

    expect(await screen.findByText(STRAY.record_problem!)).toBeInTheDocument();
    expect(screen.getByText(STRAY.tree.dataset_root)).toBeInTheDocument();
  });

  it("opens a project by its id, then selects a dataset inside its own directory", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(LISTING);
    vi.mocked(api.projects.open).mockResolvedValue(openedAs(VALLEY));
    vi.mocked(api.dataset.select).mockResolvedValue(SELECT_ANSWER);
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText(VALLEY.record.display_name));
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    await waitFor(() => expect(api.dataset.select).toHaveBeenCalledTimes(1));
    expect(api.projects.open).toHaveBeenCalledWith({ id: VALLEY.record.id, user: "grower" });
    expect(vi.mocked(api.dataset.select).mock.calls[0][0]).toMatchObject({
      dataset_root: VALLEY.tree.dataset_root,
      date: VALLEY.tree.captures.at(-1)!.name,
    });
    expect(useStore.getState().openProject).toEqual(openedAs(VALLEY).project);
  });

  it("lists a project created while the picker is on screen without a reload", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    try {
      vi.mocked(api.projects.list).mockResolvedValueOnce(NO_PROJECTS).mockResolvedValue(LISTING);
      render(<ProjectPicker />);
      expect(await screen.findByText("No projects.")).toBeInTheDocument();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(LISTING_REFRESH_MS);
      });

      expect(await screen.findByText(VALLEY.record.display_name)).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("opens a project with no dated images and selects no dataset", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(LISTING);
    vi.mocked(api.projects.open).mockResolvedValue(openedAs(EMPTY));
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText(EMPTY.record.display_name));
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    await waitFor(() => expect(useStore.getState().openProject).toEqual(openedAs(EMPTY).project));
    expect(api.projects.open).toHaveBeenCalledWith({ id: EMPTY.record.id, user: "grower" });
    expect(api.dataset.select).not.toHaveBeenCalled();
  });

  it("refuses to open a project while the Annotator field holds no name, and says why at the field", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(LISTING);
    useStore.getState().setUser("");
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText(VALLEY.record.display_name));
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
    vi.mocked(api.projects.list).mockResolvedValue(LISTING);
    useStore.getState().setUser("");
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText(VALLEY.record.display_name));
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
    vi.mocked(api.projects.list).mockResolvedValue(LISTING);
    vi.mocked(api.projects.open).mockResolvedValue(openedAs(VALLEY));
    vi.mocked(api.dataset.select).mockResolvedValue(SELECT_ANSWER);
    useStore.getState().setUser("");
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText(VALLEY.record.display_name));
    fireEvent.change(screen.getByLabelText("Your name"), { target: { value: " jordan " } });
    fireEvent.click(screen.getByRole("button", { name: "Open" }));

    await waitFor(() =>
      expect(api.projects.open).toHaveBeenCalledWith({ id: VALLEY.record.id, user: " jordan " }),
    );
    expect(useStore.getState().user).toBe(" jordan ");
  });

  describe("an open in flight", () => {
    const VALLEY_OPENED = openedAs(VALLEY);
    const RIDGE_OPENED = openedAs(RIDGE);

    function pendingOpen() {
      const open = Promise.withResolvers<OpenAnswer>();
      vi.mocked(api.projects.list).mockResolvedValue(LISTING);
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
      fireEvent.click(await screen.findByText(VALLEY.record.display_name));
      fireEvent.click(screen.getByRole("button", { name: "Open" }));
      await waitFor(() => expect(useStore.getState().opening?.projectId).toBe(VALLEY.record.id));
    }

    it("records the failure for the project it was started for while that project is chosen", async () => {
      const reject = pendingOpen();
      await startOpeningValley();

      await act(() => Promise.resolve(reject(new Error("valley refused"))));

      expect(useStore.getState().openError?.projectId).toBe(VALLEY.record.id);
      expect(useStore.getState().openError?.message).toContain("valley refused");
      expect(await screen.findByText(/valley refused/)).toBeInTheDocument();
      expect(useStore.getState().opening).toBeNull();
    });

    it("drops the failure once another project has been chosen", async () => {
      const reject = pendingOpen();
      await startOpeningValley();
      fireEvent.click(screen.getByText(RIDGE.record.display_name));

      await act(() => Promise.resolve(reject(new Error("valley refused"))));

      expect(useStore.getState().openError).toBeNull();
      expect(screen.queryByText(/valley refused/)).not.toBeInTheDocument();
      expect(useStore.getState().opening).toBeNull();
    });

    it("drops the failure once a snapshot has adopted another project", async () => {
      const reject = pendingOpen();
      await startOpeningValley();
      merge(RIDGE_OPENED.project);

      await act(() => Promise.resolve(reject(new Error("valley refused"))));

      expect(useStore.getState().openError).toBeNull();
    });

    it("applies a success while the project is still the chosen one", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      vi.mocked(api.dataset.select).mockResolvedValue(SELECT_ANSWER);
      await startOpeningValley();

      await waitFor(() => expect(useStore.getState().openProject).toEqual(VALLEY_OPENED.project));
    });

    it("drops a success once another project has been chosen", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      const select = pendingSelect();
      await startOpeningValley();
      await waitFor(() => expect(api.dataset.select).toHaveBeenCalled());
      fireEvent.click(screen.getByText(RIDGE.record.display_name));

      await act(() => Promise.resolve(select.resolve(SELECT_ANSWER)));

      expect(useStore.getState().openProject).toBeNull();
      expect(useStore.getState().opening).toBeNull();
    });

    it("drops a success arriving after a newer snapshot adopted another project", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      const select = pendingSelect();
      await startOpeningValley();
      await waitFor(() => expect(api.dataset.select).toHaveBeenCalled());
      merge(RIDGE_OPENED.project);

      await act(() => Promise.resolve(select.resolve(SELECT_ANSWER)));

      expect(useStore.getState().openProject).toEqual(RIDGE_OPENED.project);
    });

    it("drops a success once the project open before it has been adopted again", async () => {
      pendingOpen();
      vi.mocked(api.projects.open).mockResolvedValue(VALLEY_OPENED);
      const select = pendingSelect();
      const ridge = RIDGE_OPENED.project;
      useStore.setState({ openProject: ridge });
      await startOpeningValley();
      await waitFor(() => expect(api.dataset.select).toHaveBeenCalled());
      merge(VALLEY_OPENED.project, 1);
      merge(ridge, 2);

      await act(() => Promise.resolve(select.resolve(SELECT_ANSWER)));

      expect(useStore.getState().openProject).toEqual(ridge);
    });

    it("selects no dataset for an open whose project was superseded before it answered", async () => {
      vi.mocked(api.projects.list).mockResolvedValue(LISTING);
      const open = Promise.withResolvers<OpenAnswer>();
      vi.mocked(api.projects.open).mockReturnValue(open.promise);
      await startOpeningValley();
      merge(RIDGE_OPENED.project);

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
      merge(RIDGE_OPENED.project, 9);

      expect(useStore.getState().opening?.projectId).toBe(VALLEY.record.id);
      await act(() => Promise.resolve(select.resolve(SELECT_ANSWER)));

      expect(useStore.getState().opening).toBeNull();
      expect(useStore.getState().openProject).toEqual(VALLEY_OPENED.project);
      expect(useStore.getState().gui.dataset).toEqual(SELECT_ANSWER.selection);
    });

    it("preselects the listed open project under the replayed mount effect of StrictMode", async () => {
      backendHoldsValleyOpen();
      render(
        <StrictMode>
          <ProjectPicker />
        </StrictMode>,
      );

      await screen.findByText(VALLEY.record.display_name);
      await waitFor(() => expect(useStore.getState().openChoice.projectId).toBe(VALLEY.record.id));
      expect(api.projects.open).not.toHaveBeenCalled();
      expect(useStore.getState().opening).toBeNull();
    });

    it("leaves a newer request for the same project held when the obsolete one ends", async () => {
      vi.mocked(api.projects.list).mockResolvedValue(LISTING);
      const obsolete = Promise.withResolvers<OpenAnswer>();
      vi.mocked(api.projects.open)
        .mockReturnValueOnce(obsolete.promise)
        .mockReturnValueOnce(new Promise(() => {}));
      await startOpeningValley();
      const first = useStore.getState().opening?.requestId;
      fireEvent.click(screen.getByText(RIDGE.record.display_name));
      fireEvent.click(screen.getByText(VALLEY.record.display_name));
      fireEvent.click(screen.getByRole("button", { name: "Open" }));
      await waitFor(() => expect(api.projects.open).toHaveBeenCalledTimes(2));
      const second = useStore.getState().opening?.requestId;
      expect(second).not.toBe(first);

      await act(() => Promise.resolve(obsolete.reject(new Error("first refused"))));

      expect(useStore.getState().opening).toEqual({
        requestId: second,
        projectId: VALLEY.record.id,
      });
      expect(useStore.getState().openError).toBeNull();
    });

    it("leaves one request when a second open is started beside one in flight", async () => {
      pendingOpen();
      render(<ProjectPicker />);
      fireEvent.click(await screen.findByText(VALLEY.record.display_name));
      const [subject] = VALLEY.tree.subjects_by_date[LANDED];
      void startOpen(VALLEY, LANDED, subject, null);
      void startOpen(VALLEY, LANDED, subject, null);

      await waitFor(() => expect(api.projects.open).toHaveBeenCalled());
      expect(api.projects.open).toHaveBeenCalledTimes(1);
    });

    it("shows a failure only on the card of the project it belongs to", async () => {
      vi.mocked(api.projects.list).mockResolvedValue(LISTING);
      render(<ProjectPicker />);
      await screen.findByText(RIDGE.record.display_name);
      act(() => {
        const state = useStore.getState();
        state.patchOpenChoice({
          projectId: RIDGE.record.id,
          date: RIDGE.tree.captures[0].name,
        });
        state.patchOpenStatus({
          openError: { projectId: VALLEY.record.id, message: "valley refused" },
        });
      });

      expect(screen.queryByText(/valley refused/)).not.toBeInTheDocument();

      act(() =>
        useStore
          .getState()
          .patchOpenStatus({ openError: { projectId: RIDGE.record.id, message: "ridge refused" } }),
      );
      expect(screen.getByText(/ridge refused/)).toBeInTheDocument();
    });
  });

  it("marks the project the listing names as open", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(VALLEY_OPEN);
    render(<ProjectPicker />);

    await screen.findByText(RIDGE.record.display_name);
    expect(screen.getAllByText("open")).toHaveLength(1);
    expect(screen.getByText("open").closest(".tcip-panel")).toHaveTextContent(
      VALLEY.record.display_name,
    );
  });

  it("preselects the backend's open project and opens nothing until the person does", async () => {
    backendHoldsValleyOpen();
    render(<ProjectPicker />);

    await screen.findByText(VALLEY.record.display_name);
    await waitFor(() => expect(useStore.getState().openChoice.projectId).toBe(VALLEY.record.id));
    expect(api.projects.open).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Open" })).toBeEnabled();
  });

  it("keeps the project the person chose over the preselection", async () => {
    backendHoldsValleyOpen();
    useStore.getState().patchOpenChoice({ projectId: "other0000000" });
    render(<ProjectPicker />);

    await screen.findByText(VALLEY.record.display_name);
    expect(api.projects.open).not.toHaveBeenCalled();
    expect(useStore.getState().openChoice.projectId).toBe("other0000000");
  });

  it("removes a project by its id once its display name is typed", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(LISTING);
    const { removed } = PROJECT_FIXTURES;
    vi.mocked(api.projects.remove).mockResolvedValue(removed);
    useStore.setState({ user: "grower" });
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText(VALLEY.record.display_name));
    fireEvent.click(screen.getByText("Remove…"));
    const confirm = screen.getByRole("button", { name: "Remove" });
    expect(confirm).toBeDisabled();
    fireEvent.change(screen.getByLabelText("Type the project name to confirm"), {
      target: { value: VALLEY.record.display_name },
    });
    fireEvent.click(confirm);

    await waitFor(() => expect(api.projects.remove).toHaveBeenCalledTimes(1));
    expect(api.projects.remove).toHaveBeenCalledWith({
      id: VALLEY.record.id,
      confirm_name: VALLEY.record.display_name,
      user: "grower",
    });
    await waitFor(() =>
      expect(useStore.getState().toasts.at(-1)?.message).toBe(
        `Removed ${VALLEY.record.display_name}. Archived as ${removed.archive_name}.`,
      ),
    );
  });

  it("renames a project's display name by its id", async () => {
    vi.mocked(api.projects.list).mockResolvedValue(LISTING);
    const { renamed } = PROJECT_FIXTURES;
    vi.mocked(api.projects.rename).mockResolvedValue(renamed);
    useStore.setState({ user: "grower" });
    render(<ProjectPicker />);

    fireEvent.click(await screen.findByText(VALLEY.record.display_name));
    fireEvent.click(screen.getByText("Rename…"));
    fireEvent.change(screen.getByLabelText("New name"), {
      target: { value: renamed.record.display_name },
    });
    fireEvent.click(screen.getByRole("button", { name: "Rename" }));

    await waitFor(() => expect(api.projects.rename).toHaveBeenCalledTimes(1));
    expect(api.projects.rename).toHaveBeenCalledWith({
      id: VALLEY.record.id,
      display_name: renamed.record.display_name,
      user: "grower",
    });
  });
});
