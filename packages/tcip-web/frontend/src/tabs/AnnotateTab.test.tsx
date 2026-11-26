// Auto-cleanup needs vitest globals (not enabled here), so clean up explicitly.
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

import { api, loadedLabels, type LoadedLabels, type LoadedSave } from "@/api/client";
import { sessionsApi } from "@/api/sessions";
import { subjectColor, subjectsApi } from "@/api/subjects";
import { CONF_REFUSED, type ImageBands, PLAIN } from "@/api/types.generated";
import { stateSocket } from "@/api/ws";
import { HelpOverlay } from "@/components/HelpOverlay";
import { NotificationsPane } from "@/components/Notifications";
import type { BandChoice } from "@/lib/bandSelection";
import { notifyCanvasStateRequest } from "@/lib/canvasSync";
import { dismissLabel } from "@/lib/dismissLabel";
import type { LoadedImage } from "@/lib/imageLoader";
import { CUT_MISSES_REFUSAL } from "@/lib/polygonGeometry";
import { refusedRegistryKey, saveRegistry } from "@/lib/registrySave";
import type { ReviewItem } from "@/lib/reviewItems";
import {
  CASING,
  FLAG_MARK,
  FOCUS_HALO,
  MATCH_COLORS,
  MATCH_TYPES,
  MATCH_WORDS,
  proposalLabel,
} from "@/lib/symbology";
import type { HostSize } from "@/lib/viewGeometry";
import { useStore } from "@/store";
import type { AnnotateView } from "@/store/types";
import { AnnotateTab } from "@/tabs/AnnotateTab";
import {
  ANNOTATE_ANSWERS,
  answerAnnotateMount,
  LANDED,
  LOADED,
  loadHolding,
  loadOf,
  POINT_SUBJECT,
  registryOf,
  SECOND_SUBJECT,
  TOOL_SELECTION,
} from "@/test/annotate";
import { BAND_ANSWERS } from "@/test/bands";
import { dockWithNotificationsOpen, measureWindow, PANE_FOCUS } from "@/test/dock";
import { LABEL_ANSWERS, servedAnnotation, UNSTORED } from "@/test/labels";
import { PANEL_EVENTS } from "@/test/panelEvents";
import { openedAs, PICKER, PROJECT_FIXTURES } from "@/test/projectRecords";
import { thrown } from "@/test/refusals";
import { SESSION_ANSWERS, sessionOf } from "@/test/sessions";
import {
  changeSelection,
  LATER_SELECTION,
  openTestProject,
  RAW_SELECTION,
  SELECTED_SUBJECT as SUBJECT,
  selected,
  TEST_PROJECT,
  testProject,
} from "@/test/store";

// Konva needs a real 2D canvas; these tests exercise label I/O ordering and
// store->canvas prop flow, not drawing. Render Konva shapes as inspectable divs
// and CanvasStage as a passthrough so AnnotationShapes' memo behavior is intact.
vi.mock("konva", () => ({ default: {} }));
vi.mock("react-konva", () => ({
  Group: (props: { children?: React.ReactNode }) => <>{props.children}</>,
  Rect: (props: {
    stroke?: string;
    dash?: number[];
    fill?: string;
    opacity?: number;
    shadowColor?: string;
  }) => (
    <div
      data-testid="k-rect"
      data-stroke={props.stroke}
      data-dash={props.dash ? "true" : undefined}
      data-fill={props.fill}
      data-opacity={props.opacity}
      data-shadow={props.shadowColor}
    />
  ),
  Line: (props: {
    stroke?: string;
    points?: number[];
    closed?: boolean;
    dash?: number[];
    opacity?: number;
    shadowColor?: string;
  }) => (
    <div
      data-testid="k-line"
      data-stroke={props.stroke}
      data-closed={props.closed ? "true" : undefined}
      data-dash={props.dash ? "true" : undefined}
      data-points={props.points?.join(",")}
      data-opacity={props.opacity}
      data-shadow={props.shadowColor}
    />
  ),
  Circle: (props: { x?: number; y?: number; fill?: string; radius?: number }) => (
    <div
      data-testid="k-circle"
      data-x={props.x}
      data-y={props.y}
      data-fill={props.fill}
      data-radius={props.radius}
    />
  ),
  Text: (props: { text?: string; fill?: string }) => (
    <div data-testid="k-text" data-text={props.text} data-fill={props.fill} />
  ),
}));
// Pixel handlers are forwarded as plain mouse events (clientX/Y stand in for image-pixel coords;
// the real screen<->image conversion is CanvasStage's own concern) so tests can drive the drawing
// tools without a real Konva stage.
// What the stand-in stage measures and serves, for a test to set before it renders or to report
// anew once it has: `size` is the host size it reports, `facts` the base serve's facts.
const stage = vi.hoisted(() => ({
  size: null as HostSize | null,
  facts: null as LoadedImage | null,
  reportSize: (_size: HostSize | null) => {},
}));
vi.mock("@/components/Canvas/CanvasStage", async () => {
  const { useEffect } = await import("react");
  return {
    CanvasStage: (props: {
      children?: React.ReactNode;
      overlay?: React.ReactNode;
      imageUrl?: string | null;
      onHostSize: (size: HostSize | null) => void;
      onBaseFacts: (facts: LoadedImage | null) => void;
      onPixelDown?: (x: number, y: number, ev: unknown) => void;
      onPixelMove?: (x: number, y: number, ev: unknown) => void;
      onPixelUp?: (x: number, y: number, ev: unknown) => void;
      onPixelClick?: (x: number, y: number, ev: unknown) => void;
      onPixelDoubleClick?: (x: number, y: number, ev: unknown) => void;
      onPixelContextMenu?: (x: number, y: number, ev: unknown) => void;
    }) => {
      const { onHostSize, onBaseFacts } = props;
      useEffect(() => {
        stage.reportSize = onHostSize;
        onHostSize(stage.size);
        onBaseFacts(stage.facts);
      }, [onHostSize, onBaseFacts]);
      return (
        <div
          data-testid="canvas-stage"
          role="presentation"
          data-image-url={props.imageUrl ?? ""}
          onMouseDown={(e) =>
            props.onPixelDown?.(e.clientX, e.clientY, { evt: { button: e.button } })
          }
          onMouseMove={(e) => props.onPixelMove?.(e.clientX, e.clientY, { evt: { buttons: 1 } })}
          onMouseUp={(e) => props.onPixelUp?.(e.clientX, e.clientY, { evt: {} })}
          onClick={(e) => props.onPixelClick?.(e.clientX, e.clientY, { evt: { button: e.button } })}
          onDoubleClick={(e) => props.onPixelDoubleClick?.(e.clientX, e.clientY, { evt: {} })}
          onContextMenu={(e) =>
            props.onPixelContextMenu?.(e.clientX, e.clientY, {
              evt: { button: 2, preventDefault: () => {} },
            })
          }
        >
          {props.children}
          {props.overlay}
        </div>
      );
    },
  };
});
vi.mock("@/components/AnnotateToolbar", async () => {
  const { SUBJECT_CONTROL_ID } =
    await vi.importActual<typeof import("@/lib/registrySave")>("@/lib/registrySave");
  return {
    AnnotateToolbar: (props: {
      bandChoice: BandChoice;
      subjectState: string | null;
      onComplete: (next: boolean) => void;
      onCancelSave: () => void;
      proposing: boolean;
      onPhrase: (phrase: string) => void;
      onPropose: () => void;
      view: AnnotateView;
      onView: (next: AnnotateView) => void;
      predictionsAvailable: boolean;
      focused: ReviewItem | null;
    }) => (
      <div
        data-testid="toolbar"
        data-band-choice={props.bandChoice.kind}
        data-band-count={props.bandChoice.kind === "banded" ? props.bandChoice.info.band_count : ""}
        data-subject-state={props.subjectState ?? ""}
        data-proposing={props.proposing}
        data-view-proposals={props.view.proposals}
        data-view-predictions={props.view.predictions}
        data-predictions-available={props.predictionsAvailable}
        data-focused={
          props.focused?.kind === "annotation"
            ? `annotation ${props.focused.shape} ${props.focused.ref}`
            : (props.focused?.kind ?? "")
        }
      >
        <button onClick={() => props.onComplete(true)}>toolbar-complete</button>
        <button onClick={props.onCancelSave}>toolbar-cancel-save</button>
        <button onClick={() => props.onPhrase("open bud")}>toolbar-phrase</button>
        <button onClick={props.onPropose}>toolbar-propose</button>
        <button onClick={() => props.onView({ ...props.view, proposals: !props.view.proposals })}>
          toolbar-view-proposals
        </button>
        <button
          onClick={() => props.onView({ ...props.view, predictions: !props.view.predictions })}
        >
          toolbar-view-predictions
        </button>
        <button id={SUBJECT_CONTROL_ID}>toolbar-subject</button>
      </div>
    ),
  };
});

// The key that hides or shows every proposal and prediction at once.
const toggleProposals = () => fireEvent.keyDown(window, { key: "h" });

const initialStoreState = useStore.getState();

// Each photograph's produced load carries its own token, so a save's echoed token identifies
// which image's load it came from.
const [IMG1, IMG2] = RAW_SELECTION.images.map((image) => image.path);
const IMG1_NAME = RAW_SELECTION.images[0].name;

/** A one-ring triangle near the frame's corner. */
const TRIANGLE: [number, number][][] = [
  [
    [0, 0],
    [10, 0],
    [10, 10],
  ],
];

/** The save's refusal of a document that changed since it was loaded, as the client throws it. */
const versionConflict = () => thrown(LABEL_ANSWERS.version_conflict);

/** The sentence that refusal states, which the tab shows. */
const CONFLICT_SENTENCE = LABEL_ANSWERS.version_conflict.body.detail.message;

function setupDataset() {
  openTestProject(selected({ subject: SUBJECT, count: 2 }), {
    mode: "box",
    active_subject: SUBJECT,
  });
}

function addBox() {
  useStore
    .getState()
    .add("boxes", { x1: 10, y1: 10, x2: 50, y2: 50, subject: SUBJECT, attributes: {} });
}

const flush = () => act(async () => {});

// Save now lives in the (mocked-out) toolbar's Editor shelf; drive it via its Ctrl+S shortcut.
const pressSave = () => fireEvent.keyDown(window, { key: "s", ctrlKey: true });

/** The entries the canvas shows in place, for the image in view. */
const onCanvas = () => screen.queryAllByTestId("entry-in-place");

/** The entries the notifications area lists. */
const listed = () => useStore.getState().notifications;

let loadSpy: MockInstance<typeof api.annotate.load>;
let saveSpy: MockInstance<typeof api.annotate.save>;

/** Hold every save until the test answers it; each save joins the list, oldest first. */
function heldSaves(): PromiseWithResolvers<LoadedSave>[] {
  const pending: PromiseWithResolvers<LoadedSave>[] = [];
  saveSpy.mockImplementation(() => {
    const save = Promise.withResolvers<LoadedSave>();
    pending.push(save);
    return save.promise;
  });
  return pending;
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  useStore.getState().setUser("grower");
  setupDataset();
  loadSpy = answerAnnotateMount();
  saveSpy = vi.spyOn(api.annotate, "save").mockResolvedValue(LANDED.box);
  // jsdom lays nothing out, so the stand-in stage reports the host size the viewport math needs.
  stage.size = { w: 1000, h: 800 };
  stage.facts = null;
  // One jsdom instance per file, not per test: a recolor left by an earlier test must not leak
  // into a later one's derived-color assertions.
  try {
    localStorage.removeItem("tcip.annotate.subjectColors");
  } catch {
    /* not available in this environment, nothing to clear */
  }
});

afterEach(cleanup);

describe("AnnotateTab session contributions", () => {
  async function leaveFirstImage() {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(() => {
      changeSelection({ current_image_index: 1 });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
  }

  it("posts a named visit with its person and activity, and retires it once accepted", async () => {
    useStore.setState({ user: "jordan" });
    await leaveFirstImage();

    expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(1);
    expect(vi.mocked(sessionsApi.imageEvent).mock.calls[0][0]).toMatchObject({
      image_name: IMG1_NAME,
      activity: "review",
      user: "jordan",
    });
    expect(useStore.getState().heldContributions).toEqual([]);
  });

  it("posts the open visit when the tab unmounts, and retires it once accepted", async () => {
    useStore.setState({ user: "jordan" });
    const view = render(<AnnotateTab />);
    await waitFor(() =>
      expect(useStore.getState().sessionTracking.currentImageName).toBe(IMG1_NAME),
    );
    expect(sessionsApi.imageEvent).not.toHaveBeenCalled();

    view.unmount();
    await flush();

    expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(1);
    expect(vi.mocked(sessionsApi.imageEvent).mock.calls[0][0]).toMatchObject({
      image_name: IMG1_NAME,
      user: "jordan",
    });
    expect(useStore.getState().heldContributions).toEqual([]);
  });

  it("opens a visit of the image whose edits a remount keeps, and posts it when the tab unmounts again", async () => {
    useStore.setState({ user: "jordan" });
    const first = render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox);
    saveSpy.mockRejectedValueOnce(await versionConflict());
    first.unmount();
    await flush();
    const second = render(<AnnotateTab />);
    await flush();
    expect(loadSpy).toHaveBeenCalledTimes(1);
    expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(1);

    second.unmount();
    await flush();

    expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(2);
    expect(vi.mocked(sessionsApi.imageEvent).mock.calls[1][0]).toMatchObject({
      image_name: IMG1_NAME,
      user: "jordan",
    });
  });

  it("records the session an accepted visit landed in, by the project and the answered stamp", async () => {
    useStore.setState({ user: "jordan" });
    await leaveFirstImage();

    expect(useStore.getState().recordedSession).toEqual({
      project_id: useStore.getState().openProject!.id,
      started: sessionOf(SESSION_ANSWERS.landed).started,
      user: "jordan",
    });
  });

  it("records no session for a visit the backend recorded nothing of", async () => {
    useStore.setState({ user: "jordan" });
    vi.mocked(sessionsApi.imageEvent).mockResolvedValueOnce(SESSION_ANSWERS.empty);
    await leaveFirstImage();

    expect(useStore.getState().heldContributions).toEqual([]);
    expect(useStore.getState().recordedSession).toBeNull();
  });

  it("holds a visit the backend did not accept", async () => {
    useStore.setState({ user: "jordan" });
    vi.mocked(sessionsApi.imageEvent).mockRejectedValueOnce(new Error("backend down"));
    await leaveFirstImage();

    expect(useStore.getState().heldContributions).toHaveLength(1);
  });

  it("posts a held visit to its own project once another is open", async () => {
    useStore.setState({ user: "jordan" });
    vi.mocked(sessionsApi.imageEvent).mockRejectedValueOnce(new Error("backend down"));
    await leaveFirstImage();
    const own = useStore.getState().openProject!.id;

    act(() =>
      useStore.setState({ openProject: testProject({ id: "other0000000", path: "C:/other" }) }),
    );
    await flush();

    expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(2);
    expect(vi.mocked(sessionsApi.imageEvent).mock.calls[1][0].project_id).toBe(own);
  });

  it("retires a visit the backend committed without its line, adopting the session it answered", async () => {
    useStore.setState({ user: "jordan" });
    const gap = SESSION_ANSWERS.audit_gap;
    vi.mocked(sessionsApi.imageEvent).mockRejectedValueOnce(await thrown(gap));
    await leaveFirstImage();
    act(() => useStore.setState({ openProject: { ...useStore.getState().openProject! } }));
    await flush();

    expect(sessionsApi.imageEvent).toHaveBeenCalledTimes(1);
    expect(useStore.getState().heldContributions).toEqual([]);
    expect(useStore.getState().recordedSession).toEqual({
      project_id: useStore.getState().openProject!.id,
      started: sessionOf(gap.body.detail.what).started,
      user: "jordan",
    });
  });
});

describe("AnnotateTab save/load race", () => {
  it("saves to the loaded path and re-echoes the returned mtime on the next save", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    saveSpy.mockResolvedValueOnce(LANDED.box);
    act(addBox);
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].image_path).toBe(IMG1);
    expect(saveSpy.mock.calls[0][0].base_mtime).toBe(loadOf(IMG1).base_mtime);
    expect(useStore.getState().canvas.dirty).toBe(false);

    // Second save on the same image must echo the mtime the first save returned.
    act(addBox);
    pressSave();
    await flush();
    expect(saveSpy).toHaveBeenCalledTimes(2);
    expect(LANDED.box.document.base_mtime).not.toBe(loadOf(IMG1).base_mtime);
    expect(saveSpy.mock.calls[1][0].base_mtime).toBe(LANDED.box.document.base_mtime);
  });

  it("a flush save resolving after navigation does not rewind the loaded path or wipe dirty", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // Dirty img1, then navigate. flushLeaving() fires the save without awaiting
    // it; hold its response so the img2 load resolves first (slow label write vs
    // cached read): this interleaving must never corrupt cross-image GT.
    act(addBox);
    const flushSave = Promise.withResolvers<LoadedSave>();
    saveSpy.mockReturnValueOnce(flushSave.promise);
    act(() => {
      changeSelection({ current_image_index: 1 });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].image_path).toBe(IMG1);
    expect(saveSpy.mock.calls[0][0].base_mtime).toBe(loadOf(IMG1).base_mtime);

    // Edit img2 while the img1 save is still in flight, then let it resolve late.
    act(addBox);
    await act(() => Promise.resolve(flushSave.resolve(LANDED.box)));

    // The stale result must not markClean() the img2 edits...
    expect(useStore.getState().canvas.dirty).toBe(true);

    // ...and the next save must target img2 with img2's loaded mtime, not
    // img1's file with the stale save's echoed mtime.
    pressSave();
    await flush();
    expect(saveSpy).toHaveBeenCalledTimes(2);
    expect(saveSpy.mock.calls[1][0].image_path).toBe(IMG2);
    expect(loadOf(IMG2).base_mtime).not.toBe(loadOf(IMG1).base_mtime);
    expect(saveSpy.mock.calls[1][0].base_mtime).toBe(loadOf(IMG2).base_mtime);
  });

  it("an earlier image's save settling does not release the hold of the later image's save", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    const pending = heldSaves();
    act(() => {
      changeSelection({ current_image_index: 1 });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    act(addBox);
    pressSave();
    await flush();
    expect(pending).toHaveLength(2);
    expect(useStore.getState().canvas.saving?.image).toBe(IMG2);

    await act(() => Promise.resolve(pending[0].resolve(LANDED.box)));
    expect(useStore.getState().canvas.saving?.image).toBe(IMG2);
    act(addBox);
    expect(useStore.getState().canvas.boxes).toHaveLength(1);

    await act(() => Promise.resolve(pending[1].resolve(LANDED.cleared)));
    expect(useStore.getState().canvas.saving).toBeNull();
    expect(useStore.getState().canvas.boxes).toHaveLength(0);
  });

  describe("a save in flight when the editor unmounts and a new one mounts", () => {
    async function departing() {
      const first = render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
      await flush();
      act(addBox);
      const save = Promise.withResolvers<LoadedSave>();
      let signal!: AbortSignal;
      saveSpy.mockImplementationOnce((_body, s) => {
        signal = s as AbortSignal;
        return save.promise;
      });
      pressSave();
      await flush();
      first.unmount();
      render(<AnnotateTab />);
      await flush();
      return { answer: save.resolve, signal };
    }

    it("is kept by the new editor, which sends no save and loads nothing over it", async () => {
      await departing();
      expect(loadSpy).toHaveBeenCalledTimes(1);
      expect(saveSpy).toHaveBeenCalledTimes(1);
      expect(useStore.getState().canvas.saving).not.toBeNull();
      expect(useStore.getState().canvas.boxes).toHaveLength(1);
    });

    it("is canceled by the toolbar of the new editor, which keeps the edits", async () => {
      const { signal } = await departing();
      expect(useStore.getState().canvas.saving).not.toBeNull();
      fireEvent.click(screen.getByRole("button", { name: "toolbar-cancel-save" }));
      expect(signal.aborted).toBe(true);
      expect(useStore.getState().canvas.saving).toBeNull();
      expect(useStore.getState().canvas.boxes).toHaveLength(1);
      expect(useStore.getState().canvas.dirty).toBe(true);
    });

    it("lands into the canvas the new editor kept, which adopts its answer", async () => {
      const { answer } = await departing();
      await act(() => Promise.resolve(answer(LANDED.box)));
      expect(useStore.getState().canvas.saving).toBeNull();
      expect(useStore.getState().canvas.dirty).toBe(false);
      expect(useStore.getState().canvas.loaded?.base_mtime).toBe(LANDED.box.document.base_mtime);
    });
  });

  it("canceling reports nothing over an image loaded after the save began", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox);
    const save = Promise.withResolvers<LoadedSave>();
    saveSpy.mockReturnValueOnce(save.promise);
    pressSave();
    await flush();
    fireEvent.click(screen.getByRole("button", { name: "toolbar-cancel-save" }));
    act(() => {
      changeSelection({ current_image_index: 1 });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    await act(() => Promise.resolve(save.reject(new DOMException("aborted", "AbortError"))));
    expect(screen.queryByText(/Save canceled/)).not.toBeInTheDocument();
    expect(listed()).toEqual([]);
  });

  describe("a save from an editor that has since unmounted", () => {
    /** Unmount over a save held in flight, mount again, and try an edit there. */
    async function remountedWithEdit() {
      const first = render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
      await flush();
      act(addBox);
      const pending = heldSaves();
      pressSave();
      await flush();
      first.unmount();
      render(<AnnotateTab />);
      await flush();
      act(addBox);
      return pending;
    }

    it("leaves the remounted editor no edit while it holds the canvas, and one once it is adopted", async () => {
      const pending = await remountedWithEdit();
      expect(useStore.getState().canvas.boxes).toHaveLength(1);
      await act(() => Promise.resolve(pending[0].resolve(LANDED.box)));
      expect(useStore.getState().canvas.dirty).toBe(false);
      act(addBox);
      expect(useStore.getState().canvas.boxes).toHaveLength(2);
      expect(useStore.getState().canvas.dirty).toBe(true);
    });

    it("sends no save of the remounted editor while it holds the canvas, and the next save carries its answer's version", async () => {
      const pending = await remountedWithEdit();
      pressSave();
      await flush();
      expect(pending).toHaveLength(1);
      await act(() => Promise.resolve(pending[0].resolve(LANDED.box)));
      act(addBox);
      pressSave();
      await flush();
      expect(pending).toHaveLength(2);
      expect(saveSpy.mock.calls[1][0].base_mtime).toBe(LANDED.box.document.base_mtime);
    });
  });

  it("canceling a save in flight releases the canvas, keeps the edits and drops a late answer", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox);
    const late = Promise.withResolvers<LoadedSave>();
    let signal: AbortSignal | undefined;
    saveSpy.mockImplementationOnce((_body, s) => {
      signal = s;
      return late.promise;
    });
    pressSave();
    await flush();
    fireEvent.click(screen.getByRole("button", { name: "toolbar-cancel-save" }));
    await flush();

    expect(signal?.aborted).toBe(true);
    expect(useStore.getState().canvas.saving).toBeNull();
    act(addBox);
    expect(useStore.getState().canvas.boxes).toHaveLength(2);
    await act(() => Promise.resolve(late.resolve(LANDED.cleared)));
    expect(useStore.getState().canvas.boxes).toHaveLength(2);
    expect(useStore.getState().canvas.dirty).toBe(true);
  });

  it("a save whose request fails releases the canvas", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    saveSpy.mockRejectedValueOnce(new Error("network down"));
    pressSave();
    await flush();
    expect(useStore.getState().canvas.saving).toBeNull();
    act(addBox);
    expect(useStore.getState().canvas.boxes).toHaveLength(2);
  });

  it("a stale conflict for a since-left image is listed naming it, with nothing on the new image's canvas", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // Interactive Ctrl+S save held in flight, then the user navigates away.
    act(addBox);
    const pending = heldSaves();
    pressSave();
    act(() => {
      changeSelection({ current_image_index: 1 });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();

    // The interactive save 409s late; the navigation's flush sent none while it held the canvas.
    expect(pending).toHaveLength(1);
    const conflict = await versionConflict();
    await act(() => {
      for (const save of pending) save.reject(conflict);
      return Promise.resolve();
    });
    expect(screen.queryByText("Reload")).not.toBeInTheDocument();
    expect(screen.queryByText(CONFLICT_SENTENCE, { exact: false })).not.toBeInTheDocument();
    expect(onCanvas()).toHaveLength(0);
    expect(listed()).toEqual([
      expect.objectContaining({
        kind: "failure",
        sentence: `Save failed for ${IMG1_NAME}: ${CONFLICT_SENTENCE}`,
        key: undefined,
        context: { project: TEST_PROJECT.id, target: IMG1_NAME },
      }),
    ]);
  });

  it("a save failing after its own image was reloaded is listed naming that image, with nothing on the canvas", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox);
    saveSpy.mockRejectedValueOnce(await versionConflict());
    pressSave();
    await flush();

    // A second save goes out under the conflict's entry, then its Reload re-reads the same image.
    const pending = heldSaves();
    pressSave();
    await flush();
    expect(pending).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Reload" }));
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    expect(loadSpy.mock.calls[1][0]).toBe(IMG1);

    await act(() => Promise.resolve(pending[0].reject(new Error("network down"))));
    expect(screen.queryByText(/Could not save annotations/)).not.toBeInTheDocument();
    expect(onCanvas()).toHaveLength(0);
    expect(listed().at(-1)).toMatchObject({
      kind: "failure",
      level: "error",
      sentence: `Save failed: ${IMG1_NAME}'s edits were not written.`,
      key: undefined,
      context: { target: IMG1_NAME },
    });
  });

  it("names the person the app is set to on the save, and sends no provenance of its own", async () => {
    act(() => useStore.getState().setUser("breeder"));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    pressSave();
    await flush();

    const body = saveSpy.mock.calls[0][0];
    expect(body.user).toBe("breeder");
    for (const a of body.annotations) {
      expect(Object.keys(a).filter((k) => /_(by|at)$/.test(k))).toEqual([]);
    }
  });
});

describe("AnnotateTab subject rendering", () => {
  it("renders a box with the subject-derived color, named on selection", async () => {
    useStore.getState().setRegistry(registryOf(POINT_SUBJECT), UNSTORED);
    useStore.setState((s) => ({ gui: { ...s.gui, active_subject: POINT_SUBJECT } }));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(() =>
      useStore
        .getState()
        .add("boxes", { x1: 10, y1: 10, x2: 50, y2: 50, subject: POINT_SUBJECT, attributes: {} }),
    );
    // Color is GUI-local (name-derived); the label is the subject name, no integer id, and
    // appears on selection (labels are hover/selection-only; the legend is the standing key).
    expect(screen.getByTestId("k-rect")).toHaveAttribute(
      "data-stroke",
      subjectColor(POINT_SUBJECT),
    );
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), { clientX: 30, clientY: 30 });
    expect(screen.getAllByTestId("k-text")[0]).toHaveAttribute("data-text", POINT_SUBJECT);
  });

  it("box mode draws an active-subject polygon's read-only derived box (no handles), never a stored box", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    const poly = servedAnnotation({ subject: SUBJECT, rings: TRIANGLE, index: 0 });
    loadSpy.mockImplementation((imagePath) => Promise.resolve(loadHolding(imagePath, [poly])));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // Box mode (setupDataset). The polygon shows only its derived box: a single Rect with no corner
    // handles (handles are extra Rects), never in canvas.boxes, so unsaveable; read-only is structural.
    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(1);
    expect(rects[0]).not.toHaveAttribute("data-dash");
    expect(rects[0]).toHaveAttribute("data-stroke", subjectColor(SUBJECT));
    expect(useStore.getState().canvas.boxes).toHaveLength(0);
    expect(useStore.getState().canvas.polygons).toHaveLength(1);
  });

  it("a tool-authored polygon's derived box draws in the polygon's own line style, unlabeled", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    const [poly] = LOADED.tool_polygon.polygons;
    expect(poly.authorship).toBe("tool");
    loadSpy.mockImplementation(() => Promise.resolve(LOADED.tool_polygon));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // Line style says whose shape it is: the derived box is the tool's polygon's, so it draws
    // dotted like the polygon; it is no item of its own, so nothing labels it.
    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(1);
    expect(rects[0]).toHaveAttribute("data-dash", "true");
    expect(screen.queryAllByTestId("k-text")).toHaveLength(0);
  });

  it("box mode draws a person's editable box solid and a person's polygon's derived box solid too", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(1);
    expect(rects[0]).not.toHaveAttribute("data-dash");
  });

  it("point mode draws a placed point as its reticle mark, never a box or a closed outline", async () => {
    // A point asserts a location and no extent: rendering it as a box (or letting it render as a
    // degenerate polygon) would show the annotator an extent the annotation does not claim.
    useStore.getState().setRegistry(registryOf(POINT_SUBJECT), UNSTORED);
    useStore.setState((s) => ({
      gui: { ...s.gui, mode: "point" as const, active_subject: POINT_SUBJECT },
    }));
    const point = servedAnnotation({ subject: POINT_SUBJECT, point: [100, 200], index: 0 });
    loadSpy.mockImplementation((imagePath) => Promise.resolve(loadHolding(imagePath, [point])));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    // The core sits exactly on the coordinate, in the subject's color...
    const core = screen.getByTestId("k-circle");
    expect(core).toHaveAttribute("data-x", "100");
    expect(core).toHaveAttribute("data-y", "200");
    expect(core).toHaveAttribute("data-fill", subjectColor(POINT_SUBJECT));
    // ...with four radial ticks (the mark that reads as a location, not a tiny shape)...
    expect(screen.getAllByTestId("k-line")).toHaveLength(4);
    // ...and no box of any kind. (Naming on selection/hover is covered by the label tests.)
    expect(screen.queryAllByTestId("k-rect")).toHaveLength(0);
  });

  it("polygon mode draws every ring of an occlusion-split shape, labeled once", async () => {
    // An organ behind a branch loads as one annotation with two disjoint regions. Drawing only
    // the first would show the breeder part of the object and let them confirm it as the whole.
    useStore.getState().setRegistry(registryOf(POINT_SUBJECT), UNSTORED);
    useStore.setState((s) => ({
      gui: { ...s.gui, mode: "polygon" as const, active_subject: POINT_SUBJECT },
    }));
    const rings: [number, number][][] = [
      [
        [0, 0],
        [10, 0],
        [10, 10],
      ],
      [
        [40, 40],
        [60, 40],
        [60, 60],
      ],
    ];
    const shape = servedAnnotation({ subject: POINT_SUBJECT, rings, index: 0 });
    loadSpy.mockImplementation((imagePath) => Promise.resolve(loadHolding(imagePath, [shape])));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    const lines = screen.getAllByTestId("k-line");
    expect(lines).toHaveLength(2);
    expect(lines.map((l) => l.getAttribute("data-points"))).toEqual([
      "0,0,10,0,10,10",
      "40,40,60,40,60,60",
    ]);
    // Both parts wear the annotation's own color...
    expect(lines.every((l) => l.getAttribute("data-stroke") === subjectColor(POINT_SUBJECT))).toBe(
      true,
    );
    // ...and selecting it names the annotation once, not once per ring (HaloLabel = halo + fill).
    fireEvent.click(screen.getByTestId("canvas-stage"), { clientX: 8, clientY: 5 });
    expect(
      screen.getAllByTestId("k-text").filter((t) => t.getAttribute("data-text") === POINT_SUBJECT),
    ).toHaveLength(2);
  });
});

describe("AnnotateTab authorship symbology", () => {
  it("a tool's box and an unattributed one draw dotted, a person's and an accepted tool's solid; focus names the tool's", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    const { boxes } = LOADED.authored;
    expect(boxes.map((box) => box.authorship)).toEqual([
      "tool",
      "person",
      "tool_accepted",
      "unattributed",
    ]);
    loadSpy.mockImplementation(() => Promise.resolve(LOADED.authored));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    const rects = screen.getAllByTestId("k-rect");
    expect(rects).toHaveLength(4);
    expect(rects[0]).toHaveAttribute("data-dash", "true"); // tool: dotted
    expect(rects[1]).not.toHaveAttribute("data-dash"); // person: solid
    expect(rects[2]).not.toHaveAttribute("data-dash"); // tool_accepted: solid
    expect(rects[3]).toHaveAttribute("data-dash", "true"); // unattributed: no person behind it

    // Focusing the tool box (a press inside it) names it with the authorship it draws with.
    const [tool] = boxes;
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), {
      clientX: (tool.x1 + tool.x2) / 2,
      clientY: (tool.y1 + tool.y2) / 2,
      button: 0,
    });
    await flush();
    expect(
      screen
        .getAllByTestId("k-text")
        .some((t) => t.getAttribute("data-text") === `${SUBJECT}, tool`),
    ).toBe(true);
  });

  it("the legend states the dotted stroke means proposed", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(screen.getByText("Dotted: proposed")).toBeInTheDocument();
  });
});

describe("AnnotateTab point tool", () => {
  const stage = () => screen.getByTestId("canvas-stage");
  // The move handler is rAF-throttled; jsdom fires rAF off a timer, so let one frame land.
  const frame = () => act(async () => void (await new Promise((r) => setTimeout(r, 25))));

  async function mountPointMode(points: { x: number; y: number }[] = []) {
    useStore.getState().setRegistry(registryOf(POINT_SUBJECT), UNSTORED);
    useStore.setState((s) => ({
      gui: { ...s.gui, mode: "point" as const, active_subject: POINT_SUBJECT },
    }));
    const placed = points.map(({ x, y }, index) =>
      servedAnnotation({ subject: POINT_SUBJECT, point: [x, y], index }),
    );
    loadSpy.mockImplementation((imagePath) => Promise.resolve(loadHolding(imagePath, placed)));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
  }

  it("a single click commits one point at the clicked coordinate", async () => {
    await mountPointMode();
    fireEvent.click(stage(), { clientX: 120, clientY: 340 });
    await flush();

    // One click is the whole gesture: no drag-out, no second click to close.
    expect(useStore.getState().canvas.points).toEqual([
      { x: 120, y: 340, subject: POINT_SUBJECT, attributes: {}, iscrowd: false },
    ]);
    expect(useStore.getState().canvas.dirty).toBe(true);
  });

  it("refuses to place a point with no subject selected, and says so once on the canvas", async () => {
    await mountPointMode();
    act(() => useStore.getState().setActiveSubject(null));
    fireEvent.click(stage(), { clientX: 10, clientY: 10 });
    fireEvent.click(stage(), { clientX: 30, clientY: 30 });
    await flush();

    expect(useStore.getState().canvas.points).toHaveLength(0);
    const [entry] = onCanvas();
    expect(entry).toHaveTextContent("Choose a subject first.");
    expect(listed()).toEqual([
      expect.objectContaining({ kind: "failure", sentence: "Choose a subject first.", count: 1 }),
    ]);
  });

  it("a press-drag repositions a placed point, with one undo snapshot for the whole drag", async () => {
    await mountPointMode([{ x: 100, y: 100 }]);
    expect(useStore.getState().canvas.undoStack).toHaveLength(0);

    fireEvent.mouseDown(stage(), { clientX: 102, clientY: 101, button: 0 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "point", index: 0 });
    fireEvent.mouseMove(stage(), { clientX: 300, clientY: 250 });
    await frame();
    fireEvent.mouseMove(stage(), { clientX: 310, clientY: 260 });
    await frame();
    fireEvent.mouseUp(stage(), { clientX: 310, clientY: 260 });
    await flush();

    expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 310, y: 260 });
    // One snapshot for the gesture: a per-move push would evict the whole 30-entry history.
    expect(useStore.getState().canvas.undoStack).toHaveLength(1);
    act(() => useStore.getState().undo());
    expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 100, y: 100 });
  });

  it("a drag in progress when a save starts ends there and does not resume on the adopted document", async () => {
    await mountPointMode([{ x: 100, y: 100 }]);
    fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 });
    fireEvent.mouseMove(stage(), { clientX: 200, clientY: 200 });
    await frame();
    const answer = Promise.withResolvers<LoadedSave>();
    saveSpy.mockReturnValueOnce(answer.promise);
    pressSave();
    await flush();
    // The save's answer holds the point where the drag left it.
    const [moved] = LANDED.point.document.points;
    await act(() => Promise.resolve(answer.resolve(LANDED.point)));

    fireEvent.mouseMove(stage(), { clientX: 400, clientY: 400 });
    await frame();
    expect(useStore.getState().canvas.points[0]).toMatchObject({ x: moved.x, y: moved.y });
    expect(useStore.getState().canvas.undoStack).toHaveLength(0);
  });

  describe("with the canvas held by a save", () => {
    const hold = () => act(() => void useStore.getState().holdForSave("x"));
    const unhold = () =>
      act(() => {
        const held = useStore.getState().canvas.saving;
        if (held) useStore.getState().releaseSave(held);
      });

    it("a click leaves the focus where it was", async () => {
      await mountPointMode([{ x: 100, y: 100 }]);
      act(() => useStore.getState().setFocus({ kind: "point", index: 0 }));
      hold();
      fireEvent.click(stage(), { clientX: 500, clientY: 500 });
      expect(useStore.getState().canvas.focus).toEqual({ kind: "point", index: 0 });
    });

    it("a right-click leaves the focus where it was", async () => {
      await mountPointMode([{ x: 100, y: 100 }]);
      act(() => useStore.getState().setFocus({ kind: "point", index: 0 }));
      hold();
      fireEvent.contextMenu(stage(), { clientX: 500, clientY: 500 });
      expect(useStore.getState().canvas.focus).toEqual({ kind: "point", index: 0 });
    });

    it("a move queued before the hold does nothing when its frame fires", async () => {
      useStore.getState().setRegistry(registryOf(POINT_SUBJECT), UNSTORED);
      useStore.setState((s) => ({
        gui: { ...s.gui, mode: "polygon" as const, active_subject: POINT_SUBJECT },
      }));
      const rings: [number, number][][] = [
        [
          [100, 100],
          [200, 100],
          [200, 200],
        ],
      ];
      const shape = servedAnnotation({ subject: POINT_SUBJECT, rings, index: 0 });
      loadSpy.mockImplementation((imagePath) => Promise.resolve(loadHolding(imagePath, [shape])));
      render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
      await flush();
      fireEvent.mouseMove(stage(), { clientX: 180, clientY: 150 });
      hold();
      await frame();
      expect(useStore.getState().annotateUi.hoveredPolygonIdx).toBeNull();
    });

    it("Escape ends a point drag in progress, which later movement does not resume", async () => {
      await mountPointMode([{ x: 100, y: 100 }]);
      fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 });
      fireEvent.keyDown(window, { key: "Escape" });
      fireEvent.mouseMove(stage(), { clientX: 300, clientY: 300 });
      await frame();
      expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 100, y: 100 });
    });

    it("a double-click leaves a freehand stream running", async () => {
      useStore.getState().setRegistry(registryOf(POINT_SUBJECT), UNSTORED);
      useStore.setState((s) => ({
        gui: { ...s.gui, mode: "polygon" as const, active_subject: POINT_SUBJECT },
      }));
      useStore.getState().setStream(true);
      render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
      await flush();
      fireEvent.click(stage(), { clientX: 300, clientY: 300 });
      expect(useStore.getState().canvas.currentPolygon).toHaveLength(1);
      hold();
      fireEvent.doubleClick(stage(), { clientX: 300, clientY: 300 });
      unhold();
      fireEvent.mouseMove(stage(), { clientX: 500, clientY: 500 });
      await frame();
      expect(useStore.getState().canvas.currentPolygon).toHaveLength(2);
    });

    it("a release leaves the drag in progress for the save that ends it", async () => {
      await mountPointMode([{ x: 100, y: 100 }]);
      fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 });
      hold();
      fireEvent.mouseUp(stage(), { clientX: 100, clientY: 100 });
      unhold();
      fireEvent.mouseMove(stage(), { clientX: 300, clientY: 300 });
      await frame();
      expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 300, y: 300 });
    });
  });

  it("the click that ends a drag does not place a second point on top of the moved one", async () => {
    await mountPointMode([{ x: 100, y: 100 }]);
    fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 });
    fireEvent.mouseMove(stage(), { clientX: 150, clientY: 150 });
    await frame();
    fireEvent.mouseUp(stage(), { clientX: 150, clientY: 150 });
    fireEvent.click(stage(), { clientX: 150, clientY: 150 }); // the release's trailing click
    await flush();

    expect(useStore.getState().canvas.points).toHaveLength(1);
  });

  it("right-click removes the point under the cursor and leaves a neighbor alone", async () => {
    await mountPointMode([
      { x: 100, y: 100 },
      { x: 400, y: 400 },
    ]);
    fireEvent.contextMenu(stage(), { clientX: 103, clientY: 100 });
    await flush();

    expect(useStore.getState().canvas.points).toHaveLength(1);
    expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 400, y: 400 });
  });

  it("Delete removes the selected point", async () => {
    await mountPointMode([{ x: 100, y: 100 }]);
    fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 });
    fireEvent.mouseUp(stage(), { clientX: 100, clientY: 100 });
    await flush();
    expect(useStore.getState().canvas.focus).toEqual({ kind: "point", index: 0 });

    fireEvent.keyDown(window, { key: "Delete" });
    await flush();
    expect(useStore.getState().canvas.points).toHaveLength(0);
    expect(useStore.getState().canvas.focus).toBeNull();
  });

  it("saves a placed point as a `point` payload the save route can author", async () => {
    await mountPointMode();
    fireEvent.click(stage(), { clientX: 12, clientY: 34 });
    await flush();
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].annotations).toEqual([
      {
        subject: POINT_SUBJECT,
        bbox: null,
        points: null,
        rings: null,
        point: [12, 34],
        attributes: {},
        iscrowd: false,
      },
    ]);
  });

  it("m cycles Box -> Polygon -> Point -> Box", async () => {
    await mountPointMode();
    fireEvent.keyDown(window, { key: "m" });
    expect(useStore.getState().gui.mode).toBe("box");
    fireEvent.keyDown(window, { key: "m" });
    expect(useStore.getState().gui.mode).toBe("polygon");
    fireEvent.keyDown(window, { key: "m" });
    expect(useStore.getState().gui.mode).toBe("point");
  });
});

describe("AnnotateTab casing", () => {
  it("draws every outline on the dark casing, the focus halo without one", async () => {
    await mountTab();
    act(addBox);
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), { clientX: 30, clientY: 30 });
    const rects = screen.getAllByTestId("k-rect");
    const cased = rects.filter((r) => r.getAttribute("data-shadow") === CASING.color);
    expect(cased.map((r) => r.getAttribute("data-stroke"))).toEqual([subjectColor(SUBJECT)]);
    expect(
      rects.find((r) => r.getAttribute("data-stroke") === FOCUS_HALO.color),
    ).not.toHaveAttribute("data-shadow");
  });
});

describe("AnnotateTab image states", () => {
  it("reads every image's state for the selection, then follows each load and save", async () => {
    const { states, complete } = ANNOTATE_ANSWERS;
    const statesSpy = vi.spyOn(api.annotate, "states").mockResolvedValue(states);
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve(imagePath === IMG1 ? loadedLabels(complete) : loadOf(imagePath)),
    );
    await mountTab();
    expect(statesSpy).toHaveBeenCalledWith(SUBJECT);
    // The read says unannotated; the document the load answered says complete.
    expect(states.states[IMG1_NAME]).toBe("unannotated");
    expect(useStore.getState().annotateUi.imageStates).toEqual({
      ...states.states,
      [IMG1_NAME]: complete.completion[SUBJECT],
    });

    saveSpy.mockResolvedValueOnce(LANDED.negative);
    fireEvent.click(screen.getByText("toolbar-complete"));
    await flush();
    expect(useStore.getState().annotateUi.imageStates[IMG1_NAME]).toBe(
      LANDED.negative.document.completion[SUBJECT],
    );
  });
});

describe("AnnotateTab focused item", () => {
  it("hands the focused box to the toolbar, whose subject menu edits its attributes", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    expect(screen.getByTestId("toolbar")).toHaveAttribute("data-focused", "");

    act(addBox);
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), { clientX: 30, clientY: 30 });
    expect(screen.getByTestId("toolbar")).toHaveAttribute("data-focused", "annotation box 0");
    expect(screen.queryByTestId("subject-attributes")).toBeNull();
  });
});

describe("AnnotateTab legend keyboard access", () => {
  it("opens the legend by keyboard and reaches a subject row on the following Tab", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    const legendButton = screen.getByRole("button", { name: "Legend" });
    expect(legendButton).toHaveAttribute("aria-expanded", "false");
    fireEvent.click(legendButton);
    expect(legendButton).toHaveAttribute("aria-expanded", "true");

    const subjectRow = screen.getByRole("button", { name: SUBJECT });
    expect(subjectRow).toBeInTheDocument();
    // DOM order: the panel follows the button, so a forward Tab from it reaches the row.
    expect(
      legendButton.compareDocumentPosition(subjectRow) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
  });

  it("opens the color picker as a labeled dialog with a named hex input", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    fireEvent.click(screen.getByRole("button", { name: "Legend" }));
    fireEvent.click(screen.getByRole("button", { name: SUBJECT }));

    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(dialog).toHaveAccessibleName(`${SUBJECT} color`);
    expect(screen.getByRole("textbox", { name: "hex color" })).toBeInTheDocument();
  });
});

describe("AnnotateTab entries on the canvas", () => {
  const unreadable = ANNOTATE_ANSWERS.unreadable;

  /** Draw the box mounted in the first image, and answer its save with `refusal`. */
  async function refusedSave(refusal: unknown) {
    render(
      <>
        <AnnotateTab />
        <NotificationsPane />
      </>,
    );
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox);
    saveSpy.mockRejectedValueOnce(refusal);
    pressSave();
    await flush();
  }

  /** The one entry the notifications area lists. */
  const inArea = () =>
    within(screen.getByRole("region", { name: "Notifications" })).getByRole("listitem");

  it("names why the labels failed to load", async () => {
    loadSpy.mockRejectedValueOnce(await thrown(unreadable));

    render(<AnnotateTab />);

    const entry = await screen.findByTestId("entry-in-place");
    expect(entry).toHaveTextContent(
      `Could not load this image's labels: ${unreadable.body.detail}. Saving is off.`,
    );
  });

  it("a version conflict offers Reload on the canvas, which reloads the image, and none in the area", async () => {
    await refusedSave(await versionConflict());

    const [entry] = onCanvas();
    expect(entry).toHaveTextContent(CONFLICT_SENTENCE);
    const area = inArea();
    expect(area).toHaveTextContent(CONFLICT_SENTENCE);
    expect(within(area).queryByRole("button", { name: "Reload" })).toBeNull();

    fireEvent.click(within(entry).getByRole("button", { name: "Reload" }));
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    expect(loadSpy.mock.calls[1][0]).toBe(IMG1);
    expect(onCanvas()).toHaveLength(0);
    expect(inArea()).toHaveTextContent(CONFLICT_SENTENCE);
  });

  it("another save failure stands on the canvas offering no Reload", async () => {
    await refusedSave(new Error("network down"));

    const [entry] = onCanvas();
    expect(entry).toHaveTextContent("Could not save annotations (network down)");
    expect(within(entry).queryByRole("button", { name: "Reload" })).toBeNull();
  });

  it("a failure standing for one image is gone from the canvas as the next is stepped to, before its load lands", async () => {
    await refusedSave(new Error("network down"));
    expect(onCanvas()).toHaveLength(1);

    loadSpy.mockImplementation(() => new Promise(() => {}));
    act(() => changeSelection({ current_image_index: 1 }));
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));

    expect(onCanvas()).toHaveLength(0);
    expect(inArea()).toHaveTextContent("Could not save annotations");
  });

  it("reads a failed image once, stepping back to it before the next image lands", async () => {
    loadSpy.mockRejectedValueOnce(await thrown(unreadable));
    render(<AnnotateTab />);
    await screen.findByText(/Saving is off/);

    loadSpy.mockImplementation(() => new Promise(() => {}));
    act(() => changeSelection({ current_image_index: 1 }));
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    act(() => changeSelection({ current_image_index: 0 }));
    await flush();

    expect(loadSpy).toHaveBeenCalledTimes(2);
    expect(useStore.getState().canvas.failedLoad).toBe(IMG1);
  });

  it("shows a refused new subject first in the strip, and dismissed puts focus on the subject control", async () => {
    await refusedSave(new Error("network down"));
    const stale = LABEL_ANSWERS.registry_stale;
    vi.spyOn(subjectsApi, "save").mockRejectedValue(await thrown(stale));
    vi.spyOn(subjectsApi, "load").mockResolvedValue(LABEL_ANSWERS.registry_loaded);
    const root = useStore.getState().gui.dataset!.dataset_root;
    await act(() =>
      saveRegistry(
        { husk: { description: "one husk" } },
        "Could not add the subject",
        refusedRegistryKey("new subject", root),
      ),
    );

    const [subjectEntry, saveEntry] = onCanvas();
    expect(subjectEntry).toHaveTextContent(`Could not add the subject: ${stale.body.detail}`);
    expect(saveEntry).toHaveTextContent("Could not save annotations (network down)");
    fireEvent.click(
      within(subjectEntry).getByRole("button", { name: /^Dismiss: Could not add the subject/ }),
    );
    expect(onCanvas()).toHaveLength(1);
    expect(screen.getByRole("button", { name: "toolbar-subject" })).toHaveFocus();
  });

  it("is dismissed by a control named from its text, taking Reload with it", async () => {
    await refusedSave(await versionConflict());

    const [entry] = onCanvas();
    fireEvent.click(
      within(entry).getByRole("button", {
        name: (name) =>
          name.startsWith(
            dismissLabel(
              `${CONFLICT_SENTENCE}. Reload to load the latest (discards your unsaved edits), or keep editing.`,
            ),
          ),
      }),
    );
    expect(onCanvas()).toHaveLength(0);
    expect(screen.queryByText("Reload")).not.toBeInTheDocument();
    expect(listed()).toEqual([]);
  });

  it("sits in the row below the Fit pill, never on top of it", async () => {
    await refusedSave(await versionConflict());

    const line = screen.getByTestId("entry-in-place").parentElement;
    const fit = screen.getByRole("button", { name: "Fit" });
    expect(line).toHaveClass("top-12");
    expect(fit).toHaveClass("top-3");
  });

  it("dismissed in place, puts focus on the canvas column", async () => {
    await refusedSave(new Error("network down"));

    const [entry] = onCanvas();
    const column = entry.parentElement!.parentElement!;
    fireEvent.click(
      within(entry).getByRole("button", { name: /^Dismiss: Could not save annotations/ }),
    );
    expect(onCanvas()).toHaveLength(0);
    expect(column).toHaveFocus();
  });

  it("shows a save canceled while its image is still loaded on the canvas, as a warning notice", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox);
    const save = Promise.withResolvers<LoadedSave>();
    saveSpy.mockReturnValueOnce(save.promise);
    pressSave();
    await flush();
    fireEvent.click(screen.getByRole("button", { name: "toolbar-cancel-save" }));
    await act(() => Promise.resolve(save.reject(new DOMException("aborted", "AbortError"))));

    const [entry] = onCanvas();
    expect(entry).toHaveTextContent("Save canceled.");
    expect(listed()).toEqual([
      expect.objectContaining({
        kind: "notice",
        level: "warning",
        context: { project: TEST_PROJECT.id, target: IMG1_NAME },
      }),
    ]);
  });

  it("puts a failed reload on the canvas and keeps Reload offered", async () => {
    await refusedSave(await versionConflict());
    loadSpy.mockRejectedValueOnce(new Error("network down"));
    fireEvent.click(within(onCanvas()[0]).getByRole("button", { name: "Reload" }));
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();

    const [failure, conflict] = onCanvas();
    expect(failure).toHaveTextContent("Reload failed: network down");
    expect(conflict).toHaveTextContent(CONFLICT_SENTENCE);
    expect(within(conflict).getByRole("button", { name: "Reload" })).toBeInTheDocument();
  });

  it("takes Reload over the edits a remount kept as a reload, keeping them when it fails", async () => {
    const first = render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox);
    saveSpy.mockRejectedValueOnce(await versionConflict());
    first.unmount();
    await flush();
    render(<AnnotateTab />);
    await flush();
    loadSpy.mockRejectedValueOnce(await thrown(unreadable));
    expect(onCanvas()).toHaveLength(1);

    fireEvent.click(within(onCanvas()[0]).getByRole("button", { name: "Reload" }));
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();

    expect(onCanvas()[0]).toHaveTextContent(`Reload failed: ${unreadable.body.detail}`);
    expect(useStore.getState().canvas.boxes).toHaveLength(1);
    expect(useStore.getState().canvas.dirty).toBe(true);
  });

  it("puts a failed proposals read on the canvas", async () => {
    vi.spyOn(api.annotate, "stagings").mockRejectedValue(new Error("network down"));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    const [entry] = onCanvas();
    expect(entry).toHaveTextContent("Could not load proposals: network down");
    expect(listed()).toEqual([
      expect.objectContaining({
        kind: "failure",
        level: "error",
        context: { project: TEST_PROJECT.id, target: IMG1_NAME },
      }),
    ]);
  });

  it("puts a refused find on the canvas", async () => {
    vi.spyOn(api.annotate, "find").mockRejectedValue(new Error("network down"));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    fireEvent.click(screen.getByRole("button", { name: "toolbar-phrase" }));
    await flush();

    const [entry] = onCanvas();
    expect(entry).toHaveTextContent("Could not find: network down");
    expect(listed()).toEqual([expect.objectContaining({ kind: "failure", level: "error" })]);
  });

  it("lists found boxes answered after the image changed as an information notice naming the image, holding none", async () => {
    const answer = Promise.withResolvers<Awaited<ReturnType<typeof api.annotate.find>>>();
    vi.spyOn(api.annotate, "find").mockReturnValue(answer.promise);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    fireEvent.click(screen.getByRole("button", { name: "toolbar-phrase" }));
    await flush();
    act(() => changeSelection({ current_image_index: 1 }));
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    await act(() => Promise.resolve(answer.resolve(ANNOTATE_ANSWERS.found)));

    expect(useStore.getState().annotateUi.pendingPrompts.found).toEqual([]);
    expect(onCanvas()).toHaveLength(0);
    expect(listed()).toEqual([
      expect.objectContaining({
        kind: "notice",
        level: "info",
        sentence: `${ANNOTATE_ANSWERS.found.boxes.length} found on ${IMG1_NAME}, not held.`,
        context: { project: TEST_PROJECT.id, target: IMG1_NAME },
      }),
    ]);
  });

  it("lists a failed read of the images' states naming the capture, with nothing on the canvas", async () => {
    vi.spyOn(api.annotate, "states").mockRejectedValue(new Error("network down"));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(onCanvas()).toHaveLength(0);
    expect(listed()).toEqual([
      expect.objectContaining({
        kind: "failure",
        level: "error",
        sentence: "Could not read the images' states: network down",
        key: undefined,
        context: { project: TEST_PROJECT.id, target: useStore.getState().gui.dataset!.date },
      }),
    ]);
  });
});

describe("AnnotateTab host size", () => {
  const { img_width: IMG_W, img_height: IMG_H } = loadOf(IMG1);
  const SMALLER = { w: Math.floor(IMG_W * 0.6), h: Math.floor(IMG_H * 0.5) };
  const SMALLEST = { w: Math.floor(IMG_W * 0.3), h: Math.floor(IMG_H * 0.25) };

  /** Mount over the first image at scale 1, in a host smaller than the image. */
  async function mountInHost(size: HostSize) {
    stage.size = size;
    useStore.getState().setView({ scale: 1, offset_x: 0, offset_y: 0 });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
  }

  it("plans the region reads for the size the stage reports, and again when it reports another", async () => {
    // The base bitmap is served at half size, so the view at scale 1 holds more than it carries.
    stage.facts = {
      ok: true,
      image: null,
      aborted: false,
      servedSize: { w: IMG_W / 2, h: IMG_H / 2 },
      imageError: null,
    };
    const reads = vi.spyOn(api.images, "viewReads").mockResolvedValue({ reads: [] });
    await mountInHost(SMALLER);
    await waitFor(() =>
      expect(reads).toHaveBeenLastCalledWith(IMG1, expect.any(Number), {
        x0: 0,
        y0: 0,
        x1: SMALLER.w,
        y1: SMALLER.h,
      }),
    );

    act(() => stage.reportSize(SMALLEST));
    await waitFor(() =>
      expect(reads).toHaveBeenLastCalledWith(IMG1, expect.any(Number), {
        x0: 0,
        y0: 0,
        x1: SMALLEST.w,
        y1: SMALLEST.h,
      }),
    );
  });

  it("fits the image to the size the stage reports", async () => {
    await mountInHost(SMALLER);
    act(() => useStore.getState().setView({ scale: 2, offset_x: 0, offset_y: 0 }));
    fireEvent.click(screen.getByRole("button", { name: "Fit" }));

    const scale = Math.min(SMALLER.w / IMG_W, SMALLER.h / IMG_H);
    expect(useStore.getState().gui.view).toEqual({
      scale,
      offset_x: (SMALLER.w - IMG_W * scale) / 2,
      offset_y: (SMALLER.h - IMG_H * scale) / 2,
    });
  });

  it("pushes the viewport the stage's size shows to the agent", async () => {
    const pushSpy = vi.spyOn(api.canvas, "pushState").mockResolvedValue(null);
    await mountInHost(SMALLER);
    act(() => notifyCanvasStateRequest());
    await flush();

    expect(pushSpy.mock.calls.at(-1)?.[0].viewport).toEqual({
      x: 0,
      y: 0,
      w: SMALLER.w,
      h: SMALLER.h,
      scale: 1,
    });
  });

  it("schedules a push of the new viewport when the stage reports another size", async () => {
    const pushSpy = vi.spyOn(api.canvas, "pushState").mockResolvedValue(null);
    await mountInHost(SMALLER);
    // Settle the pushes the mount scheduled, so any push after this is the resize's own.
    act(() => notifyCanvasStateRequest());
    await flush();
    pushSpy.mockClear();

    vi.useFakeTimers();
    try {
      act(() => stage.reportSize(SMALLEST));
      await act(() => vi.advanceTimersByTimeAsync(2000));
    } finally {
      vi.useRealTimers();
    }

    expect(pushSpy.mock.calls.at(-1)?.[0].viewport).toEqual({
      x: 0,
      y: 0,
      w: SMALLEST.w,
      h: SMALLEST.h,
      scale: 1,
    });
  });
});

describe("AnnotateTab labels written by the agent", () => {
  // The produced saves, each naming the first photograph of the produced capture, the image in
  // view: one declaring no client, and one declaring the agent's.
  const { labels_written: written, labels_written_declared: declared } = PANEL_EVENTS;
  // Another project than the one each save was posted for.
  const ridge = openedAs(PICKER.ridge).project;

  /** Render the tab with `ridge` open and wait for the image in view to load. */
  async function renderOpenOnRidge() {
    expect(ridge.id).not.toBe(written.project_id);
    useStore.setState({ openProject: ridge });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
  }

  it("names the declared harness that wrote the current image's labels, under the save's project", async () => {
    expect(declared.event.image_path).toBe(IMG1);
    const { agent_client_name: name, agent_client_version: version } = declared.identity;
    await renderOpenOnRidge();

    act(addBox);
    act(() => {
      useStore.getState().pushAgentActivity(declared);
    });

    const [entry] = onCanvas();
    expect(entry).toHaveTextContent(
      `${name} ${version} just updated this image's labels. Reload to load them (discards your unsaved edits), or keep editing.`,
    );
    expect(within(entry).getByRole("button", { name: "Reload" })).toBeInTheDocument();
    expect(listed()).toEqual([
      expect.objectContaining({
        kind: "notice",
        level: "warning",
        context: { project: declared.project_id, target: IMG1_NAME },
      }),
    ]);
  });

  it("names a process when the write declared no actor, under the save's project", async () => {
    await renderOpenOnRidge();

    act(addBox);
    act(() => {
      useStore.getState().pushAgentActivity(written);
    });

    expect(
      screen.getByText(
        "A process just updated this image's labels. Reload to load them (discards your unsaved edits), or keep editing.",
      ),
    ).toBeInTheDocument();
    expect(listed()).toEqual([
      expect.objectContaining({ context: { project: written.project_id, target: IMG1_NAME } }),
    ]);
  });

  it("loads the image in view again over a clean canvas", async () => {
    expect(written.event.image_path).toBe(IMG1);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(() => {
      useStore.getState().pushAgentActivity(written);
    });

    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    expect(loadSpy).toHaveBeenLastCalledWith(IMG1);
    expect(listed()).toEqual([]);
  });
});

describe("AnnotateTab legend", () => {
  it("lists the subjects until proposals are shown, then the match types", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    const proposalsSpy = withBucket();
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    await waitFor(() => expect(proposalsSpy).toHaveBeenCalled());
    await flush();

    // The legend button is hover-revealed, so query its content directly.
    const legend = within(document.getElementById("annotate-legend-panel")!);
    for (const match of MATCH_TYPES) {
      expect(legend.getByText(MATCH_WORDS[match])).toBeInTheDocument();
    }
    expect(legend.queryByRole("button", { name: SUBJECT })).not.toBeInTheDocument();

    toggleProposals();
    await flush();
    expect(legend.getByRole("button", { name: SUBJECT })).toBeInTheDocument();
    expect(legend.queryByText(MATCH_WORDS.matched)).not.toBeInTheDocument();
  });

  it("a recolored subject's box stroke and the pushed canvas_meta swatch both follow", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", subjectColor(SUBJECT));

    fireEvent.click(screen.getByRole("button", { name: SUBJECT }));
    const hexInput = screen.getByRole("textbox", { name: "hex color" });
    fireEvent.change(hexInput, { target: { value: "#123456" } });
    fireEvent.keyDown(hexInput, { key: "Enter" });
    fireEvent.click(screen.getByRole("button", { name: "OK" }));
    await flush();

    expect(subjectColor(SUBJECT)).toBe("#123456");
    expect(screen.getByTestId("k-rect")).toHaveAttribute("data-stroke", "#123456");

    const pushSpy = vi.spyOn(api.canvas, "pushState").mockResolvedValue(null);
    act(() => notifyCanvasStateRequest());
    await flush();
    const pushed = pushSpy.mock.calls.at(-1)?.[0];
    expect(pushed?.classes).toEqual(expect.arrayContaining([{ name: SUBJECT, color: "#123456" }]));
  });
});

describe("AnnotateTab canvas push refused", () => {
  it("resyncs to the backend's open project when it holds another one open", async () => {
    vi.spyOn(api.canvas, "pushState").mockRejectedValue(
      await thrown(PROJECT_FIXTURES.project_not_open),
    );
    const resync = vi.spyOn(stateSocket, "resync").mockImplementation(() => {});
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(() => notifyCanvasStateRequest());
    await flush();
    expect(resync).toHaveBeenCalled();
  });
});

describe("AnnotateTab band-composite wiring", () => {
  it("offers the toolbar no band choice for a plain photograph", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    expect(BAND_ANSWERS.plain.read.kind).toBe(PLAIN);
    expect(screen.getByTestId("toolbar")).toHaveAttribute("data-band-choice", "none");
    // No bands/stretch param for a frame the server displays plain: the canvas URL is unaffected.
    const url = screen.getByTestId("canvas-stage").getAttribute("data-image-url") ?? "";
    expect(url).not.toContain("bands=");
    expect(url).not.toContain("stretch=");
  });

  it("carries a >3-band dataset's picked bands/stretch into the canvas image URL", async () => {
    vi.spyOn(api.images, "bands").mockResolvedValue(BAND_ANSWERS.four_spectral);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    await waitFor(() =>
      expect(screen.getByTestId("toolbar")).toHaveAttribute(
        "data-band-count",
        String(BAND_ANSWERS.four_spectral.band_count),
      ),
    );
    const url = screen.getByTestId("canvas-stage").getAttribute("data-image-url") ?? "";
    const params = new URLSearchParams(url.split("?")[1]);
    // Defaulted to the server's default bands and its default stretch.
    expect(params.getAll("bands")).toEqual(BAND_ANSWERS.four_spectral.default_bands);
    expect(params.get("stretch")).toBe(BAND_ANSWERS.four_spectral.default_stretch);
  });

  it("requests the next image without the previous image's bands until its own answer arrives", async () => {
    const nextAnswer = Promise.withResolvers<ImageBands>();
    vi.spyOn(api.images, "bands").mockImplementation((path) =>
      path === IMG1 ? Promise.resolve(BAND_ANSWERS.four_spectral) : nextAnswer.promise,
    );
    render(<AnnotateTab />);
    await waitFor(() =>
      expect(screen.getByTestId("toolbar")).toHaveAttribute("data-band-choice", "banded"),
    );

    act(() => changeSelection({ current_image_index: 1 }));
    await flush();
    const url = screen.getByTestId("canvas-stage").getAttribute("data-image-url") ?? "";
    expect(url).toContain(encodeURIComponent(IMG2));
    expect(url).not.toContain("bands=");
    expect(screen.getByTestId("toolbar")).toHaveAttribute("data-band-choice", "none");

    await act(() => Promise.resolve(nextAnswer.resolve(BAND_ANSWERS.four_spectral)));
    expect(screen.getByTestId("toolbar")).toHaveAttribute("data-band-choice", "banded");
  });
});

// The CanvasStage mock forwards clientX/Y as image-pixel coords, so these drive the real
// pointer state machine against a 1000x800 image.
const POLY_A = {
  rings: [
    [
      [10, 10],
      [200, 10],
      [200, 200],
      [10, 200],
    ] as [number, number][],
  ],
  subject: SUBJECT,
  attributes: {},
  iscrowd: false,
};
const POLY_B = {
  rings: [
    [
      [300, 300],
      [400, 300],
      [400, 400],
      [300, 400],
    ] as [number, number][],
  ],
  subject: SUBJECT,
  attributes: {},
  iscrowd: false,
};

function seedPolygons(polygons: (typeof POLY_A)[]) {
  useStore.getState().loadLabelsIntoCanvas(
    loadHolding(
      IMG1,
      polygons.map(({ subject, rings }, index) => servedAnnotation({ subject, rings, index })),
    ),
  );
}

const nextFrame = () =>
  act(() => new Promise<void>((resolve) => requestAnimationFrame(() => resolve())));

async function renderPolygonCanvas(
  polygons: (typeof POLY_A)[] = [POLY_A, POLY_B],
  alongside: React.ReactNode = null,
) {
  render(
    <>
      <AnnotateTab />
      {alongside}
    </>,
  );
  await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
  await flush();
  act(() => {
    useStore.getState().setMode("polygon");
    seedPolygons(polygons);
  });
  return screen.getByTestId("canvas-stage");
}

describe("click-selection parity across the Snap/Stream toggles", () => {
  it("a click on a polygon selects it with Stream on, never starts a new one", async () => {
    const stage = await renderPolygonCanvas();
    act(() => useStore.getState().setStream(true));
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });

  it("a click on another polygon switches the selection with Stream on", async () => {
    const stage = await renderPolygonCanvas();
    act(() => {
      useStore.getState().setStream(true);
      useStore.getState().setFocus({ kind: "polygon", index: 0 });
    });
    fireEvent.click(stage, { clientX: 350, clientY: 350 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 1 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });

  it("with Stream on, empty space deselects first and a later click still streams", async () => {
    const stage = await renderPolygonCanvas();
    act(() => {
      useStore.getState().setStream(true);
      useStore.getState().setFocus({ kind: "polygon", index: 0 });
    });
    fireEvent.click(stage, { clientX: 600, clientY: 600 });
    expect(useStore.getState().canvas.focus).toBeNull();
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
    fireEvent.click(stage, { clientX: 600, clientY: 600 });
    expect(useStore.getState().canvas.currentPolygon).toEqual([[600, 600]]);
  });

  it("a click on a polygon selects it with Snap on", async () => {
    const stage = await renderPolygonCanvas();
    act(() => useStore.getState().setSnap(true));
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });
});

describe("AnnotateTab with the help open", () => {
  const openHelp = () => {
    fireEvent.keyDown(document.body, { key: "?" });
    expect(screen.getByRole("dialog", { name: "Shortcuts" })).toBeInTheDocument();
  };
  const help = <HelpOverlay activeTab="annotate" />;

  it("Delete leaves the focused annotation, and removes it once the help is closed", async () => {
    const stage = await renderPolygonCanvas([POLY_A, POLY_B], help);
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });

    openHelp();
    fireEvent.keyDown(document.body, { key: "Delete" });
    expect(useStore.getState().canvas.polygons).toHaveLength(2);

    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    fireEvent.keyDown(document.body, { key: "Delete" });
    expect(useStore.getState().canvas.polygons).toHaveLength(1);
  });

  it("Ctrl+S does not save", async () => {
    await renderPolygonCanvas([POLY_A, POLY_B], help);
    openHelp();
    fireEvent.keyDown(document.body, { key: "s", ctrlKey: true });
    await flush();
    expect(saveSpy).not.toHaveBeenCalled();
  });

  it("Escape closes only the help and leaves the focus", async () => {
    const stage = await renderPolygonCanvas([POLY_A, POLY_B], help);
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    openHelp();

    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });

    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(useStore.getState().canvas.focus).toBeNull();
  });

  it("Escape closes only the help and leaves a polygon in progress", async () => {
    const stage = await renderPolygonCanvas([], help);
    fireEvent.click(stage, { clientX: 600, clientY: 600 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(1);
    openHelp();

    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(1);

    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });
});

describe("AnnotateTab with the notifications pane open", () => {
  beforeEach(measureWindow);
  const openPane = dockWithNotificationsOpen;
  /** A polygon begun on empty space, then the first polygon selected with the review-path key. */
  const selectWithPolygonInProgress = (stage: HTMLElement) => {
    fireEvent.click(stage, { clientX: 600, clientY: 600 });
    fireEvent.keyDown(document.body, { key: "ArrowDown" });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(1);
  };

  const where = Object.keys(PANE_FOCUS) as (keyof typeof PANE_FOCUS)[];
  it.each(where.flatMap((target) => ["Delete", "Escape"].map((key) => [key, target] as const)))(
    "%s with focus on the pane's %s keeps the selection and the polygon in progress",
    async (key, target) => {
      const stage = await renderPolygonCanvas([POLY_A, POLY_B], openPane());
      selectWithPolygonInProgress(stage);
      const focused = await PANE_FOCUS[target]();
      focused.focus();
      expect(focused).toHaveFocus();

      fireEvent.keyDown(focused, { key });
      expect(useStore.getState().canvas.polygons).toHaveLength(2);
      expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });
      expect(useStore.getState().canvas.currentPolygon).toHaveLength(1);
    },
  );

  it("the same keys on the canvas, pane open, still delete the selection and cancel the polygon", async () => {
    const stage = await renderPolygonCanvas([POLY_A, POLY_B], openPane());
    await PANE_FOCUS.heading();
    selectWithPolygonInProgress(stage);

    fireEvent.keyDown(document.body, { key: "Delete" });
    expect(useStore.getState().canvas.polygons).toHaveLength(1);
    fireEvent.keyDown(document.body, { key: "Escape" });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });

  it("the help overlay stays closed for a ? typed on the pane's heading", async () => {
    await renderPolygonCanvas(
      [POLY_A, POLY_B],
      <>
        {openPane()}
        <HelpOverlay activeTab="annotate" />
      </>,
    );
    const heading = await PANE_FOCUS.heading();
    heading.focus();

    fireEvent.keyDown(heading, { key: "?" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    fireEvent.keyDown(document.body, { key: "?" });
    expect(screen.getByRole("dialog", { name: "Shortcuts" })).toBeInTheDocument();
  });
});

describe("a polygon gesture is one undo step", () => {
  const undo = () => act(() => useStore.getState().undo());
  const polygons = () => useStore.getState().canvas.polygons;
  /** The loaded polygon, and it with a value set on it. */
  const seededAndHealthy = () => {
    const [seeded] = polygons();
    return [seeded, { ...seeded, attributes: { health: "good" } }] as const;
  };

  it("one undo restores a right-click-deleted vertex, the next reaches the edit before it", async () => {
    const stage = await renderPolygonCanvas([POLY_A]);
    const [seeded, healthy] = seededAndHealthy();
    act(() => useStore.getState().update("polygons", 0, healthy));
    act(() => useStore.getState().setFocus({ kind: "polygon", index: 0 }));
    fireEvent.contextMenu(stage, { clientX: 10, clientY: 10 });
    expect(polygons()[0].rings[0]).toHaveLength(3);

    undo();
    expect(polygons()[0]).toEqual(healthy);
    undo();
    expect(polygons()[0]).toEqual(seeded);
  });

  it("one undo removes a streamed polygon, the next reaches the edit before it", async () => {
    const stage = await renderPolygonCanvas([POLY_A]);
    const [seeded, healthy] = seededAndHealthy();
    act(() => {
      useStore.getState().update("polygons", 0, healthy);
      useStore.getState().setStream(true);
    });
    fireEvent.click(stage, { clientX: 600, clientY: 600 });
    act(() => {
      useStore.getState().setCurrentPolygon([
        [600, 600],
        [700, 600],
        [700, 700],
      ]);
      useStore.getState().commitCurrentPolygon();
    });
    expect(polygons()).toHaveLength(2);

    undo();
    expect(polygons()).toEqual([healthy]);
    undo();
    expect(polygons()).toEqual([seeded]);
  });
});

describe("Cut tool arming", () => {
  it("x arms and disarms the cut flag in polygon mode", async () => {
    await renderPolygonCanvas();
    expect(useStore.getState().annotateUi.cut).toBe(false);
    fireEvent.keyDown(window, { key: "x" });
    expect(useStore.getState().annotateUi.cut).toBe(true);
    fireEvent.keyDown(window, { key: "x" });
    expect(useStore.getState().annotateUi.cut).toBe(false);
  });

  it("x does nothing outside polygon mode", async () => {
    await renderPolygonCanvas();
    act(() => useStore.getState().setMode("box"));
    fireEvent.keyDown(window, { key: "x" });
    expect(useStore.getState().annotateUi.cut).toBe(false);
  });

  it("arming and disarming each schedule a push carrying the current cut_armed value", async () => {
    await renderPolygonCanvas();
    const pushSpy = vi.spyOn(api.canvas, "pushState").mockResolvedValue(null);
    pushSpy.mockClear();

    vi.useFakeTimers();
    try {
      act(() => useStore.getState().setCut(true));
      act(() => {
        vi.advanceTimersByTime(1600);
      });
      expect(pushSpy.mock.calls.at(-1)?.[0].cut_armed).toBe(true);

      pushSpy.mockClear();
      act(() => useStore.getState().setCut(false));
      act(() => {
        vi.advanceTimersByTime(1600);
      });
      expect(pushSpy.mock.calls.at(-1)?.[0].cut_armed).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("Cut gesture", () => {
  function armOnPolyA(): void {
    act(() => {
      useStore.getState().setFocus({ kind: "polygon", index: 0 });
      useStore.getState().setCut(true);
    });
  }

  it("with a polygon selected, the two clicks produce two polygons and the flag stays set", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(2); // only the start is pending so far
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(3);
    expect(useStore.getState().annotateUi.cut).toBe(true);
  });

  it("a first click near the selected outline places the start and inserts no vertex", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    // 2px outside POLY_A's top edge: within the edge-insert threshold, so without the cut guard
    // onDown would splice a new vertex into the ring here instead of arming the start.
    fireEvent.mouseDown(stage, { clientX: 100, clientY: 8, button: 0 });
    fireEvent.click(stage, { clientX: 100, clientY: 8, button: 0 });
    expect(useStore.getState().canvas.polygons[0].rings[0]).toHaveLength(4);
    fireEvent.click(stage, { clientX: 100, clientY: 250, button: 0 }); // the start was real
    expect(useStore.getState().canvas.polygons).toHaveLength(3);
  });

  /** The refusal the canvas shows in place for the image in view. */
  const refusalOnCanvas = () => screen.getByTestId("entry-in-place");

  it("with none selected, the first click is refused on the canvas", async () => {
    const stage = await renderPolygonCanvas();
    act(() => useStore.getState().setCut(true));
    fireEvent.click(stage, { clientX: 500, clientY: 500, button: 0 });
    expect(refusalOnCanvas()).toHaveTextContent(
      "Select a polygon to cut, then click two points on either side of it.",
    );
  });

  it("Escape clears the start and the flag", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    fireEvent.keyDown(window, { key: "Escape" });
    expect(useStore.getState().annotateUi.cut).toBe(false);
    // and the start really cleared: re-arming and clicking once is a fresh first click, not a cut
    act(() => useStore.getState().setCut(true));
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });

  it("right-click clears the start alone and deletes nothing", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    fireEvent.contextMenu(stage, { clientX: 105, clientY: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
    expect(useStore.getState().annotateUi.cut).toBe(true);
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 }); // a fresh first click, not a cut
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });

  it("an image change clears the start and the flag", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => {
      changeSelection({ current_image_index: 1 });
    });
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    expect(useStore.getState().annotateUi.cut).toBe(false);
  });

  const POLYGON_CHANGED_SENTENCE =
    "The polygon changed since the first click; the cut was canceled. Select it and place both " +
    "points again.";

  it("a selection change between the clicks refuses with the polygon-changed sentence", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => useStore.getState().setFocus({ kind: "polygon", index: 1 }));
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
    expect(refusalOnCanvas()).toHaveTextContent(POLYGON_CHANGED_SENTENCE);
  });

  it("an attribute edit between the clicks does not cancel the cut", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => {
      const poly = useStore.getState().canvas.polygons[0];
      useStore.getState().update("polygons", 0, { ...poly, attributes: { health: "good" } });
    });
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(3); // the cut still landed
  });

  it("a vertex edit between the clicks refuses with the polygon-changed sentence", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => useStore.getState().dragVertex(0, 0, 0, [11, 11]));
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(2); // unchanged: the cut was canceled
    expect(refusalOnCanvas()).toHaveTextContent(POLYGON_CHANGED_SENTENCE);
  });

  it("a refused cut leaves the polygons unchanged and the flag set", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    fireEvent.click(stage, { clientX: 105, clientY: 5, button: 0 }); // never reaches the outline
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
    expect(useStore.getState().canvas.polygons[0].rings[0]).toHaveLength(4);
    expect(useStore.getState().annotateUi.cut).toBe(true);
    expect(refusalOnCanvas()).toHaveTextContent(CUT_MISSES_REFUSAL);
  });

  it("after cutting one polygon, a click inside another selects it with no start placed", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(3); // POLY_A split, POLY_B untouched
    expect(useStore.getState().annotateUi.cut).toBe(true);

    // POLY_B now sits at index 2 (POLY_A's two pieces occupy 0 and 1).
    fireEvent.click(stage, { clientX: 350, clientY: 350, button: 0 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 2 });
    expect(useStore.getState().canvas.polygons).toHaveLength(3); // the click authored nothing

    fireEvent.click(stage, { clientX: 305, clientY: 250, button: 0 });
    fireEvent.click(stage, { clientX: 305, clientY: 450, button: 0 });
    expect(useStore.getState().canvas.polygons).toHaveLength(4); // POLY_B cut too
  });

  it("disarming via setCut clears a pending start (the toolbar button's own action)", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => useStore.getState().setCut(false));
    act(() => useStore.getState().setCut(true));
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 }); // a fresh first click, not a cut
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });

  it("disarming via x clears a pending start", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    fireEvent.keyDown(window, { key: "x" });
    act(() => useStore.getState().setCut(true));
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 }); // a fresh first click, not a cut
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });

  it("a mode change away from polygon clears the flag and the pending start", async () => {
    const stage = await renderPolygonCanvas();
    armOnPolyA();
    fireEvent.click(stage, { clientX: 105, clientY: 0, button: 0 });
    act(() => useStore.getState().setMode("box"));
    expect(useStore.getState().annotateUi.cut).toBe(false);
    act(() => {
      useStore.getState().setMode("polygon");
      useStore.getState().setCut(true);
    });
    fireEvent.click(stage, { clientX: 105, clientY: 250, button: 0 }); // a fresh first click, not a cut
    expect(useStore.getState().canvas.polygons).toHaveLength(2);
  });
});

describe("clicks outside the image extent are inert", () => {
  it("box mode: a press-drag from outside authors no box", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    const stage = screen.getByTestId("canvas-stage");
    fireEvent.mouseDown(stage, { clientX: 1200, clientY: -50 });
    fireEvent.mouseMove(stage, { clientX: -100, clientY: 900 });
    await nextFrame();
    fireEvent.mouseUp(stage, { clientX: -100, clientY: 900 });
    expect(useStore.getState().canvas.boxes).toHaveLength(0);
  });

  it("point mode: an outside click places no point", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(() => useStore.getState().setMode("point"));
    fireEvent.click(screen.getByTestId("canvas-stage"), { clientX: 1200, clientY: 400 });
    expect(useStore.getState().canvas.points).toHaveLength(0);
  });

  it("polygon mode: an outside click starts no polygon", async () => {
    const stage = await renderPolygonCanvas([]);
    fireEvent.click(stage, { clientX: 1200, clientY: 400 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });

  it("polygon mode: an outside click adds no vertex to a polygon in progress", async () => {
    const stage = await renderPolygonCanvas([]);
    fireEvent.click(stage, { clientX: 600, clientY: 600 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(1);
    fireEvent.click(stage, { clientX: 1200, clientY: 400 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(1);
  });

  it("Escape drops the focus", async () => {
    const stage = await renderPolygonCanvas();
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });

    fireEvent.keyDown(window, { key: "Escape" });
    expect(useStore.getState().canvas.focus).toBeNull();
  });

  it("an outside click does not drop an existing selection", async () => {
    const stage = await renderPolygonCanvas();
    fireEvent.click(stage, { clientX: 50, clientY: 50 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });
    fireEvent.click(stage, { clientX: 1050, clientY: 400 });
    expect(useStore.getState().canvas.focus).toEqual({ kind: "polygon", index: 0 });
  });

  it("with Stream on, an outside click starts no stream and later moves lay nothing", async () => {
    const stage = await renderPolygonCanvas([]);
    act(() => useStore.getState().setStream(true));
    fireEvent.click(stage, { clientX: 1200, clientY: 400 });
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
    fireEvent.mouseMove(stage, { clientX: 500, clientY: 400 });
    await nextFrame();
    expect(useStore.getState().canvas.currentPolygon).toHaveLength(0);
  });
});

describe("AnnotateTab authoring writes what the annotator meant", () => {
  it("bounds a new point by the image's own height on a taller-than-wide frame", async () => {
    // Only a portrait frame separates the bounds: a y past the width but inside the height.
    useStore.getState().setRegistry(registryOf(POINT_SUBJECT), UNSTORED);
    const portrait = RAW_SELECTION.images.findIndex(({ path }) => {
      const { img_width: width, img_height: height } = loadOf(path);
      return height > width;
    });
    openTestProject(selected({ subject: SUBJECT, current_image_index: portrait }), {
      mode: "point",
      active_subject: POINT_SUBJECT,
    });
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    const { img_width: width, img_height: height } = loadOf(RAW_SELECTION.images[portrait].path);
    // A point below the frame's width and within its height, so only a portrait frame holds it.
    const y = Math.round((width + height) / 2);
    expect(y).toBeGreaterThan(width);
    expect(y).toBeLessThan(height);

    fireEvent.click(screen.getByTestId("canvas-stage"), { clientX: 420, clientY: y });
    await flush();

    expect(useStore.getState().canvas.points).toEqual([
      { x: 420, y, subject: POINT_SUBJECT, attributes: {}, iscrowd: false },
    ]);
  });

  it("addresses the save by the image alone, naming no place its labels are kept", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(addBox);
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].image_path).toBe(IMG1);
    expect(Object.keys(saveSpy.mock.calls[0][0])).not.toContain("label_path");
  });

  it("commits a drawn box under the subject the drag started on, not the one active at release", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT, SECOND_SUBJECT), UNSTORED);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(() => useStore.getState().setActiveSubject(SUBJECT));
    const stage = screen.getByTestId("canvas-stage");

    fireEvent.mouseDown(stage, { clientX: 120, clientY: 60, button: 0 });
    act(() => useStore.getState().setActiveSubject(SECOND_SUBJECT));
    fireEvent.mouseMove(stage, { clientX: 470, clientY: 330 });
    await nextFrame();
    fireEvent.mouseUp(stage, { clientX: 470, clientY: 330 });
    await flush();

    expect(useStore.getState().canvas.boxes).toEqual([
      { x1: 120, y1: 60, x2: 470, y2: 330, subject: SUBJECT, attributes: {}, iscrowd: false },
    ]);
  });

  it("keeps a freshly drawn box exactly at the minimum side", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    const stage = screen.getByTestId("canvas-stage");

    fireEvent.mouseDown(stage, { clientX: 100, clientY: 100, button: 0 });
    fireEvent.mouseMove(stage, { clientX: 103, clientY: 103 });
    await nextFrame();
    fireEvent.mouseUp(stage, { clientX: 103, clientY: 103 });
    await flush();

    expect(useStore.getState().canvas.boxes).toEqual([
      { x1: 100, y1: 100, x2: 103, y2: 103, subject: SUBJECT, attributes: {}, iscrowd: false },
    ]);
    expect(listed()).toHaveLength(0);
  });

  describe("a freshly drawn box smaller than the minimum", () => {
    const TOO_SMALL = "Box too small to keep. Drag out a bigger area.";

    /** Mount over the first image and drag a box from (100, 100) by `side` on each axis. */
    async function mountAndDraw(side: number) {
      useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
      render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
      await flush();
      await drawBox(side);
    }

    async function drawBox(side: number) {
      const stage = screen.getByTestId("canvas-stage");
      fireEvent.mouseDown(stage, { clientX: 100, clientY: 100, button: 0 });
      fireEvent.mouseMove(stage, { clientX: 100 + side, clientY: 100 + side });
      await nextFrame();
      fireEvent.mouseUp(stage, { clientX: 100 + side, clientY: 100 + side });
      await flush();
    }

    it("is refused by one failure shown on the canvas and listed", async () => {
      await mountAndDraw(2);

      expect(useStore.getState().canvas.boxes).toHaveLength(0);
      const [entry] = onCanvas();
      expect(entry).toHaveTextContent(TOO_SMALL);
      expect(listed()).toEqual([
        expect.objectContaining({
          kind: "failure",
          level: "error",
          sentence: TOO_SMALL,
          context: { project: TEST_PROJECT.id, target: IMG1_NAME },
        }),
      ]);
    });

    it("goes from the canvas and from the list when dismissed", async () => {
      await mountAndDraw(2);
      const [entry] = onCanvas();
      fireEvent.click(within(entry).getByRole("button", { name: /^Dismiss: Box too small/ }));

      expect(onCanvas()).toHaveLength(0);
      expect(listed()).toEqual([]);
    });

    it("goes from the canvas when the next box is kept, and stays listed", async () => {
      await mountAndDraw(2);
      await drawBox(60);

      expect(useStore.getState().canvas.boxes).toHaveLength(1);
      expect(onCanvas()).toHaveLength(0);
      expect(listed()).toEqual([expect.objectContaining({ sentence: TOO_SMALL, key: undefined })]);
    });

    it("goes from the canvas when the image changes, and stays listed", async () => {
      await mountAndDraw(2);
      // The next image's proposals read never lands, so nothing but its own key can hide this.
      const stagings = vi
        .spyOn(api.annotate, "stagings")
        .mockImplementation((path) =>
          path === IMG2 ? new Promise(() => {}) : Promise.resolve(ANNOTATE_ANSWERS.no_stagings),
        );
      act(() => changeSelection({ current_image_index: 1 }));
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
      await flush();

      expect(stagings).toHaveBeenCalledWith(IMG2);
      expect(onCanvas()).toHaveLength(0);
      expect(listed()).toEqual([expect.objectContaining({ sentence: TOO_SMALL })]);
    });
  });

  it("undoes a resize that shrinks a box below the minimum, refused on the canvas", async () => {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(addBox); // {x1:10, y1:10, x2:50, y2:50}
    const stage = screen.getByTestId("canvas-stage");

    fireEvent.mouseDown(stage, { clientX: 30, clientY: 30, button: 0 }); // press inside selects
    fireEvent.mouseUp(stage, { clientX: 30, clientY: 30 });
    await flush();

    fireEvent.mouseDown(stage, { clientX: 50, clientY: 50, button: 0 }); // bottom-right corner
    fireEvent.mouseMove(stage, { clientX: 11, clientY: 11 });
    await nextFrame();
    fireEvent.mouseUp(stage, { clientX: 11, clientY: 11 });
    await flush();

    expect(useStore.getState().canvas.boxes).toEqual([
      { x1: 10, y1: 10, x2: 50, y2: 50, subject: SUBJECT, attributes: {}, iscrowd: false },
    ]);
    expect(screen.getByTestId("entry-in-place")).toHaveTextContent(
      "Box too small to keep; the resize was undone.",
    );
  });

  describe("a save pressed while a box resize is still being dragged", () => {
    async function dragCornerTo(x: number, y: number) {
      useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
      render(<AnnotateTab />);
      await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
      await flush();
      act(addBox); // {x1:10, y1:10, x2:50, y2:50}
      const stage = screen.getByTestId("canvas-stage");
      fireEvent.mouseDown(stage, { clientX: 30, clientY: 30, button: 0 });
      fireEvent.mouseUp(stage, { clientX: 30, clientY: 30 });
      await flush();
      fireEvent.mouseDown(stage, { clientX: 50, clientY: 50, button: 0 });
      fireEvent.mouseMove(stage, { clientX: x, clientY: y });
      await nextFrame();
    }

    it("undoes one below the minimum side and saves the box as it was", async () => {
      await dragCornerTo(11, 11);
      pressSave();
      await flush();
      expect(saveSpy.mock.calls[0][0].annotations[0]).toMatchObject({ bbox: [10, 10, 50, 50] });
      expect(listed()).toEqual([
        expect.objectContaining({ sentence: "Box too small to keep; the resize was undone." }),
      ]);
    });

    it("reverts the resize, not a vertex of a polygon being drawn, and leaves the draft whole", async () => {
      await dragCornerTo(11, 11);
      const draft: [number, number][] = [
        [200, 200],
        [260, 200],
        [260, 260],
      ];
      act(() => useStore.getState().setCurrentPolygon(draft));
      let draftWhenSent: unknown;
      saveSpy.mockImplementationOnce(() => {
        draftWhenSent = useStore.getState().canvas.currentPolygon;
        return Promise.resolve(LANDED.box);
      });
      pressSave();
      await flush();
      expect(saveSpy.mock.calls[0][0].annotations[0]).toMatchObject({ bbox: [10, 10, 50, 50] });
      expect(draftWhenSent).toEqual(draft);
    });

    it("is settled before the image changes, and the next image inherits no focus from it", async () => {
      await dragCornerTo(11, 11);
      expect(useStore.getState().canvas.focus).toEqual({ kind: "box", index: 0 });
      const other = servedAnnotation({ subject: "other", bbox: [300, 300, 340, 340], index: 0 });
      loadSpy.mockImplementation((imagePath) => Promise.resolve(loadHolding(imagePath, [other])));
      act(() => {
        changeSelection({ current_image_index: 1 });
      });
      await flush();
      expect(saveSpy.mock.calls[0][0].annotations[0]).toMatchObject({ bbox: [10, 10, 50, 50] });
      await waitFor(() => expect(useStore.getState().canvas.boxes[0]?.subject).toBe("other"));
      expect(useStore.getState().canvas.focus).toBeNull();
    });

    it("saves one at or above it as dragged", async () => {
      await dragCornerTo(30, 30);
      pressSave();
      await flush();
      expect(saveSpy.mock.calls[0][0].annotations[0]).toMatchObject({ bbox: [10, 10, 30, 30] });
    });
  });

  it("carries a geometry-less image rating into the save payload", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(() => {
      const s = useStore.getState();
      s.add("imageAnnotations", { subject: SUBJECT, attributes: {} });
      s.update("imageAnnotations", 0, {
        subject: SUBJECT,
        attributes: { canopy_cover: "sparse" },
        iscrowd: false,
      });
    });
    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].annotations).toEqual([
      {
        subject: SUBJECT,
        bbox: null,
        points: null,
        rings: null,
        point: null,
        attributes: { canopy_cover: "sparse" },
        iscrowd: false,
      },
    ]);
  });
});

describe("AnnotateTab labels show on the focused item only", () => {
  // The legend is the standing symbology reference; a committed shape is named on the canvas
  // only while it is the focused item, never on hover, for every shape kind.
  const stage = () => screen.getByTestId("canvas-stage");
  const frame = () => act(async () => void (await new Promise((r) => setTimeout(r, 25))));
  const labelsNamed = (name: string) =>
    screen.queryAllByTestId("k-text").filter((t) => t.getAttribute("data-text") === name);

  function setupSubject(mode: "box" | "polygon" | "point") {
    useStore.getState().setRegistry(registryOf(POINT_SUBJECT), UNSTORED);
    useStore.setState((s) => ({ gui: { ...s.gui, mode, active_subject: POINT_SUBJECT } }));
  }

  it("a box is unlabeled at rest and under the cursor, labeled while focused", async () => {
    setupSubject("box");
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    act(() =>
      useStore
        .getState()
        .add("boxes", { x1: 10, y1: 10, x2: 50, y2: 50, subject: POINT_SUBJECT, attributes: {} }),
    );

    expect(labelsNamed(POINT_SUBJECT)).toHaveLength(0);

    fireEvent.mouseMove(stage(), { clientX: 30, clientY: 30 });
    await frame();
    expect(labelsNamed(POINT_SUBJECT)).toHaveLength(0);

    fireEvent.mouseDown(stage(), { clientX: 30, clientY: 30, button: 0 }); // press inside focuses
    await flush();
    expect(labelsNamed(POINT_SUBJECT).length).toBeGreaterThan(0);
  });

  it("a polygon is unlabeled at rest and under the cursor, labeled while focused", async () => {
    setupSubject("polygon");
    const rings: [number, number][][] = [
      [
        [100, 100],
        [300, 100],
        [300, 300],
        [100, 300],
      ],
    ];
    const shape = servedAnnotation({ subject: POINT_SUBJECT, rings, index: 0 });
    loadSpy.mockImplementation((imagePath) => Promise.resolve(loadHolding(imagePath, [shape])));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(labelsNamed(POINT_SUBJECT)).toHaveLength(0);

    fireEvent.mouseMove(stage(), { clientX: 200, clientY: 200 });
    await frame();
    expect(labelsNamed(POINT_SUBJECT)).toHaveLength(0);

    fireEvent.click(stage(), { clientX: 200, clientY: 200 }); // click inside focuses
    await flush();
    expect(labelsNamed(POINT_SUBJECT).length).toBeGreaterThan(0);
  });

  it("a point is unlabeled at rest and under the cursor, labeled while focused", async () => {
    setupSubject("point");
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve(
        loadHolding(imagePath, [
          servedAnnotation({ subject: POINT_SUBJECT, point: [100, 100], index: 0 }),
        ]),
      ),
    );
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    expect(labelsNamed(POINT_SUBJECT)).toHaveLength(0);

    fireEvent.mouseMove(stage(), { clientX: 102, clientY: 101 });
    await frame();
    expect(labelsNamed(POINT_SUBJECT)).toHaveLength(0);

    fireEvent.mouseDown(stage(), { clientX: 100, clientY: 100, button: 0 }); // press focuses
    fireEvent.mouseUp(stage(), { clientX: 100, clientY: 100 });
    await flush();
    expect(labelsNamed(POINT_SUBJECT).length).toBeGreaterThan(0);
  });
});

describe("AnnotateTab canvas push names its project", () => {
  it("pushes nothing while the backend has no project open", async () => {
    const pushSpy = vi.spyOn(api.canvas, "pushState").mockResolvedValue(null);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    useStore.setState({ openProject: null });

    act(() => notifyCanvasStateRequest());
    await flush();

    expect(pushSpy).not.toHaveBeenCalled();
  });

  it("pushes carrying the open project's id", async () => {
    const pushSpy = vi.spyOn(api.canvas, "pushState").mockResolvedValue(null);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();

    act(() => notifyCanvasStateRequest());
    await flush();

    expect(pushSpy).toHaveBeenCalled();
    expect(pushSpy.mock.calls[0][0].project_id).toBe(TEST_PROJECT.id);
  });
});

/** The selected bucket of one undecided proposal, over a document holding nothing. */
function withBucket() {
  changeSelection({ bucket: ANNOTATE_ANSWERS.buckets.single });
  return vi.spyOn(api.annotate, "proposals").mockResolvedValue(ANNOTATE_ANSWERS.single_proposals);
}

async function mountTab() {
  render(<AnnotateTab />);
  await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
  await flush();
}

const dottedRects = () =>
  screen.queryAllByTestId("k-rect").filter((r) => r.getAttribute("data-dash") === "true");

describe("AnnotateTab completion marks", () => {
  it("marks the subject through the save door and shows the state the answered document derives", async () => {
    await mountTab();
    saveSpy.mockResolvedValueOnce(LANDED.negative);

    fireEvent.click(screen.getByText("toolbar-complete"));
    await flush();

    expect(saveSpy.mock.calls[0][0]).toMatchObject({
      gestures: { complete: { [SUBJECT]: true }, rect: null, proposals_hidden: false },
    });
    await waitFor(() =>
      expect(screen.getByTestId("toolbar")).toHaveAttribute(
        "data-subject-state",
        LANDED.negative.document.completion[SUBJECT],
      ),
    );
  });

  it("records on the mark that proposals were hidden while it was made", async () => {
    withBucket();
    await mountTab();

    toggleProposals();
    await flush();
    fireEvent.click(screen.getByText("toolbar-complete"));
    await flush();

    expect(saveSpy.mock.calls[0][0].gestures.proposals_hidden).toBe(true);
  });
});

describe("AnnotateTab proposals", () => {
  it("shows the bucket's undecided proposals and requests none while they are hidden", async () => {
    const proposalsSpy = withBucket();
    await mountTab();
    await waitFor(() => expect(dottedRects()).toHaveLength(1));
    expect(proposalsSpy).toHaveBeenCalledWith(IMG1, ANNOTATE_ANSWERS.buckets.single);

    toggleProposals();
    await flush();
    expect(dottedRects()).toHaveLength(0);
    act(() => {
      changeSelection({});
    });
    await flush();
    expect(proposalsSpy).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["a", "accepted"],
    ["r", "rejected"],
  ] as const)(
    "%s adjudicates the selected proposal through the save door and adopts its answer",
    async (key, verdict) => {
      withBucket();
      await mountTab();
      await waitFor(() => expect(dottedRects()).toHaveLength(1));

      fireEvent.keyDown(window, { key });
      await flush();

      expect(saveSpy.mock.calls[0][0]).toMatchObject({
        gestures: {
          decide: { bucket: ANNOTATE_ANSWERS.buckets.single, verdicts: { 0: verdict } },
        },
      });
      expect(loadSpy).toHaveBeenCalledTimes(1);
    },
  );
});

describe("AnnotateTab review symbology", () => {
  // The review bucket's proposals: one accepted and one undecided each pairing with a box, and two
  // undecided pairing with nothing, one scored below the rest; it validates no operating point.
  const BUCKET = ANNOTATE_ANSWERS.buckets.review;
  const served = () => ANNOTATE_ANSWERS.review_proposals;
  const boxes = () => LOADED.review.boxes;
  // What the review's items count, read off the produced document and proposals: its shapes,
  // each with the proposal paired with it, and every undecided proposal no shape pairs with.
  const SHAPES = ANNOTATE_ANSWERS.review.annotations.length;
  const UNDECIDED = served().proposals.filter((p) => p.decision === null);
  const LOOSE = UNDECIDED.filter((p) => p.paired === null);
  // The unpaired proposal scored below the rest, which the review path reaches fourth, as the
  // strip names it.
  const [unpaired] = [...LOOSE].sort((a, b) => (a.score ?? 0) - (b.score ?? 0));
  const UNPAIRED_PROPOSAL = proposalLabel(unpaired, true)!;
  const PAIRED = new Set(
    served()
      .proposals.map((p) => p.paired)
      .filter((i) => i !== null),
  );
  const PAIRED_SHAPES = PAIRED.size;
  const [UNDECIDED_PAIRED] = UNDECIDED.filter((p) => p.paired !== null);
  const atLeast = (floor: number) => (p: { score: number | null }) => (p.score ?? 0) >= floor;
  const scoped = (all: number, unreviewed: number) => [
    `all (${all})`,
    `unreviewed (${unreviewed})`,
    "flagged (0)",
  ];
  const strokes = () => screen.getAllByTestId("k-rect").map((r) => r.getAttribute("data-stroke"));
  const position = () => screen.getByRole("textbox", { name: "Item position" });
  const scopes = () =>
    Array.from(
      screen.getByRole("combobox", { name: "Step through" }).querySelectorAll("option"),
      (o) => o.textContent,
    );
  const down = async () => {
    fireEvent.keyDown(window, { key: "ArrowDown" });
    await flush();
  };

  /** Mount over the review document, flagged on its first box when `flagged`, its bucket's
   *  proposals served. */
  async function mountReviewing(flagged = false) {
    const loaded = flagged ? LOADED.review_flagged : LOADED.review;
    loadSpy.mockImplementation(() => Promise.resolve(loaded));
    changeSelection({ bucket: BUCKET });
    const proposalsSpy = vi.spyOn(api.annotate, "proposals").mockResolvedValue(served());
    await mountTab();
    await waitFor(() => expect(proposalsSpy).toHaveBeenCalled());
    await flush();
    return proposalsSpy;
  }

  it("outline color says match type while proposals are shown and subject otherwise", async () => {
    await mountReviewing();
    // The undecided proposals draw first, the unpaired then the paired, then each annotation's
    // box in its order, matched where a proposal pairs with it.
    expect(strokes()).toEqual([
      ...LOOSE.map(() => MATCH_COLORS.proposal_only),
      ...UNDECIDED.filter((p) => p.paired !== null).map(() => MATCH_COLORS.matched),
      ...ANNOTATE_ANSWERS.review.annotations.map((a) =>
        PAIRED.has(a.index) ? MATCH_COLORS.matched : MATCH_COLORS.annotation_only,
      ),
    ]);

    toggleProposals();
    await flush();
    expect(strokes()).toEqual(Array(SHAPES).fill(subjectColor(SUBJECT)));
  });

  it("the confidence floor starts at the operating point and the person moves it", async () => {
    await mountReviewing();
    const floor = screen.getByRole("spinbutton", { name: "Confidence floor" });
    expect(served().operating_point.kind).toBe(CONF_REFUSED);
    expect(floor).toHaveValue(null);
    // The shapes and the unpaired proposals; a shape with its undecided paired proposal, and the
    // unpaired proposals, await a decision.
    expect(scopes()).toEqual(scoped(SHAPES + LOOSE.length, UNDECIDED.length));

    fireEvent.change(floor, { target: { value: "0.5" } });
    await flush();
    const held = UNDECIDED.filter(atLeast(0.5));
    expect(scopes()).toEqual(scoped(SHAPES + LOOSE.filter(atLeast(0.5)).length, held.length));
    expect(dottedRects()).toHaveLength(held.length);
  });

  it("a floor the person moved belongs to its bucket; another bucket starts at its own", async () => {
    await mountReviewing();
    const floor = () => screen.getByRole("spinbutton", { name: "Confidence floor" });
    fireEvent.change(floor(), { target: { value: "0.1" } });
    await flush();
    expect(floor()).toHaveValue(0.1);

    act(() => changeSelection({ bucket: ANNOTATE_ANSWERS.buckets.shaped }));
    await flush();
    expect(floor()).toHaveValue(null);
  });

  it("the match filter keeps one match type on the canvas and in the path", async () => {
    await mountReviewing();
    const filter = screen.getByRole("combobox", { name: "Show" });

    fireEvent.change(filter, { target: { value: "annotation_only" } });
    await flush();
    expect(scopes()).toEqual(scoped(SHAPES - PAIRED_SHAPES, 0));
    expect(dottedRects()).toHaveLength(0);

    fireEvent.change(filter, { target: { value: "proposal_only" } });
    await flush();
    expect(scopes()).toEqual(scoped(LOOSE.length, LOOSE.length));
    expect(dottedRects()).toHaveLength(LOOSE.length);
  });

  it("a match filter applies only while proposals are shown", async () => {
    await mountReviewing();
    fireEvent.change(screen.getByRole("combobox", { name: "Show" }), {
      target: { value: "annotation_only" },
    });
    await flush();
    expect(scopes()).toEqual(scoped(SHAPES - PAIRED_SHAPES, 0));

    toggleProposals();
    await flush();
    expect(scopes()).toEqual(scoped(SHAPES, 0));
  });

  it("the items span every shape, and the tool follows the focused item", async () => {
    await mountReviewing();
    await down();
    expect(position()).toHaveValue("1");
    expect(strokes()).toContain(FOCUS_HALO.color);

    act(() => useStore.getState().setMode("polygon"));
    await flush();
    // The path keeps every item whatever the tool; a tool change drops the focus as before.
    expect(scopes()).toEqual(scoped(SHAPES + LOOSE.length, UNDECIDED.length));
    expect(position()).toHaveValue("");

    await down();
    expect(position()).toHaveValue("1");
    expect(useStore.getState().gui.mode).toBe("box");

    const jump = position();
    fireEvent.focus(jump);
    fireEvent.change(jump, { target: { value: String(SHAPES + LOOSE.length) } });
    fireEvent.keyDown(jump, { key: "Enter" });
    await flush();
    expect(screen.getByText(`${SUBJECT} polygon`)).toBeInTheDocument();
    expect(useStore.getState().gui.mode).toBe("polygon");
  });

  it("focus is a halo under the item's own stroke, never a recolor", async () => {
    await mountReviewing();
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), {
      clientX: 30,
      clientY: 30,
      button: 0,
    });
    await flush();
    const rects = screen.getAllByTestId("k-rect");
    const halo = rects.find((r) => r.getAttribute("data-stroke") === FOCUS_HALO.color)!;
    expect(halo).toBeDefined();
    expect(Number(halo.getAttribute("data-opacity"))).toBe(FOCUS_HALO.opacity);
    // The focused box keeps its match color on its own stroke and its corner handles.
    expect(strokes()).toContain(MATCH_COLORS.matched);
    expect(strokes()).not.toContain("#00BFFF");
  });

  it("the down and up arrows step the nearest-neighbor path, set the subject and zoom", async () => {
    await mountReviewing();
    useStore.getState().setActiveSubject(null);
    const before = useStore.getState().gui.view;

    // The path starts nearest the top-left: box 0. Its subject becomes active and the view lands
    // on it (zoomed in past the fitted view).
    await down();
    expect(position()).toHaveValue("1");
    expect(screen.getByText(`${SUBJECT} box`)).toBeInTheDocument();
    expect(useStore.getState().gui.active_subject).toBe(SUBJECT);
    expect(useStore.getState().gui.view.scale).toBeGreaterThan(before.scale);

    await down();
    await down();
    expect(position()).toHaveValue("3");
    await down();
    expect(position()).toHaveValue("4");
    expect(screen.getByText(UNPAIRED_PROPOSAL)).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "ArrowUp" });
    await flush();
    expect(position()).toHaveValue("3");
    // The left and right arrows still step images, never items.
    expect(useStore.getState().gui.dataset?.current_image_index).toBe(0);

    fireEvent.change(screen.getByRole("combobox", { name: "Step through" }), {
      target: { value: "unreviewed" },
    });
    await flush();
    expect(screen.getByText(`/ ${UNDECIDED.length}`)).toBeInTheDocument();
  });

  it("a decides the focused matched annotation's own proposal", async () => {
    await mountReviewing();
    await down();
    await down();
    expect(position()).toHaveValue("2");

    fireEvent.keyDown(window, { key: "a" });
    await flush();
    expect(saveSpy.mock.calls[0][0]).toMatchObject({
      gestures: {
        decide: { bucket: BUCKET, verdicts: { [UNDECIDED_PAIRED.index]: "accepted" } },
      },
    });
  });

  it("a on a focused item decides it and steps to the next item in the order", async () => {
    await mountReviewing();
    saveSpy.mockResolvedValueOnce(LANDED.accept_paired);
    await down();
    await down();
    expect(position()).toHaveValue("2");

    fireEvent.keyDown(window, { key: "a" });
    await flush();
    await flush();
    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(position()).toHaveValue("3");
  });

  it("Accept all accepts every undecided proposal the canvas shows in one save", async () => {
    await mountReviewing();
    // No floor is in force, so the canvas draws every undecided proposal.
    expect(screen.getByRole("spinbutton", { name: "Confidence floor" })).toHaveValue(null);
    expect(dottedRects()).toHaveLength(UNDECIDED.length);

    fireEvent.click(screen.getByRole("button", { name: /^Accept all/ }));
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(saveSpy.mock.calls[0][0].gestures.decide).toEqual({
      bucket: BUCKET,
      verdicts: Object.fromEntries(UNDECIDED.map((p) => [p.index, "accepted"])),
    });
  });

  it("a with nothing focused decides the first unreviewed item and focuses nothing", async () => {
    await mountReviewing();
    expect(useStore.getState().canvas.focus).toBeNull();

    fireEvent.keyDown(window, { key: "a" });
    await flush();
    expect(saveSpy).toHaveBeenCalledTimes(1);
    expect(useStore.getState().canvas.focus).toBeNull();
  });

  it("e accepts the focused unpaired proposal so it can be corrected as an annotation", async () => {
    await mountReviewing();
    for (let i = 0; i < 4; i++) await down();
    expect(screen.getByText(UNPAIRED_PROPOSAL)).toBeInTheDocument();

    fireEvent.keyDown(window, { key: "e" });
    await flush();
    expect(saveSpy.mock.calls[0][0]).toMatchObject({
      gestures: { decide: { bucket: BUCKET, verdicts: { [unpaired.index]: "accepted" } } },
    });
  });

  /** The focus of the shape a landed save says `proposal` resolved to, among the `kind` shapes of
   *  the document it wrote. */
  const resolvedFocus = (landed: LoadedSave, proposal: number, kind: "box" | "polygon") => ({
    kind,
    index: landed.document[kind === "box" ? "boxes" : "polygons"].findIndex(
      (shape) => shape.index === landed.accepted[proposal],
    ),
  });

  /** The one proposal `landed` says it accepted, by its index in its bucket's document. */
  const acceptedProposal = (landed: LoadedSave) => {
    const accepted = Object.keys(landed.accepted).map(Number);
    if (accepted.length !== 1) throw new Error("the landed save accepted no one proposal");
    return accepted[0];
  };

  /** The tool document's box a tool wrote, and its position among the canvas's boxes. */
  const TOOL_REF = LOADED.tool_two.boxes.findIndex((b) => b.authorship === "tool");
  const TOOL_BOX = LOADED.tool_two.boxes[TOOL_REF];
  /** Press the middle of the box a tool wrote. */
  const pressToolBox = async () => {
    const { x1, y1, x2, y2 } = TOOL_BOX;
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), {
      clientX: (x1 + x2) / 2,
      clientY: (y1 + y2) / 2,
      button: 0,
    });
    await flush();
  };
  /** Mount over the tool's document with its own photograph in view, so its saves are adopted. */
  const mountToolDocument = async () => {
    openTestProject(selected({ subject: SUBJECT }, TOOL_SELECTION), {
      mode: "box",
      active_subject: SUBJECT,
    });
    loadSpy.mockImplementation(() => Promise.resolve(LOADED.tool_two));
    await mountTab();
  };

  it("a confirms a focused annotation no proposal pairs with when a tool left it unconfirmed", async () => {
    // The tool's document holds the review document's boxes, so the review's pairings stand.
    loadSpy.mockImplementation(() => Promise.resolve(LOADED.tool_two));
    changeSelection({ bucket: BUCKET });
    vi.spyOn(api.annotate, "proposals").mockResolvedValue(served());
    await mountTab();
    await flush();
    const { annotations, flags } = ANNOTATE_ANSWERS.tool_two;
    const unconfirmed = annotations.filter((a) => a.authorship === "tool");
    expect(scopes()).toEqual([
      `all (${annotations.length + LOOSE.length})`,
      `unreviewed (${UNDECIDED.length + unconfirmed.length})`,
      `flagged (${flags.length})`,
    ]);

    await pressToolBox();
    fireEvent.keyDown(window, { key: "a" });
    await flush();

    expect(saveSpy.mock.calls[0][0]).toMatchObject({
      gestures: { confirm: [TOOL_BOX.index] },
    });
    expect(saveSpy.mock.calls[0][0].gestures.decide).toBeNull();
  });

  it("a confirms an unconfirmed annotation with no bucket selected", async () => {
    loadSpy.mockImplementation(() => Promise.resolve(LOADED.tool_two));
    await mountTab();
    await flush();
    await pressToolBox();
    expect(screen.getByRole("button", { name: "Accept" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Reject" })).toBeDisabled();

    fireEvent.keyDown(window, { key: "r" });
    await flush();
    expect(saveSpy).not.toHaveBeenCalled();

    fireEvent.keyDown(window, { key: "a" });
    await flush();
    expect(saveSpy.mock.calls[0][0]).toMatchObject({
      gestures: { confirm: [TOOL_BOX.index] },
    });
  });

  it("a reject on a focused annotation with no proposal never reaches another item's proposal", async () => {
    await mountReviewing();
    const unpairedBox = boxes().findIndex((b) => b.index === undefined || !PAIRED.has(b.index));
    expect(unpairedBox).not.toBe(-1);
    act(() => useStore.getState().setFocus({ kind: "box", index: unpairedBox }));
    await flush();
    fireEvent.keyDown(window, { key: "r" });
    await flush();
    expect(saveSpy).not.toHaveBeenCalled();
  });

  it("the pointer handlers act on the item the canvas draws handles on, a paired proposal's annotation", async () => {
    await mountReviewing();
    act(() =>
      useStore
        .getState()
        .setFocus({ kind: "proposal", index: UNDECIDED_PAIRED.index, bucket: BUCKET }),
    );
    await flush();
    expect(useStore.getState().canvas.undoStack).toHaveLength(0);

    const handle = boxes().find((b) => b.index === UNDECIDED_PAIRED.paired)!;
    fireEvent.mouseDown(screen.getByTestId("canvas-stage"), {
      clientX: handle.x1,
      clientY: handle.y1,
      button: 0,
    });
    // A press on a handle of the focused annotation picks it up, one undo snapshot per drag.
    expect(useStore.getState().canvas.undoStack).toHaveLength(1);
  });

  it("a gesture's answer is adopted whole: its document and the token that names it together", async () => {
    saveSpy.mockResolvedValueOnce(LANDED.confirmed);
    await mountToolDocument();
    await pressToolBox();
    fireEvent.keyDown(window, { key: "a" });
    await flush();

    expect(useStore.getState().canvas.boxes[TOOL_REF].authorship).toBe("tool_accepted");
    expect(loadSpy).toHaveBeenCalledTimes(1);
    addBox();
    pressSave();
    await flush();
    expect(saveSpy.mock.calls[1][0].base_mtime).toBe(LANDED.confirmed.document.base_mtime);
  });

  it("e on a proposal whose save the canvas did not adopt focuses no annotation", async () => {
    const [loose] = LOOSE;
    await mountReviewing();
    act(() =>
      useStore.getState().setFocus({ kind: "proposal", index: loose.index, bucket: BUCKET }),
    );
    saveSpy.mockRejectedValueOnce(await versionConflict());
    fireEvent.keyDown(window, { key: "e" });
    await flush();
    expect(useStore.getState().canvas.focus).toEqual({
      kind: "proposal",
      index: loose.index,
      bucket: BUCKET,
    });
  });

  it("e whose answer arrives after another image was selected focuses nothing there", async () => {
    const [loose] = LOOSE;
    await mountReviewing();
    act(() =>
      useStore.getState().setFocus({ kind: "proposal", index: loose.index, bucket: BUCKET }),
    );
    const answer = Promise.withResolvers<LoadedSave>();
    saveSpy.mockReturnValueOnce(answer.promise);
    fireEvent.keyDown(window, { key: "e" });
    await flush();

    loadSpy.mockImplementation(() => new Promise(() => {}));
    act(() => {
      changeSelection({ current_image_index: 1 });
    });
    await flush();
    await act(() => Promise.resolve(answer.resolve(LANDED.accept_unpaired)));

    expect(useStore.getState().canvas.focus).toBeNull();
  });

  it("e focuses the annotation the answer says the proposal resolved to, a new one", async () => {
    const proposal = acceptedProposal(LANDED.accept_unpaired);
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "proposal", index: proposal, bucket: BUCKET }));
    saveSpy.mockResolvedValueOnce(LANDED.accept_unpaired);
    fireEvent.keyDown(window, { key: "e" });
    await flush();
    expect(useStore.getState().canvas.focus).toEqual(
      resolvedFocus(LANDED.accept_unpaired, proposal, "box"),
    );
  });

  it("e focuses the annotation the answer says the proposal resolved to, one already drawn", async () => {
    const proposal = acceptedProposal(LANDED.accept_drawn);
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "proposal", index: proposal, bucket: BUCKET }));
    saveSpy.mockResolvedValueOnce(LANDED.accept_drawn);
    fireEvent.keyDown(window, { key: "e" });
    await flush();
    // The save drew a box over the proposal first, so the proposal resolved to that box.
    expect(useStore.getState().canvas.focus).toEqual(
      resolvedFocus(LANDED.accept_drawn, proposal, "box"),
    );
  });

  it("e's focus is cleared by a navigation that follows its adoption, and the next image keeps none", async () => {
    const proposal = acceptedProposal(LANDED.accept_unpaired);
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "proposal", index: proposal, bucket: BUCKET }));
    const answer = Promise.withResolvers<LoadedSave>();
    saveSpy.mockReturnValueOnce(answer.promise);
    fireEvent.keyDown(window, { key: "e" });
    await flush();
    let armed = true;
    const stop = useStore.subscribe((now, prev) => {
      if (!armed || now.canvas.boxes === prev.canvas.boxes) return;
      armed = false;
      queueMicrotask(() => {
        changeSelection({ current_image_index: 1 });
      });
    });
    loadSpy.mockImplementation((imagePath) =>
      Promise.resolve(
        loadHolding(imagePath, [servedAnnotation({ subject: "other", bbox: [10, 10, 50, 50] })]),
      ),
    );
    await act(() => Promise.resolve(answer.resolve(LANDED.accept_unpaired)));
    stop();
    await waitFor(() => expect(useStore.getState().canvas.boxes[0]?.subject).toBe("other"));
    expect(useStore.getState().canvas.focus).toBeNull();
  });

  it("e sets the tool from the tool in use when the answer lands, not when e was pressed", async () => {
    const proposal = acceptedProposal(LANDED.accept_unpaired);
    await mountReviewing();
    act(() => useStore.getState().setFocus({ kind: "proposal", index: proposal, bucket: BUCKET }));
    const answer = Promise.withResolvers<LoadedSave>();
    saveSpy.mockReturnValueOnce(answer.promise);
    fireEvent.keyDown(window, { key: "e" });
    await flush();
    act(() => useStore.getState().setMode("polygon"));
    await act(() => Promise.resolve(answer.resolve(LANDED.accept_unpaired)));
    expect(useStore.getState().gui.mode).toBe("box");
    expect(useStore.getState().canvas.focus).toEqual(
      resolvedFocus(LANDED.accept_unpaired, proposal, "box"),
    );
  });

  it("e focuses a polygon the proposal resolved to, switching to its tool", async () => {
    const shaped = ANNOTATE_ANSWERS.buckets.shaped;
    const accepted = acceptedProposal(LANDED.accept_polygon);
    const polygon = ANNOTATE_ANSWERS.shaped_proposals.proposals.find((p) => p.index === accepted);
    if (polygon === undefined) throw new Error("the shaped bucket served no accepted proposal");
    // The second bucket's proposals were served over the document the unpaired acceptance wrote.
    loadSpy.mockImplementation(() => Promise.resolve(LANDED.accept_unpaired.document));
    changeSelection({ bucket: shaped });
    vi.spyOn(api.annotate, "proposals").mockResolvedValue(ANNOTATE_ANSWERS.shaped_proposals);
    await mountTab();
    await flush();
    act(() =>
      useStore.getState().setFocus({ kind: "proposal", index: polygon.index, bucket: shaped }),
    );
    saveSpy.mockResolvedValueOnce(LANDED.accept_polygon);
    fireEvent.keyDown(window, { key: "e" });
    await flush();
    expect(useStore.getState().gui.mode).toBe("polygon");
    expect(useStore.getState().canvas.focus).toEqual(
      resolvedFocus(LANDED.accept_polygon, polygon.index, "polygon"),
    );
  });

  describe("an adopted save, ordinary or gesture", () => {
    const editedToolBox = async () => {
      await mountToolDocument();
      act(() => useStore.getState().update("boxes", TOOL_REF, { ...TOOL_BOX, x2: 541 }));
      expect(useStore.getState().canvas.undoStack).toHaveLength(1);
    };

    it("gives the canvas the answer's authorship and drops the undo history", async () => {
      await editedToolBox();
      saveSpy.mockResolvedValueOnce(LANDED.person_boxes);
      pressSave();
      await flush();
      const { canvas } = useStore.getState();
      expect(canvas.boxes[TOOL_REF].authorship).toBe("person");
      expect([canvas.dirty, canvas.undoStack.length]).toEqual([false, 0]);
    });

    it("does the same for a gesture's answer", async () => {
      await editedToolBox();
      await pressToolBox();
      saveSpy.mockResolvedValueOnce(LANDED.confirmed);
      fireEvent.keyDown(window, { key: "a" });
      await flush();
      const { canvas } = useStore.getState();
      expect(canvas.boxes[TOOL_REF].authorship).toBe("tool_accepted");
      expect([canvas.dirty, canvas.undoStack.length]).toEqual([false, 0]);
    });

    it("serves the bucket's proposals again, dropping the pairings the old document gave", async () => {
      const proposalsSpy = await mountReviewing();
      act(() => useStore.getState().update("boxes", 0, { ...boxes()[0], x2: 51 }));
      saveSpy.mockResolvedValueOnce(LANDED.review_saved);
      pressSave();
      await flush();
      expect(proposalsSpy).toHaveBeenCalledTimes(2);
    });

    it("refuses an edit made while it is in flight, then adopts the answer whole", async () => {
      await editedToolBox();
      const answer = Promise.withResolvers<LoadedSave>();
      saveSpy.mockReturnValueOnce(answer.promise);
      pressSave();
      await flush();
      expect(useStore.getState().canvas.saving).not.toBeNull();

      act(() => useStore.getState().add("boxes", { ...boxes()[0], index: undefined }));
      await pressToolBox();
      const rectsBefore = screen.getAllByTestId("k-rect").length;
      fireEvent.mouseDown(screen.getByTestId("canvas-stage"), {
        clientX: 100,
        clientY: 100,
        button: 0,
      });
      await flush();
      expect(screen.getAllByTestId("k-rect")).toHaveLength(rectsBefore);
      expect(useStore.getState().canvas.boxes).toHaveLength(LOADED.tool_two.boxes.length);
      expect(useStore.getState().canvas.focus).toBeNull();

      await act(() => Promise.resolve(answer.resolve(LANDED.person_boxes)));
      const { canvas } = useStore.getState();
      expect(canvas.saving).toBeNull();
      expect(canvas.boxes[TOOL_REF].authorship).toBe("person");
      expect(canvas.dirty).toBe(false);
    });

    it("cancels a box being drawn when it starts, and counts no box it did not take", async () => {
      await editedToolBox();
      const baseline = () => useStore.getState().sessionTracking.annotationsAddedDelta;
      const added = baseline();
      const rectsBefore = screen.getAllByTestId("k-rect").length;
      const stage = screen.getByTestId("canvas-stage");
      fireEvent.mouseDown(stage, { clientX: 100, clientY: 100, button: 0 });
      fireEvent.mouseMove(stage, { clientX: 200, clientY: 200 });
      await act(async () => void (await new Promise((r) => setTimeout(r, 25))));
      expect(screen.getAllByTestId("k-rect").length).toBe(rectsBefore + 1);

      saveSpy.mockImplementationOnce(() => new Promise<LoadedSave>(() => {}));
      pressSave();
      await flush();
      expect(screen.getAllByTestId("k-rect")).toHaveLength(rectsBefore);
      fireEvent.mouseUp(stage, { clientX: 200, clientY: 200 });
      await flush();
      expect(useStore.getState().canvas.boxes).toHaveLength(LOADED.tool_two.boxes.length);
      expect(baseline()).toBe(added);
    });

    it("proposals are dropped when the answer is adopted, before they are served again", async () => {
      const proposalsSpy = await mountReviewing();
      expect(strokes()).toContain(MATCH_COLORS.proposal_only);
      act(() => useStore.getState().update("boxes", 0, { ...boxes()[0], x2: 51 }));
      proposalsSpy.mockImplementation(() => new Promise(() => {}));
      saveSpy.mockResolvedValueOnce(LANDED.review_saved);
      pressSave();
      await flush();
      expect(strokes()).not.toContain(MATCH_COLORS.proposal_only);
    });
  });

  describe("the agent's mirror and the canvas", () => {
    /** One item as either side states it: its kind, stroke, whether it draws dashed, whether the
     *  focus halo is under it, and its label. */
    interface Mark {
      kind: string;
      stroke: string | null;
      dashed: boolean;
      halo: boolean;
      label: string | null;
    }

    /** The items the mocked canvas draws, in draw order: each Konva node folded into the item it
     *  belongs to (a halo into the item it sits under, a point's four ticks and core into one
     *  point, the polygon in progress's line and vertex dots into one polyline, a label's two
     *  copies into one label), the corner handles and vertex dots dropped. A node that fits no
     *  item is refused. */
    const drawnMarks = (): Mark[] => {
      const marks: Mark[] = [];
      let halo = false;
      const nodes = Array.from(
        screen.getByTestId("canvas-stage").querySelectorAll<HTMLElement>('[data-testid^="k-"]'),
      );
      for (let i = 0; i < nodes.length; i++) {
        const { testid, stroke, fill, dash, closed, text } = nodes[i].dataset;
        const item = (kind: string, color: string | undefined, dashed: boolean) => {
          marks.push({ kind, stroke: color ?? null, dashed, halo, label: null });
          halo = false;
        };
        if (testid === "k-rect") {
          if (fill) continue;
          if (stroke === FOCUS_HALO.color) halo = true;
          else item("box", stroke, !!dash);
        } else if (testid === "k-line" && closed) {
          if (stroke === FOCUS_HALO.color) halo = true;
          else item("polygon", stroke, !!dash);
        } else if (testid === "k-line") {
          const ticks = nodes.slice(i, i + 4);
          const isPoint =
            ticks.length === 4 &&
            ticks.every((t) => t.dataset.testid === "k-line" && !t.dataset.closed) &&
            nodes[i + 4]?.dataset.testid === "k-circle";
          if (isPoint) {
            item("point", stroke, !!dash);
            i += 4;
          } else {
            // The polygon in progress: one open line, then a dot on each vertex laid.
            item("polyline", stroke, !!dash);
            while (nodes[i + 1]?.dataset.testid === "k-circle") i++;
          }
        } else if (testid === "k-circle") {
          if (!fill) halo = true;
        } else if (testid === "k-text") {
          const labeled = nodes[++i].dataset;
          if (labeled.text === FLAG_MARK.glyph && labeled.fill === FLAG_MARK.color) {
            item("point", FLAG_MARK.color, false);
            marks[marks.length - 1].label = FLAG_MARK.glyph;
          } else if (marks.length > 0) {
            marks[marks.length - 1].label = labeled.text ?? null;
          } else {
            throw new Error(`the canvas draws the label ${text} over nothing`);
          }
        } else {
          throw new Error(`the canvas draws a ${testid} this reading does not place`);
        }
      }
      return marks;
    };

    /** The items the last full push states, in its order; a flag's label is its glyph on the
     *  canvas and the glyph with its comment in the mirror, so a flag is read by its glyph, and
     *  the mirror names a polyline ("drawing", "cut") where the canvas draws no label. */
    const pushedMarks = (pushed: ReturnType<typeof canvasPushes>): Mark[] =>
      pushed.map((s) => ({
        kind: s.kind,
        stroke: s.color,
        dashed: !!s.dash,
        halo: !!s.halo,
        label:
          s.kind === "polyline"
            ? null
            : s.tag === "flag" && s.label?.startsWith(`${FLAG_MARK.glyph} `)
              ? FLAG_MARK.glyph
              : (s.label ?? null),
      }));

    let pushSpy: MockInstance<typeof api.canvas.pushState>;
    const canvasPushes = () => pushSpy.mock.calls.at(-1)?.[0].shapes ?? [];

    /** Mount over the flagged review document with its bucket as the Predictions and a staged
     *  bucket of polygon proposals as the Proposals, its first box focused. */
    async function mountWithBoth() {
      pushSpy = vi.spyOn(api.canvas, "pushState").mockResolvedValue(null);
      loadSpy.mockImplementation(() => Promise.resolve(LOADED.review_flagged));
      changeSelection({ bucket: BUCKET });
      vi.spyOn(api.annotate, "stagings").mockResolvedValue(ANNOTATE_ANSWERS.stagings);
      const proposalsSpy = vi
        .spyOn(api.annotate, "proposals")
        .mockImplementation((_, bucket) =>
          Promise.resolve(bucket === BUCKET ? served() : ANNOTATE_ANSWERS.staged_proposals),
        );
      await mountTab();
      await waitFor(() => expect(proposalsSpy).toHaveBeenCalledTimes(2));
      await flush();
      act(() => useStore.getState().setFocus({ kind: "box", index: 0 }));
      await flush();
    }

    const hold = () =>
      act(() => {
        const { setPrompt, addPromptPoint, addPromptBox, setFoundBoxes } = useStore.getState();
        setPrompt(true);
        addPromptPoint([120, 130]);
        addPromptBox([200, 200, 260, 280]);
        setFoundBoxes([[400, 400, 440, 450]]);
      });
    const labelsOff = () => act(() => useStore.getState().setVisible(false));
    const view = (name: "toolbar-view-proposals" | "toolbar-view-predictions") =>
      fireEvent.click(screen.getByText(name));

    const dotted = (marks: Mark[], kind: string) =>
      marks.filter((m) => m.dashed && m.kind === kind);
    const HELD = { stroke: subjectColor(SUBJECT), dashed: true, halo: false, label: null };
    const held = [
      { kind: "point", ...HELD },
      { kind: "box", ...HELD },
      { kind: "box", ...HELD },
    ];

    // Each case names the state it sets and what the canvas then holds, so a state that drew
    // nothing could not pass for agreement.
    const CASES: [string, () => void | Promise<void>, (drawn: Mark[]) => void][] = [
      [
        "every View toggle on",
        () => {},
        (drawn) => {
          expect(dotted(drawn, "polygon").length).toBeGreaterThan(0);
          expect(dotted(drawn, "box").length).toBeGreaterThan(0);
          expect(drawn.some((m) => !m.dashed && m.kind === "box")).toBe(true);
          expect(drawn.some((m) => m.stroke === FLAG_MARK.color)).toBe(true);
        },
      ],
      [
        "Labels off",
        labelsOff,
        (drawn) => {
          expect(dotted(drawn, "polygon").length).toBeGreaterThan(0);
          expect(dotted(drawn, "box").length).toBeGreaterThan(0);
          expect(drawn.some((m) => !m.dashed && m.stroke !== FLAG_MARK.color)).toBe(false);
          expect(drawn.some((m) => m.stroke === FLAG_MARK.color)).toBe(true);
        },
      ],
      [
        "Proposals off",
        () => view("toolbar-view-proposals"),
        (drawn) => {
          expect(dotted(drawn, "polygon")).toHaveLength(0);
          expect(dotted(drawn, "box").length).toBeGreaterThan(0);
        },
      ],
      [
        "Predictions off",
        () => view("toolbar-view-predictions"),
        (drawn) => {
          expect(dotted(drawn, "box")).toHaveLength(0);
          expect(dotted(drawn, "polygon").length).toBeGreaterThan(0);
        },
      ],
      [
        "a held prompt point and boxes",
        hold,
        (drawn) => expect(drawn.slice(-held.length)).toEqual(held),
      ],
      [
        "Labels off over a held prompt",
        () => {
          labelsOff();
          hold();
        },
        (drawn) => {
          expect(drawn.slice(-held.length)).toEqual(held);
          expect(drawn.some((m) => !m.dashed && m.stroke !== FLAG_MARK.color)).toBe(false);
        },
      ],
      [
        "the polygon tool",
        () => act(() => useStore.getState().setMode("polygon")),
        (drawn) => expect(drawn.some((m) => !m.dashed && m.kind === "polygon")).toBe(true),
      ],
      [
        "a laid polygon vertex beside a held prompt",
        () => {
          act(() => useStore.getState().setMode("polygon"));
          act(() => useStore.getState().setCurrentPolygon([[50, 50]]));
          hold();
        },
        (drawn) =>
          expect(drawn.slice(-held.length - 1)).toEqual([
            { kind: "polyline", ...HELD, label: null },
            ...held,
          ]),
      ],
      [
        "a box mid-drag beside a held prompt",
        async () => {
          hold();
          const stage = screen.getByTestId("canvas-stage");
          fireEvent.mouseDown(stage, { clientX: 100, clientY: 100, button: 0 });
          fireEvent.mouseMove(stage, { clientX: 200, clientY: 200 });
          await act(async () => void (await new Promise((r) => setTimeout(r, 25))));
        },
        (drawn) =>
          expect(drawn.slice(-held.length - 1)).toEqual([...held, { kind: "box", ...HELD }]),
      ],
    ];

    it.each(CASES)("state the same items in the same order under %s", async (_, apply, holds) => {
      await mountWithBoth();
      await apply();
      await flush();
      act(() => notifyCanvasStateRequest());
      await flush();

      const drawn = drawnMarks();
      holds(drawn);
      expect(pushedMarks(canvasPushes())).toEqual(drawn);
    });

    it("the mirror states the flag's place and comment beside the canvas's glyph", async () => {
      await mountWithBoth();
      act(() => notifyCanvasStateRequest());
      await flush();
      const [flag] = canvasPushes().filter((s) => s.tag === "flag");
      const [{ text, target }] = ANNOTATE_ANSWERS.review_flagged.flags.map((f) => f.request);
      const point = target && "point" in target ? target.point : undefined;
      expect(flag).toMatchObject({ points: [point], label: `${FLAG_MARK.glyph} ${text}` });
    });

    it("a prompt placed on a settled canvas is pushed without the agent asking", async () => {
      await mountWithBoth();
      act(() => notifyCanvasStateRequest());
      await flush();
      expect(canvasPushes().some((s) => s.tag === "held_prompt")).toBe(false);

      hold();
      await waitFor(
        () => expect(canvasPushes().filter((s) => s.tag === "held_prompt")).toHaveLength(3),
        { timeout: 3000 },
      );
    });
  });
});

describe("AnnotateTab flags", () => {
  /** Mount over `loaded`, a produced document holding one box, unflagged when omitted. */
  async function mountWith(loaded: LoadedLabels = LANDED.box.document) {
    loadSpy.mockImplementation(() => Promise.resolve(loaded));
    await mountTab();
  }
  const comment = () => screen.getByRole("textbox", { name: "New flag comment" });

  it("f flags the image as a whole when nothing is focused", async () => {
    await mountWith();
    fireEvent.keyDown(window, { key: "f" });
    await flush();
    expect(screen.getByRole("dialog", { name: "Flags on this image" })).toBeInTheDocument();

    fireEvent.change(comment(), { target: { value: "glare" } });
    fireEvent.keyDown(comment(), { key: "Enter" });
    await flush();
    expect(saveSpy.mock.calls[0][0].gestures.flag).toEqual([{ text: "glare", target: null }]);
    expect(comment()).toHaveValue("");
  });

  it("keeps what was typed after a flag was submitted, when that submission lands", async () => {
    await mountWith();
    fireEvent.keyDown(window, { key: "f" });
    await flush();
    const answer = Promise.withResolvers<LoadedSave>();
    saveSpy.mockReturnValueOnce(answer.promise);
    fireEvent.change(comment(), { target: { value: "first" } });
    fireEvent.keyDown(comment(), { key: "Enter" });
    await flush();
    fireEvent.change(comment(), { target: { value: "second, not submitted" } });
    await act(() => Promise.resolve(answer.resolve(LANDED.flag_raised)));
    expect(comment()).toHaveValue("second, not submitted");
  });

  it("keeps the typed comment when its save was not taken, and clears it once one is", async () => {
    await mountWith();
    fireEvent.keyDown(window, { key: "f" });
    await flush();
    fireEvent.change(comment(), { target: { value: "glare" } });
    saveSpy.mockRejectedValueOnce(await versionConflict());
    fireEvent.keyDown(comment(), { key: "Enter" });
    await flush();
    expect(comment()).toHaveValue("glare");

    fireEvent.keyDown(comment(), { key: "Enter" });
    await flush();
    expect(comment()).toHaveValue("");
  });

  it("flags the focused annotation at a place inside it, through the save door", async () => {
    const [{ x1, y1, x2, y2 }] = LANDED.box.document.boxes;
    await mountWith();
    fireEvent.keyDown(window, { key: "ArrowDown" });
    await flush();
    fireEvent.click(screen.getByRole("button", { name: /Flag/ }));
    expect(screen.getByRole("dialog", { name: `Flags on ${SUBJECT} box` })).toBeInTheDocument();

    fireEvent.change(comment(), { target: { value: "open or closed?" } });
    fireEvent.keyDown(comment(), { key: "Enter" });
    await flush();
    expect(saveSpy.mock.calls[0][0]).toMatchObject({
      image_path: IMG1,
      gestures: { flag: [{ text: "open or closed?", target: { subject: SUBJECT } }] },
    });
    const { target } = saveSpy.mock.calls[0][0].gestures.flag[0];
    if (target === null || !("point" in target)) throw new Error("the flag targets no point");
    const [px, py] = target.point;
    expect(px).toBeGreaterThanOrEqual(x1);
    expect(px).toBeLessThanOrEqual(x2);
    expect(py).toBeGreaterThanOrEqual(y1);
    expect(py).toBeLessThanOrEqual(y2);
    expect(loadSpy).toHaveBeenCalledTimes(1);
  });

  it("counts open flags, marks them on the canvas and steps to the flagged items", async () => {
    // Two flags removed with their boxes, one open and one resolved.
    await mountWith(LOADED.flags_mixed);
    const open = LOADED.flags_mixed.flags.filter((f) => f.resolution === null).length;
    expect(
      screen.getByRole("button", { name: new RegExp(`Flag \\(${open}\\)`) }),
    ).toBeInTheDocument();
    expect(
      screen
        .getAllByTestId("k-text")
        .filter((t) => t.getAttribute("data-text") === FLAG_MARK.glyph),
    ).not.toHaveLength(0);

    fireEvent.change(screen.getByRole("combobox", { name: "Step through" }), {
      target: { value: "flagged" },
    });
    await flush();
    expect(screen.getByText(`/ ${open}`)).toBeInTheDocument();
  });

  it("resolves the focused item's flag with a reply, through the save door", async () => {
    await mountWith(LOADED.flagged);
    const [open] = LOADED.flagged.flags.filter((f) => f.resolution === null);
    const { text } = open.request;
    fireEvent.keyDown(window, { key: "ArrowDown" });
    await flush();
    fireEvent.keyDown(window, { key: "f" });
    await flush();
    expect(screen.getByText(text)).toBeInTheDocument();

    fireEvent.change(screen.getByRole("textbox", { name: `Reply to "${text}"` }), {
      target: { value: "open" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Resolve" }));
    await flush();
    expect(saveSpy.mock.calls[0][0].gestures?.resolve).toEqual({ [open.id]: "open" });
  });

  it("adopts the flags a plain save answers, so a removal's resolution shows at once", async () => {
    await mountWith(LOADED.flagged);
    saveSpy.mockResolvedValue(LANDED.flag_removed);
    act(() => useStore.getState().remove("boxes", 0));
    pressSave();
    await flush();
    expect(screen.getByRole("button", { name: /^.?Flag$/ })).toBeInTheDocument();
  });
});

describe("AnnotateTab engine prompt", () => {
  const stage = () => screen.getByTestId("canvas-stage");
  const STAGED = ANNOTATE_ANSWERS.staged;
  const BUCKET = STAGED.bucket;

  async function mountArmed(mode: "point" | "box") {
    useStore.setState((s) => ({ gui: { ...s.gui, mode } }));
    useStore.getState().setPrompt(true);
    const propose = vi.spyOn(api.annotate, "propose").mockResolvedValue(STAGED);
    const stagings = vi
      .spyOn(api.annotate, "stagings")
      .mockResolvedValue(ANNOTATE_ANSWERS.no_stagings);
    const proposals = vi
      .spyOn(api.annotate, "proposals")
      .mockResolvedValue(ANNOTATE_ANSWERS.staged_proposals);
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
    return { propose, stagings, proposals };
  }

  it("armed, clicks in point mode hold point prompts and place no point until Propose sends them", async () => {
    const { propose } = await mountArmed("point");
    fireEvent.click(stage(), { clientX: 120, clientY: 340 });
    fireEvent.click(stage(), { clientX: 200, clientY: 100 });
    await flush();

    expect(propose).not.toHaveBeenCalled();
    expect(useStore.getState().annotateUi.pendingPrompts.points).toEqual([
      [120, 340],
      [200, 100],
    ]);
    expect(useStore.getState().canvas.points).toEqual([]);

    fireEvent.click(screen.getByRole("button", { name: "toolbar-propose" }));
    await flush();

    expect(propose).toHaveBeenCalledWith({
      image_path: IMG1,
      subject: SUBJECT,
      user: "grower",
      prompt: {
        points: [
          [120, 340],
          [200, 100],
        ],
        boxes: [],
      },
    });
    expect(useStore.getState().annotateUi.pendingPrompts).toEqual({
      points: [],
      boxes: [],
      found: [],
    });
  });

  it("armed, a box drawn in box mode is held as a prompt and adds no box", async () => {
    const { propose } = await mountArmed("box");
    fireEvent.mouseDown(stage(), { clientX: 10, clientY: 20, button: 0 });
    fireEvent.mouseUp(stage(), { clientX: 90, clientY: 80 });
    await flush();
    fireEvent.click(screen.getByRole("button", { name: "toolbar-propose" }));
    await flush();

    expect(propose).toHaveBeenCalledWith({
      image_path: IMG1,
      subject: SUBJECT,
      user: "grower",
      prompt: { points: [], boxes: [[10, 20, 90, 80]] },
    });
    expect(useStore.getState().canvas.boxes).toEqual([]);
  });

  it("armed, a click with no subject holds nothing and says so", async () => {
    const { propose } = await mountArmed("point");
    act(() => useStore.getState().setActiveSubject(null));
    fireEvent.click(stage(), { clientX: 120, clientY: 340 });
    await flush();

    expect(propose).not.toHaveBeenCalled();
    expect(useStore.getState().annotateUi.pendingPrompts.points).toEqual([]);
    expect(screen.getByTestId("entry-in-place")).toHaveTextContent("Choose a subject first.");
  });

  it("held prompts are dropped when the image changes", async () => {
    await mountArmed("point");
    fireEvent.click(stage(), { clientX: 120, clientY: 340 });
    act(() => {
      changeSelection({ current_image_index: 1 });
    });
    await flush();

    expect(useStore.getState().annotateUi.pendingPrompts.points).toEqual([]);
  });

  it("Find holds the detector's boxes as prompts, stages nothing, and a right-click drops one", async () => {
    const { propose } = await mountArmed("box");
    const find = vi.spyOn(api.annotate, "find").mockResolvedValue(ANNOTATE_ANSWERS.found);
    fireEvent.click(screen.getByRole("button", { name: "toolbar-phrase" }));
    await flush();

    expect(find).toHaveBeenCalledWith({
      image_path: IMG1,
      phrase: "open bud",
      user: "grower",
    });
    expect(propose).not.toHaveBeenCalled();
    const [dropped, kept] = ANNOTATE_ANSWERS.found.boxes;
    expect(useStore.getState().annotateUi.pendingPrompts.found).toEqual([dropped, kept]);
    // The held boxes are drawn on the canvas, so the find raises no message.
    expect(listed()).toEqual([]);

    fireEvent.contextMenu(stage(), { clientX: dropped[0] + 10, clientY: dropped[1] + 10 });
    await flush();

    expect(useStore.getState().annotateUi.pendingPrompts.found).toEqual([kept]);
    fireEvent.click(screen.getByRole("button", { name: "toolbar-propose" }));
    await flush();
    expect(propose).toHaveBeenCalledWith(
      expect.objectContaining({ prompt: { points: [], boxes: [kept] } }),
    );
  });

  it("a second Find replaces what the first found and keeps the boxes the person drew", async () => {
    const { propose } = await mountArmed("box");
    fireEvent.mouseDown(stage(), { clientX: 10, clientY: 20, button: 0 });
    fireEvent.mouseUp(stage(), { clientX: 90, clientY: 80 });
    await flush();
    const find = vi.spyOn(api.annotate, "find").mockResolvedValueOnce(ANNOTATE_ANSWERS.found);
    fireEvent.click(screen.getByRole("button", { name: "toolbar-phrase" }));
    await flush();
    find.mockResolvedValueOnce(ANNOTATE_ANSWERS.found_other);
    fireEvent.click(screen.getByRole("button", { name: "toolbar-phrase" }));
    await flush();

    expect(useStore.getState().annotateUi.pendingPrompts).toEqual({
      points: [],
      boxes: [[10, 20, 90, 80]],
      found: ANNOTATE_ANSWERS.found_other.boxes,
    });
    fireEvent.click(screen.getByRole("button", { name: "toolbar-propose" }));
    await flush();
    expect(propose).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt: { points: [], boxes: [[10, 20, 90, 80], ...ANNOTATE_ANSWERS.found_other.boxes] },
      }),
    );
  });

  it("Find with nothing found says so and holds nothing", async () => {
    await mountArmed("box");
    vi.spyOn(api.annotate, "find").mockResolvedValue(ANNOTATE_ANSWERS.found_none);
    fireEvent.click(screen.getByRole("button", { name: "toolbar-phrase" }));
    await flush();

    expect(useStore.getState().annotateUi.pendingPrompts.found).toEqual([]);
    expect(listed()).toEqual([
      expect.objectContaining({
        kind: "notice",
        level: "info",
        sentence: 'Nothing found for "open bud".',
        context: { project: TEST_PROJECT.id, target: IMG1_NAME },
      }),
    ]);
  });

  it("shows the answered proposals under their shape's tool and focuses the first, the bucket unselected, raising no message", async () => {
    const { stagings, proposals } = await mountArmed("point");
    stagings.mockResolvedValue(ANNOTATE_ANSWERS.stagings);
    fireEvent.click(stage(), { clientX: 120, clientY: 340 });
    fireEvent.click(screen.getByRole("button", { name: "toolbar-propose" }));
    await flush();
    await flush();

    await waitFor(() => expect(proposals).toHaveBeenCalledWith(IMG1, BUCKET));
    expect(useStore.getState().gui.dataset?.bucket).toBeNull();
    expect(screen.getByTestId("toolbar")).toHaveAttribute("data-view-proposals", "true");
    expect(listed()).toEqual([]);
    await waitFor(() =>
      expect(useStore.getState().canvas.focus).toEqual({
        kind: "proposal",
        index: 0,
        bucket: BUCKET,
      }),
    );
    // The engine staged each candidate's outline, so its proposals are polygons.
    expect(useStore.getState().gui.mode).toBe("polygon");
  });

  it("serves a staging beside the selected bucket's proposals, each decided in its own bucket", async () => {
    const { stagings, proposals } = await mountArmed("box");
    const selected = ANNOTATE_ANSWERS.buckets.single;
    changeSelection({ bucket: selected });
    stagings.mockResolvedValue(ANNOTATE_ANSWERS.stagings);
    proposals.mockImplementation((_image, bucket) =>
      Promise.resolve(
        bucket === BUCKET ? ANNOTATE_ANSWERS.staged_proposals : ANNOTATE_ANSWERS.single_proposals,
      ),
    );
    await waitFor(() => expect(proposals).toHaveBeenCalledWith(expect.any(String), BUCKET));
    await flush();

    // The selected bucket's box proposal draws as a dotted box, the staging's as polygons.
    expect(dottedRects()).toHaveLength(1);
    fireEvent.keyDown(window, { key: "ArrowDown" });
    await flush();
    expect(useStore.getState().canvas.focus).toEqual({
      kind: "proposal",
      index: 0,
      bucket: BUCKET,
    });
    fireEvent.keyDown(window, { key: "a" });
    await flush();
    expect(saveSpy.mock.calls[0][0]).toMatchObject({
      gestures: { decide: { bucket: BUCKET, verdicts: { 0: "accepted" } } },
    });
  });

  it("refuses a second send while one is in flight", async () => {
    const { propose } = await mountArmed("point");
    propose.mockReturnValue(new Promise(() => {}));
    fireEvent.click(stage(), { clientX: 120, clientY: 340 });
    fireEvent.click(screen.getByRole("button", { name: "toolbar-propose" }));
    await flush();
    fireEvent.click(stage(), { clientX: 200, clientY: 200 });
    fireEvent.click(screen.getByRole("button", { name: "toolbar-propose" }));
    await flush();

    expect(propose).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("toolbar")).toHaveAttribute("data-proposing", "true");
  });

  it("an answer arriving after the person moved on serves nothing and names its bucket", async () => {
    const { propose, proposals } = await mountArmed("point");
    const answer = Promise.withResolvers<Awaited<ReturnType<typeof api.annotate.propose>>>();
    propose.mockReturnValue(answer.promise);
    fireEvent.click(stage(), { clientX: 120, clientY: 340 });
    fireEvent.click(screen.getByRole("button", { name: "toolbar-propose" }));
    await flush();
    act(() => {
      changeSelection({ current_image_index: 1 });
    });
    await flush();
    act(() => answer.resolve(STAGED));
    await flush();

    expect(useStore.getState().gui.dataset?.current_image_index).toBe(1);
    expect(proposals).not.toHaveBeenCalledWith(expect.any(String), BUCKET);
    expect(listed()).toEqual([
      expect.objectContaining({
        kind: "notice",
        level: "info",
        sentence: `${STAGED.proposal_count} proposals for ${IMG1_NAME}, staged under ${BUCKET}.`,
        context: { project: TEST_PROJECT.id, target: IMG1_NAME },
      }),
    ]);
  });

  it("a refused prompt is refused on the canvas, carrying the route's text", async () => {
    const { propose } = await mountArmed("point");
    const refusal = ANNOTATE_ANSWERS.engine_refusal;
    propose.mockRejectedValue(await thrown(refusal));
    fireEvent.click(stage(), { clientX: 120, clientY: 340 });
    fireEvent.click(screen.getByRole("button", { name: "toolbar-propose" }));
    await flush();

    expect(screen.getByTestId("entry-in-place")).toHaveTextContent(refusal.body.detail);
    expect(listed()).toEqual([expect.objectContaining({ kind: "failure", level: "error" })]);
  });

  it("a prompt the engine answers with nothing serves nothing and says so", async () => {
    const { propose, proposals } = await mountArmed("point");
    propose.mockResolvedValue(ANNOTATE_ANSWERS.nothing_proposed);
    fireEvent.click(stage(), { clientX: 120, clientY: 340 });
    fireEvent.click(screen.getByRole("button", { name: "toolbar-propose" }));
    await flush();

    expect(proposals).not.toHaveBeenCalled();
    expect(listed()).toEqual([
      expect.objectContaining({
        kind: "notice",
        level: "info",
        sentence: `No proposal for ${IMG1_NAME}.`,
        context: { project: TEST_PROJECT.id, target: IMG1_NAME },
      }),
    ]);
  });

  it("disarmed, a click in point mode places a point and sends no prompt", async () => {
    const { propose } = await mountArmed("point");
    act(() => useStore.getState().setPrompt(false));
    fireEvent.click(stage(), { clientX: 120, clientY: 340 });
    await flush();

    expect(propose).not.toHaveBeenCalled();
    expect(useStore.getState().canvas.points).toEqual([
      { x: 120, y: 340, subject: SUBJECT, attributes: {}, iscrowd: false },
    ]);
  });
});

describe("AnnotateTab refusals on the canvas, each kind on its own line", () => {
  const GESTURE = "Box too small to keep. Drag out a bigger area.";
  const RESIZED = "Box too small to keep; the resize was undone.";
  const PROMPT = "Could not propose: network down";
  const FIND = "Could not find: network down";
  const PROPOSALS = "Could not load proposals: network down";
  const stage = () => screen.getByTestId("canvas-stage");
  const propose = () => fireEvent.click(screen.getByRole("button", { name: "toolbar-propose" }));
  const find = () => fireEvent.click(screen.getByRole("button", { name: "toolbar-phrase" }));
  const toggleProposalsView = () =>
    fireEvent.click(screen.getByRole("button", { name: "toolbar-view-proposals" }));

  /** The sentence of each entry the canvas shows in place, in the order it shows them. */
  const standing = () =>
    onCanvas().map((entry) => entry.querySelector("p")?.firstChild?.textContent);

  /** Mount over the first image, its document holding `placed`. */
  async function mount(placed: ReturnType<typeof servedAnnotation>[] = []) {
    useStore.getState().setRegistry(registryOf(SUBJECT), UNSTORED);
    if (placed.length) {
      loadSpy.mockImplementation((imagePath) => Promise.resolve(loadHolding(imagePath, placed)));
    }
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    await flush();
  }

  /** Drag a box out from (`x`, `y`) by `side` on each axis, in the tool in use. */
  async function dragBox(x: number, y: number, side: number) {
    fireEvent.mouseDown(stage(), { clientX: x, clientY: y, button: 0 });
    fireEvent.mouseMove(stage(), { clientX: x + side, clientY: y + side });
    await nextFrame();
    fireEvent.mouseUp(stage(), { clientX: x + side, clientY: y + side });
    await flush();
  }

  /** Refuse a box drawn below the minimum side, returning to the tool in use. */
  async function refuseGesture() {
    const tool = useStore.getState().gui.mode;
    act(() => useStore.getState().setMode("box"));
    await dragBox(100, 100, 2);
    act(() => useStore.getState().setMode(tool));
  }

  /** Add a box, select it, and drag its bottom-right corner to (`to`, `to`), not released. */
  async function dragAddedCorner(to: number) {
    act(addBox); // {x1:10, y1:10, x2:50, y2:50}
    fireEvent.mouseDown(stage(), { clientX: 30, clientY: 30, button: 0 });
    fireEvent.mouseUp(stage(), { clientX: 30, clientY: 30 });
    await flush();
    fireEvent.mouseDown(stage(), { clientX: 50, clientY: 50, button: 0 });
    fireEvent.mouseMove(stage(), { clientX: to, clientY: to });
    await nextFrame();
  }

  /** Step to the image at `index`, once its load has been asked for `loads` times. */
  async function stepTo(index: number, loads: number) {
    act(() => changeSelection({ current_image_index: index }));
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(loads));
    await flush();
  }

  it("a failed proposals read leaves the canvas once a later read lands, and stays listed", async () => {
    const stagings = vi
      .spyOn(api.annotate, "stagings")
      .mockRejectedValue(new Error("network down"));
    await mount();
    expect(standing()).toEqual([PROPOSALS]);

    stagings.mockResolvedValue(ANNOTATE_ANSWERS.no_stagings);
    toggleProposalsView();
    toggleProposalsView();
    await flush();

    expect(standing()).toEqual([]);
    expect(listed()).toEqual([expect.objectContaining({ sentence: PROPOSALS, key: undefined })]);
  });

  it("a resize undone by a save stays on the canvas once that save is adopted", async () => {
    const stagings = vi
      .spyOn(api.annotate, "stagings")
      .mockResolvedValue(ANNOTATE_ANSWERS.no_stagings);
    await mount();
    await dragAddedCorner(11);
    pressSave();
    // The adopted save reads the proposals again, and that read lands.
    await waitFor(() => expect(stagings).toHaveBeenCalledTimes(2));
    await flush();

    expect(standing()).toEqual([RESIZED]);
  });

  it("a failed proposals read stays on the canvas when a box is kept", async () => {
    vi.spyOn(api.annotate, "stagings").mockRejectedValue(new Error("network down"));
    await mount();
    await dragBox(100, 100, 60);

    expect(useStore.getState().canvas.boxes).toHaveLength(1);
    expect(standing()).toEqual([PROPOSALS]);
  });

  describe("a refused gesture leaves the canvas, and stays listed, when a gesture is kept", () => {
    const expectGestureEnded = () => {
      expect(standing()).toEqual([]);
      expect(listed()).toEqual([expect.objectContaining({ sentence: GESTURE, key: undefined })]);
    };

    it("by a committed polygon", async () => {
      await renderPolygonCanvas();
      await refuseGesture();
      expect(standing()).toEqual([GESTURE]);

      fireEvent.click(stage(), { clientX: 600, clientY: 600 });
      fireEvent.click(stage(), { clientX: 700, clientY: 600 });
      fireEvent.click(stage(), { clientX: 700, clientY: 700 });
      fireEvent.doubleClick(stage(), { clientX: 700, clientY: 700 });
      await flush();
      expect(useStore.getState().canvas.polygons).toHaveLength(3);
      expectGestureEnded();
    });

    it("by a kept resize", async () => {
      await mount();
      await refuseGesture();
      expect(standing()).toEqual([GESTURE]);

      await dragAddedCorner(30);
      fireEvent.mouseUp(stage(), { clientX: 30, clientY: 30 });
      await flush();
      expect(useStore.getState().canvas.boxes[0]).toMatchObject({ x2: 30, y2: 30 });
      expectGestureEnded();
    });

    it("by a held prompt box", async () => {
      await mount();
      await refuseGesture();
      expect(standing()).toEqual([GESTURE]);

      act(() => useStore.getState().setPrompt(true));
      await dragBox(10, 20, 70);
      expect(useStore.getState().annotateUi.pendingPrompts.boxes).toEqual([[10, 20, 80, 90]]);
      expectGestureEnded();
    });

    it("by a held prompt point", async () => {
      await mount();
      await refuseGesture();
      expect(standing()).toEqual([GESTURE]);

      act(() => {
        useStore.getState().setMode("point");
        useStore.getState().setPrompt(true);
      });
      fireEvent.click(stage(), { clientX: 120, clientY: 340 });
      await flush();
      expect(useStore.getState().annotateUi.pendingPrompts.points).toEqual([[120, 340]]);
      expectGestureEnded();
    });

    it("by a point drag ended on release", async () => {
      await mount([servedAnnotation({ subject: SUBJECT, point: [400, 400], index: 0 })]);
      await refuseGesture();
      expect(standing()).toEqual([GESTURE]);

      act(() => useStore.getState().setMode("point"));
      fireEvent.mouseDown(stage(), { clientX: 400, clientY: 400, button: 0 });
      fireEvent.mouseMove(stage(), { clientX: 450, clientY: 450 });
      await nextFrame();
      fireEvent.mouseUp(stage(), { clientX: 450, clientY: 450 });
      await flush();
      expect(useStore.getState().canvas.points[0]).toMatchObject({ x: 450, y: 450 });
      expectGestureEnded();
    });

    it("by a vertex drag ended on release", async () => {
      await renderPolygonCanvas();
      await refuseGesture();
      expect(standing()).toEqual([GESTURE]);

      act(() => useStore.getState().setFocus({ kind: "polygon", index: 0 }));
      fireEvent.mouseDown(stage(), { clientX: 10, clientY: 10, button: 0 });
      fireEvent.mouseMove(stage(), { clientX: 20, clientY: 20 });
      await nextFrame();
      fireEvent.mouseUp(stage(), { clientX: 20, clientY: 20 });
      await flush();
      expect(useStore.getState().canvas.polygons[0].rings[0][0]).toEqual([20, 20]);
      expectGestureEnded();
    });
  });

  it("a refused prompt leaves the canvas once a prompt is answered, and stays listed", async () => {
    const proposeSpy = vi
      .spyOn(api.annotate, "propose")
      .mockRejectedValueOnce(new Error("network down"));
    await mount();
    act(() => useStore.getState().addPromptPoint([120, 340]));
    propose();
    await flush();
    expect(standing()).toEqual([PROMPT]);

    proposeSpy.mockResolvedValueOnce(ANNOTATE_ANSWERS.nothing_proposed);
    propose();
    await flush();

    expect(standing()).toEqual([]);
    expect(listed()).toContainEqual(expect.objectContaining({ sentence: PROMPT, key: undefined }));
  });

  it("a refused find leaves the canvas once a find is answered, and stays listed", async () => {
    const findSpy = vi.spyOn(api.annotate, "find").mockRejectedValueOnce(new Error("network down"));
    await mount();
    find();
    await flush();
    expect(standing()).toEqual([FIND]);

    findSpy.mockResolvedValueOnce(ANNOTATE_ANSWERS.found_none);
    find();
    await flush();

    expect(standing()).toEqual([]);
    expect(listed()).toContainEqual(expect.objectContaining({ sentence: FIND, key: undefined }));
  });

  describe("with a refused gesture, prompt, find and proposals read standing at once", () => {
    let proposeSpy: MockInstance<typeof api.annotate.propose>;
    let findSpy: MockInstance<typeof api.annotate.find>;
    let stagings: MockInstance<typeof api.annotate.stagings>;

    beforeEach(async () => {
      stagings = vi.spyOn(api.annotate, "stagings").mockRejectedValue(new Error("network down"));
      proposeSpy = vi.spyOn(api.annotate, "propose").mockRejectedValue(new Error("network down"));
      findSpy = vi.spyOn(api.annotate, "find").mockRejectedValue(new Error("network down"));
      await mount();
      act(() => useStore.getState().addPromptPoint([120, 340]));
      propose();
      await flush();
      find();
      await flush();
      await refuseGesture();
      expect(standing()).toEqual([GESTURE, PROMPT, FIND, PROPOSALS]);
    });

    it("another refused gesture and a kept one touch no other kind's", async () => {
      act(() => useStore.getState().setActiveSubject(null));
      fireEvent.mouseDown(stage(), { clientX: 300, clientY: 300, button: 0 });
      fireEvent.mouseUp(stage(), { clientX: 300, clientY: 300 });
      await flush();
      expect(standing()).toEqual(["Choose a subject first.", PROMPT, FIND, PROPOSALS]);

      act(() => useStore.getState().setActiveSubject(SUBJECT));
      await dragBox(100, 100, 60);
      expect(standing()).toEqual([PROMPT, FIND, PROPOSALS]);
    });

    it("another refused prompt and an answered one touch no other kind's", async () => {
      proposeSpy.mockRejectedValueOnce(new Error("engine down"));
      propose();
      await flush();
      expect(standing()).toEqual([GESTURE, "Could not propose: engine down", FIND, PROPOSALS]);

      proposeSpy.mockResolvedValueOnce(ANNOTATE_ANSWERS.nothing_proposed);
      propose();
      await flush();
      expect(standing()).toEqual([GESTURE, FIND, PROPOSALS]);
    });

    it("another refused find and an answered one touch no other kind's", async () => {
      findSpy.mockRejectedValueOnce(new Error("detector down"));
      find();
      await flush();
      expect(standing()).toEqual([GESTURE, PROMPT, "Could not find: detector down", PROPOSALS]);

      findSpy.mockResolvedValueOnce(ANNOTATE_ANSWERS.found_none);
      find();
      await flush();
      expect(standing()).toEqual([GESTURE, PROMPT, PROPOSALS]);
    });

    it("a landed proposals read and another failed one touch no other kind's", async () => {
      toggleProposalsView();
      await flush();
      expect(standing()).toEqual([GESTURE, PROMPT, FIND]);

      stagings.mockRejectedValue(new Error("disk busy"));
      toggleProposalsView();
      await flush();
      expect(standing()).toEqual([GESTURE, PROMPT, FIND, "Could not load proposals: disk busy"]);
    });

    /** Keep a box in the box tool, a gesture that succeeds. */
    async function keepBox() {
      act(() => useStore.getState().setMode("box"));
      await dragBox(600, 600, 60);
    }

    /** Lay `polygons` on the canvas in the polygon tool, the one at `focus` selected, cut armed. */
    function armCut(polygons: (typeof POLY_A)[], focus: number | null) {
      act(() => {
        useStore.getState().setMode("polygon");
        seedPolygons(polygons);
        if (focus !== null) useStore.getState().setFocus({ kind: "polygon", index: focus });
        useStore.getState().setCut(true);
      });
    }

    it("a Propose with no subject replaces the refused prompt alone", async () => {
      act(() => useStore.getState().setActiveSubject(null));
      propose();
      await flush();
      expect(standing()).toEqual([GESTURE, "Choose a subject first.", FIND, PROPOSALS]);
    });

    it("a Find with no subject replaces the refused find alone", async () => {
      act(() => useStore.getState().setActiveSubject(null));
      find();
      await flush();
      expect(standing()).toEqual([GESTURE, PROMPT, "Choose a subject first.", PROPOSALS]);
    });

    it("an undone resize replaces the refused gesture alone, and a kept box ends it alone", async () => {
      await dragAddedCorner(11);
      fireEvent.mouseUp(stage(), { clientX: 11, clientY: 11 });
      await flush();
      expect(standing()).toEqual([RESIZED, PROMPT, FIND, PROPOSALS]);

      await keepBox();
      expect(standing()).toEqual([PROMPT, FIND, PROPOSALS]);
    });

    it("a cut with no polygon selected replaces the refused gesture alone, and a kept box ends it alone", async () => {
      armCut([POLY_A, POLY_B], null);
      fireEvent.click(stage(), { clientX: 500, clientY: 500, button: 0 });
      expect(standing()).toEqual([
        expect.stringMatching(/^Select a polygon to cut/),
        PROMPT,
        FIND,
        PROPOSALS,
      ]);

      await keepBox();
      expect(standing()).toEqual([PROMPT, FIND, PROPOSALS]);
    });

    it("a cut whose polygon changed replaces the refused gesture alone, and a kept box ends it alone", async () => {
      armCut([POLY_A, POLY_B], 0);
      fireEvent.click(stage(), { clientX: 105, clientY: 0, button: 0 });
      act(() => useStore.getState().setFocus({ kind: "polygon", index: 1 }));
      fireEvent.click(stage(), { clientX: 105, clientY: 250, button: 0 });
      expect(standing()).toEqual([
        expect.stringMatching(/^The polygon changed since the first click/),
        PROMPT,
        FIND,
        PROPOSALS,
      ]);

      await keepBox();
      expect(standing()).toEqual([PROMPT, FIND, PROPOSALS]);
    });

    it("a cut across a shape of two parts replaces the refused gesture alone, and a kept box ends it alone", async () => {
      armCut([{ ...POLY_A, rings: [POLY_A.rings[0], POLY_B.rings[0]] }], 0);
      fireEvent.click(stage(), { clientX: 250, clientY: 250, button: 0 });
      fireEvent.click(stage(), { clientX: 250, clientY: 450, button: 0 });
      expect(standing()).toEqual([
        expect.stringMatching(/^This shape covers 2 separate parts/),
        PROMPT,
        FIND,
        PROPOSALS,
      ]);

      await keepBox();
      expect(standing()).toEqual([PROMPT, FIND, PROPOSALS]);
    });

    it("a cut that misses its polygon replaces the refused gesture alone, and a kept box ends it alone", async () => {
      armCut([POLY_A, POLY_B], 0);
      fireEvent.click(stage(), { clientX: 105, clientY: 0, button: 0 });
      fireEvent.click(stage(), { clientX: 105, clientY: 5, button: 0 });
      expect(standing()).toEqual([CUT_MISSES_REFUSAL, PROMPT, FIND, PROPOSALS]);

      await keepBox();
      expect(standing()).toEqual([PROMPT, FIND, PROPOSALS]);
    });
  });

  it("a refused gesture, prompt and find leave the canvas with their image, and stay listed on return", async () => {
    const stagings = vi
      .spyOn(api.annotate, "stagings")
      .mockResolvedValue(ANNOTATE_ANSWERS.no_stagings);
    vi.spyOn(api.annotate, "propose").mockRejectedValue(new Error("network down"));
    vi.spyOn(api.annotate, "find").mockRejectedValue(new Error("network down"));
    await mount();
    act(() => useStore.getState().addPromptPoint([120, 340]));
    propose();
    await flush();
    find();
    await flush();
    await refuseGesture();
    expect(standing()).toEqual([GESTURE, PROMPT, FIND]);

    await stepTo(1, 2);
    expect(standing()).toEqual([]);
    await stepTo(0, 3);
    // The returned-to image's proposals read lands, which ends nothing but its own kind.
    await waitFor(() => expect(stagings).toHaveBeenLastCalledWith(IMG1));
    await flush();

    expect(standing()).toEqual([]);
    for (const sentence of [GESTURE, PROMPT, FIND]) {
      expect(listed()).toContainEqual(expect.objectContaining({ sentence, key: undefined }));
    }
  });

  it("a refused gesture stays off its image's canvas when that image left the view while the tab was away", async () => {
    await mount();
    await refuseGesture();
    expect(standing()).toEqual([GESTURE]);
    cleanup();
    act(() => changeSelection({ current_image_index: 1 }));
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();
    await stepTo(0, 3);

    expect(standing()).toEqual([]);
    expect(listed()).toEqual([expect.objectContaining({ sentence: GESTURE, key: undefined })]);
  });

  it("a refused gesture stays on the canvas when the tab is left and its image is in view on return", async () => {
    await mount();
    await refuseGesture();
    cleanup();
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(2));
    await flush();

    expect(standing()).toEqual([GESTURE]);
  });

  it("a refused gesture leaves the canvas when another capture's image of the same name is stepped to", async () => {
    const [twin] = LATER_SELECTION.images;
    expect(twin.name).toBe(IMG1_NAME);
    expect(twin.path).not.toBe(IMG1);
    await mount();
    await refuseGesture();
    const raw = useStore.getState().gui.dataset!;
    act(() => useStore.setState((s) => ({ gui: { ...s.gui, dataset: LATER_SELECTION } })));
    await waitFor(() => expect(loadSpy).toHaveBeenLastCalledWith(twin.path));
    await flush();
    act(() => useStore.setState((s) => ({ gui: { ...s.gui, dataset: raw } })));
    await waitFor(() => expect(loadSpy).toHaveBeenLastCalledWith(IMG1));
    await flush();

    expect(standing()).toEqual([]);
    expect(listed()).toEqual([expect.objectContaining({ sentence: GESTURE, key: undefined })]);
  });

  it("says a subject is needed on another capture's image of the same name, once said on the first", async () => {
    const [twin] = LATER_SELECTION.images;
    await mount();
    act(() => useStore.getState().setActiveSubject(null));
    fireEvent.mouseDown(stage(), { clientX: 300, clientY: 300, button: 0 });
    fireEvent.mouseUp(stage(), { clientX: 300, clientY: 300 });
    await flush();
    expect(standing()).toEqual(["Choose a subject first."]);

    act(() => useStore.setState((s) => ({ gui: { ...s.gui, dataset: LATER_SELECTION } })));
    await waitFor(() => expect(loadSpy).toHaveBeenLastCalledWith(twin.path));
    await flush();
    expect(standing()).toEqual([]);
    fireEvent.mouseDown(stage(), { clientX: 4, clientY: 4, button: 0 });
    fireEvent.mouseUp(stage(), { clientX: 4, clientY: 4 });
    await flush();

    expect(standing()).toEqual(["Choose a subject first."]);
  });

  it("a prompt refused after its image was left is listed naming that image, on neither canvas", async () => {
    const answer = Promise.withResolvers<Awaited<ReturnType<typeof api.annotate.propose>>>();
    vi.spyOn(api.annotate, "propose").mockReturnValue(answer.promise);
    await mount();
    act(() => useStore.getState().addPromptPoint([120, 340]));
    propose();
    await flush();
    await stepTo(1, 2);
    act(() => answer.reject(new Error("network down")));
    await flush();

    expect(standing()).toEqual([]);
    expect(listed()).toEqual([
      expect.objectContaining({
        sentence: PROMPT,
        key: undefined,
        context: { project: TEST_PROJECT.id, target: IMG1_NAME },
      }),
    ]);
    await stepTo(0, 3);
    expect(standing()).toEqual([]);
  });

  it("a resize undone by leaving its image is listed naming that image, not on the next one's canvas", async () => {
    await mount();
    await dragAddedCorner(11);
    await stepTo(1, 2);

    expect(standing()).toEqual([]);
    expect(listed()).toEqual([
      expect.objectContaining({
        sentence: RESIZED,
        key: undefined,
        context: { project: TEST_PROJECT.id, target: IMG1_NAME },
      }),
    ]);
  });

  it("a refused save leaves the canvas once a later save is adopted", async () => {
    await mount();
    act(addBox);
    saveSpy.mockRejectedValueOnce(new Error("network down"));
    pressSave();
    await flush();
    expect(standing()).toEqual([
      "Could not save annotations (network down). Your edits are kept in the editor; press Save to retry.",
    ]);

    pressSave();
    await flush();

    expect(saveSpy).toHaveBeenCalledTimes(2);
    expect(standing()).toEqual([]);
  });

  it("the Reload a conflict offers leaves the canvas when its image's next load fails", async () => {
    await mount();
    act(addBox);
    saveSpy.mockRejectedValueOnce(await versionConflict());
    pressSave();
    await flush();
    expect(screen.getByRole("button", { name: "Reload" })).toBeInTheDocument();

    await stepTo(1, 2);
    loadSpy.mockRejectedValueOnce(new Error("network down"));
    await stepTo(0, 3);

    expect(standing()).toEqual([
      "Could not load this image's labels: network down. Saving is off.",
    ]);
    expect(screen.queryByRole("button", { name: "Reload" })).toBeNull();
  });
});

describe("AnnotateTab heading", () => {
  it("renders exactly one top-level heading naming the tab", async () => {
    render(<AnnotateTab />);
    await waitFor(() => expect(loadSpy).toHaveBeenCalledTimes(1));
    const headings = screen.getAllByRole("heading", { level: 1 });
    expect(headings).toHaveLength(1);
    expect(headings[0]).toHaveTextContent("Annotate");
  });
});
