import { beforeEach, describe, expect, it, vi } from "vitest";

import { sessionsApi } from "@/api/sessions";
import type { ImageEventPayload, SessionWrite, SnapshotProject } from "@/api/types.generated";
import { endRecordedSession, postHeldContributions } from "@/lib/sessionLifecycle";
import { useStore } from "@/store";
import { textOf } from "@/store/slices/toasts";
import { thrown } from "@/test/refusals";
import { SESSION_ANSWERS, SESSION_WRITES, sessionOf } from "@/test/sessions";
import { openTestProject, stateEnvelope, TEST_PROJECT, testProject } from "@/test/store";

const initialStoreState = useStore.getState();

/** A project other than the one each test opens. */
const OTHER = testProject({ id: "f6e5d4c3b2a1", path: "C:/other" });

function contribution(image: string): ImageEventPayload {
  return {
    contribution_id: `c-${image}`,
    image_name: image,
    seconds: 2,
    annotations_added: 1,
    activity: "new_annotation",
    user: "jordan",
    project_id: TEST_PROJECT.id,
    started: null,
  };
}

/** The produced answers naming the first session, open and ended, and the held one after it. */
const { first: FIRST, firstEnded: FIRST_ENDED, held: HELD } = SESSION_WRITES;
const S1 = sessionOf(FIRST).started;
const S2 = sessionOf(HELD).started;

/** The refusals of a contribution naming an ended session, and of one naming another person's. */
const { ended_named: ENDED_NAMED, another_named: ANOTHER_NAMED } = SESSION_ANSWERS;

/** The envelope of a snapshot naming `project`, the GUI state otherwise as the store holds it. */
const naming = (project: SnapshotProject) =>
  stateEnvelope({
    state: useStore.getState().gui,
    version: useStore.getState().wsVersion,
    project,
  });

async function flush() {
  for (let i = 0; i < 5; i++) await Promise.resolve();
}

beforeEach(() => {
  useStore.setState(initialStoreState, true);
  openTestProject();
  vi.spyOn(sessionsApi, "endAsPageLeaves").mockImplementation(() => {});
});

describe("session lifecycle", () => {
  it("posts held contributions one after another, each after the one before it answers", async () => {
    const first = Promise.withResolvers<SessionWrite>();
    const second = Promise.withResolvers<SessionWrite>();
    const post = vi
      .spyOn(sessionsApi, "imageEvent")
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const a = contribution("a.jpg");
    const b = contribution("b.jpg");
    useStore.setState({ heldContributions: [a, b] });

    postHeldContributions();
    await flush();
    expect(post).toHaveBeenCalledTimes(1);
    expect(post).toHaveBeenLastCalledWith(a);

    first.resolve(FIRST);
    await flush();
    expect(post).toHaveBeenCalledTimes(2);
    expect(post).toHaveBeenLastCalledWith(b);
    expect(useStore.getState().heldContributions).toEqual([b]);

    second.resolve(FIRST);
    await flush();
    expect(useStore.getState().heldContributions).toEqual([]);
    expect(useStore.getState().recordedSession).toEqual({
      project_id: TEST_PROJECT.id,
      started: S1,
      user: "jordan",
    });
  });

  it("posts a held visit of a project no longer open, in the order it was held", async () => {
    const post = vi.spyOn(sessionsApi, "imageEvent").mockResolvedValue(FIRST);
    const departed = { ...contribution("a.jpg"), project_id: OTHER.id, started: S2 };
    const current = contribution("b.jpg");
    useStore.setState({ heldContributions: [departed, current] });

    postHeldContributions();
    await flush();
    await flush();

    expect(post.mock.calls.map(([c]) => c)).toEqual([departed, current]);
    expect(useStore.getState().heldContributions).toEqual([]);
  });

  it("ends the session a pending contribution answers, once, after that answer", async () => {
    const pending = Promise.withResolvers<SessionWrite>();
    vi.spyOn(sessionsApi, "imageEvent").mockReturnValueOnce(pending.promise);
    useStore.setState({ heldContributions: [contribution("a.jpg")] });
    postHeldContributions();
    await flush();

    endRecordedSession();
    endRecordedSession();
    await flush();
    expect(sessionsApi.endAsPageLeaves).not.toHaveBeenCalled();

    pending.resolve(FIRST);
    await flush();
    expect(sessionsApi.endAsPageLeaves).toHaveBeenCalledTimes(1);
    expect(sessionsApi.endAsPageLeaves).toHaveBeenCalledWith({
      project_id: TEST_PROJECT.id,
      started: S1,
    });
    expect(useStore.getState().recordedSession).toBeNull();
  });

  it("sends no end for the session of a project the backend no longer has open", async () => {
    vi.spyOn(sessionsApi, "imageEvent").mockResolvedValueOnce(FIRST);
    useStore.setState({ heldContributions: [contribution("a.jpg")] });
    postHeldContributions();
    await flush();
    expect(useStore.getState().recordedSession).not.toBeNull();

    useStore.getState().mergeSnapshot(naming(OTHER));
    endRecordedSession();
    await flush();

    expect(sessionsApi.endAsPageLeaves).not.toHaveBeenCalled();
  });

  const SWITCHES: [string, () => void][] = [
    ["a snapshot naming another project", () => useStore.getState().mergeSnapshot(naming(OTHER))],
    [
      "a dataset opened in another project",
      () => useStore.getState().applyRestoredDataset(useStore.getState().gui.dataset, OTHER),
    ],
  ];

  it.each(SWITCHES)(
    "retires a contribution answered after %s and records no session for it",
    async (_name, switchProject) => {
      const pending = Promise.withResolvers<SessionWrite>();
      vi.spyOn(sessionsApi, "imageEvent").mockReturnValueOnce(pending.promise);
      useStore.setState({ heldContributions: [contribution("a.jpg")] });
      postHeldContributions();
      await flush();

      switchProject();
      pending.resolve(FIRST);
      await flush();
      endRecordedSession();
      await flush();

      expect(useStore.getState().heldContributions).toEqual([]);
      expect(useStore.getState().recordedSession).toBeNull();
      expect(sessionsApi.endAsPageLeaves).not.toHaveBeenCalled();
    },
  );

  it("sends no end when no session is recorded", async () => {
    endRecordedSession();
    await flush();

    expect(sessionsApi.endAsPageLeaves).not.toHaveBeenCalled();
  });

  it("retires a contribution the backend refuses by name, says why, and never posts it again", async () => {
    const reason = ENDED_NAMED.body.detail;
    const post = vi.spyOn(sessionsApi, "imageEvent").mockRejectedValue(await thrown(ENDED_NAMED));
    useStore.setState({ heldContributions: [contribution("a.jpg")], toasts: [] });

    postHeldContributions();
    await flush();
    postHeldContributions();
    await flush();

    expect(post).toHaveBeenCalledTimes(1);
    expect(useStore.getState().heldContributions).toEqual([]);
    expect(useStore.getState().recordedSession).toBeNull();
    expect(useStore.getState().toasts.map(textOf)).toEqual([`Not recorded (${reason}): a.jpg`]);
  });

  const UNCONFIRMED = "Recording not confirmed before their project was switched away from: a.jpg";

  it("says recording is unconfirmed, never absent, for a visit whose send failed before its project departed, and withdraws the promise to send it again", async () => {
    vi.spyOn(sessionsApi, "imageEvent").mockRejectedValueOnce(new TypeError("Failed to fetch"));
    useStore.setState({ heldContributions: [contribution("a.jpg")], toasts: [] });
    postHeldContributions();
    await flush();
    expect(useStore.getState().toasts).toHaveLength(1);

    useStore.getState().mergeSnapshot(naming(OTHER));

    expect(useStore.getState().heldContributions).toEqual([]);
    expect(useStore.getState().toasts.map(textOf)).toEqual([UNCONFIRMED]);
  });

  it("says recording is unconfirmed for a visit in flight when its project departed, and nothing more when the send then fails", async () => {
    const pending = Promise.withResolvers<SessionWrite>();
    vi.spyOn(sessionsApi, "imageEvent").mockReturnValueOnce(pending.promise);
    useStore.setState({ heldContributions: [contribution("a.jpg")], toasts: [] });
    postHeldContributions();
    await flush();

    useStore.getState().mergeSnapshot(naming(OTHER));
    pending.reject(new TypeError("Failed to fetch"));
    await flush();

    expect(useStore.getState().heldContributions).toEqual([]);
    expect(useStore.getState().toasts.map(textOf)).toEqual([UNCONFIRMED]);
  });

  it("withdraws the notice of unconfirmed visits once a resend confirms them", async () => {
    vi.spyOn(sessionsApi, "imageEvent")
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(FIRST);
    useStore.setState({ heldContributions: [contribution("a.jpg")], toasts: [] });
    postHeldContributions();
    await flush();
    expect(useStore.getState().toasts).toHaveLength(1);

    postHeldContributions();
    await flush();

    expect(useStore.getState().heldContributions).toEqual([]);
    expect(useStore.getState().toasts).toEqual([]);
  });

  it("names every visit refused, under each reason, in one notice that a refusal after its dismissal starts afresh", async () => {
    const ended = ENDED_NAMED.body.detail;
    const other = ANOTHER_NAMED.body.detail;
    const refused = { [ended]: await thrown(ENDED_NAMED), [other]: await thrown(ANOTHER_NAMED) };
    const post = vi.spyOn(sessionsApi, "imageEvent");
    for (const reason of [ended, other, ended, other, ended, ended, other]) {
      post.mockRejectedValueOnce(refused[reason]);
    }
    const images = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"];
    useStore.setState({
      user: "jordan",
      recordedSession: { project_id: TEST_PROJECT.id, started: S1, user: "jordan" },
      toasts: [],
    });
    const s = useStore.getState();
    for (const image of images) {
      s.startImageSessionTracking(image, Date.now() - 2000);
      s.closeSessionInterval();
    }

    postHeldContributions();
    for (let i = 0; i < images.length; i++) await flush();

    expect(useStore.getState().heldContributions).toEqual([]);
    expect(useStore.getState().toasts.map(textOf)).toEqual([
      `Not recorded (${ended}): a.jpg, c.jpg, e.jpg, f.jpg; ` +
        `Not recorded (${other}): b.jpg, d.jpg`,
    ]);

    s.dismissToast(useStore.getState().toasts[0].id);
    s.startImageSessionTracking("g.jpg", Date.now() - 2000);
    s.closeSessionInterval();
    postHeldContributions();
    await flush();

    expect(useStore.getState().toasts.map(textOf)).toEqual([`Not recorded (${other}): g.jpg`]);
  });

  it("forgets a session learned from one visit when another's repeat answers that it ended, so the next opens fresh", async () => {
    const post = vi
      .spyOn(sessionsApi, "imageEvent")
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce(FIRST)
      .mockResolvedValueOnce(FIRST_ENDED)
      .mockResolvedValueOnce(HELD);
    useStore.setState({ user: "jordan", recordedSession: null });
    const s = useStore.getState();
    for (const image of ["a.jpg", "b.jpg"]) {
      s.startImageSessionTracking(image, Date.now() - 2000);
      s.closeSessionInterval();
    }
    postHeldContributions();
    await flush();
    await flush();
    expect(useStore.getState().recordedSession).toMatchObject({ started: S1 });

    postHeldContributions();
    await flush();
    s.startImageSessionTracking("c.jpg", Date.now() - 2000);
    s.closeSessionInterval();
    postHeldContributions();
    await flush();

    expect(post.mock.calls.map(([c]) => [c.image_name, c.started])).toEqual([
      ["a.jpg", null],
      ["b.jpg", null],
      ["a.jpg", null],
      ["c.jpg", null],
    ]);
    expect(useStore.getState().recordedSession).toMatchObject({ started: S2 });
  });

  it("keeps a newer recorded session when a repeat answers that an older one ended", async () => {
    // The first session ended; the held one, opened after it, is the recorded one.
    vi.spyOn(sessionsApi, "imageEvent").mockResolvedValueOnce(FIRST_ENDED);
    const recorded = { project_id: TEST_PROJECT.id, started: S2, user: "jordan" };
    useStore.setState({ recordedSession: recorded, heldContributions: [contribution("a.jpg")] });

    postHeldContributions();
    await flush();

    expect(useStore.getState().recordedSession).toEqual(recorded);
  });

  it("forgets the recorded session a repeat answers has ended, so the next visit opens a fresh one", async () => {
    const post = vi
      .spyOn(sessionsApi, "imageEvent")
      .mockResolvedValueOnce(FIRST_ENDED)
      .mockResolvedValueOnce(HELD);
    useStore.setState({
      user: "jordan",
      recordedSession: { project_id: TEST_PROJECT.id, started: S1, user: "jordan" },
    });
    const s = useStore.getState();
    s.startImageSessionTracking("a.jpg", Date.now() - 2000);
    s.closeSessionInterval();
    postHeldContributions();
    await flush();

    s.startImageSessionTracking("b.jpg", Date.now() - 2000);
    s.closeSessionInterval();
    postHeldContributions();
    await flush();

    expect(post.mock.calls.map(([c]) => [c.image_name, c.started])).toEqual([
      ["a.jpg", S1],
      ["b.jpg", null],
    ]);
    expect(useStore.getState().recordedSession).toMatchObject({ started: S2 });
  });

  it("keeps visits with no answer held, names every one in one live notice, and records no session for them", async () => {
    vi.spyOn(sessionsApi, "imageEvent").mockRejectedValue(new TypeError("Failed to fetch"));
    const images = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg"];
    const held = images.map(contribution);
    useStore.setState({ heldContributions: held, toasts: [] });

    postHeldContributions();
    for (let i = 0; i < images.length; i++) await flush();

    expect(useStore.getState().heldContributions).toEqual(held);
    expect(useStore.getState().recordedSession).toBeNull();
    expect(useStore.getState().toasts.map(textOf)).toEqual([
      "Could not confirm these visits were recorded; they are sent again with the next visit: " +
        images.join(", "),
    ]);
  });

  it("forgets the recorded session a refused visit named, so the next visit opens a fresh one", async () => {
    const post = vi
      .spyOn(sessionsApi, "imageEvent")
      .mockRejectedValueOnce(await thrown(ENDED_NAMED))
      .mockResolvedValueOnce(HELD);
    useStore.setState({
      user: "jordan",
      recordedSession: { project_id: TEST_PROJECT.id, started: S1, user: "jordan" },
      heldContributions: [{ ...contribution("a.jpg"), started: S1 }],
    });
    postHeldContributions();
    await flush();

    const s = useStore.getState();
    s.startImageSessionTracking("b.jpg", Date.now() - 2000);
    s.closeSessionInterval();
    postHeldContributions();
    await flush();

    expect(post.mock.calls[1][0]).toMatchObject({ image_name: "b.jpg", started: null });
    expect(useStore.getState().recordedSession).toMatchObject({ started: S2 });
  });

  it("lets the late answer to a contribution its departure retired name no session once its project is open again", async () => {
    const pending = Promise.withResolvers<SessionWrite>();
    vi.spyOn(sessionsApi, "imageEvent").mockReturnValueOnce(pending.promise);
    useStore.setState({ heldContributions: [contribution("a.jpg")] });
    postHeldContributions();
    await flush();

    useStore.getState().mergeSnapshot(naming(OTHER));
    useStore.getState().mergeSnapshot(naming(TEST_PROJECT));
    pending.resolve(FIRST);
    await flush();

    expect(useStore.getState().recordedSession).toBeNull();
  });

  it("holds a negative confirmed while the visit's clock is stopped", () => {
    const s = useStore.getState();
    s.startImageSessionTracking("a.jpg", Date.now() - 2000);
    s.pauseSessionInterval();
    useStore.setState({ heldContributions: [] });
    s.markNegativeConfirmed();

    s.closeSessionInterval();

    expect(useStore.getState().heldContributions).toEqual([
      expect.objectContaining({
        image_name: "a.jpg",
        seconds: 0,
        annotations_added: 0,
        activity: "negative_confirmation",
      }),
    ]);
  });
});
