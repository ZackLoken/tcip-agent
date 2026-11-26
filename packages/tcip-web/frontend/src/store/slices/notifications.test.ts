import { beforeEach, describe, expect, it, vi } from "vitest";

import { useStore } from "@/store";

const initialStoreState = useStore.getState();

beforeEach(() => {
  useStore.setState(initialStoreState, true);
});

/** Raise a notice on `key` whose dismissal runs `onDismiss`; return the id of its entry. */
function raiseDismissable(key: string, onDismiss: () => void): number {
  useStore.getState().raiseNotification({
    kind: "notice",
    level: "info",
    text: { sentence: `A note on ${key}.` },
    context: { project: "valley" },
    key,
    onDismiss,
  });
  const entry = useStore.getState().notifications.find((e) => e.key === key);
  if (!entry) throw new Error(`no entry stands on ${key}`);
  return entry.id;
}

describe("an entry's dismissal", () => {
  it("runs the function its raise gave once, after the entry leaves the list, and not once its key is released", () => {
    const listedWhenRun: number[] = [];
    const onDismiss = vi.fn(() => listedWhenRun.push(useStore.getState().notifications.length));
    const kept = vi.fn();
    const id = raiseDismissable("note:first", onDismiss);
    const releasedId = raiseDismissable("note:second", kept);

    useStore.getState().dismissNotification(id);
    useStore.getState().dismissNotification(id);
    useStore.getState().releaseNotification("note:second");
    useStore.getState().dismissNotification(releasedId);

    expect(onDismiss).toHaveBeenCalledTimes(1);
    expect(listedWhenRun).toEqual([1]);
    expect(kept).not.toHaveBeenCalled();
    expect(useStore.getState().notifications).toEqual([]);
  });
});
