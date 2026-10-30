import { describe, expect, it } from "vitest";

import { jumpTarget, stepTarget } from "@/hooks/useImageNav";
import { filteredIndices, rankedNames } from "@/lib/imageFilter";
import { ANNOTATE_ANSWERS } from "@/test/annotate";
import { QUEUE_ANSWERS } from "@/test/queue";
import { RAW_SELECTION } from "@/test/store";

/** The capture the produced queue ranked, as the select route served it. */
const QUEUE_SELECTION = QUEUE_ANSWERS.selection.selection;

describe("stepTarget (shared by arrows + Prev/Next)", () => {
  it("steps by delta and clamps at the ends", () => {
    const all = [0, 1, 2, 3];
    expect(stepTarget(all, 0, 1)).toBe(1);
    expect(stepTarget(all, 2, -1)).toBe(1);
    expect(stepTarget(all, 3, 1)).toBeNull();
    expect(stepTarget(all, 0, -1)).toBeNull();
    expect(stepTarget([], 0, 1)).toBeNull();
  });

  it("steps along the kept images only, and from a dropped one to the nearest kept", () => {
    const kept = [1, 4];
    expect(stepTarget(kept, 1, 1)).toBe(4);
    expect(stepTarget(kept, 4, -1)).toBe(1);
    expect(stepTarget(kept, 2, 1)).toBe(4);
    expect(stepTarget(kept, 2, -1)).toBe(1);
    expect(stepTarget(kept, 5, 1)).toBeNull();
    expect(stepTarget(kept, 0, -1)).toBeNull();
  });

  it("from an image off a rank-ordered list, steps to the first entry lying that way", () => {
    // The produced queue skips the finished second photograph and ranks the fifth first.
    const names = QUEUE_SELECTION.images.map(({ name }) => name);
    const ranked = QUEUE_ANSWERS.completed.ranked?.queue.map((e) => e.image) ?? [];
    const order = filteredIndices(QUEUE_SELECTION.images, {}, "queue", ranked);
    const off = names.findIndex((name) => !ranked.includes(name));
    expect(off).toBe(1);
    expect(names[stepTarget(order, off, 1)!]).toBe(ranked[0]);
    expect(names[stepTarget(order, off, -1)!]).toBe(names[0]);
  });

  it("steps along a rank-ordered list in its own order, whatever the indices' order", () => {
    // The produced queue's images in rank order, mapped to the selection's indices by name.
    const ranked = QUEUE_ANSWERS.completed.ranked?.queue.map((e) => e.image) ?? [];
    const order = filteredIndices(QUEUE_SELECTION.images, {}, "queue", ranked);
    expect(order).not.toEqual([...order].sort((a, b) => a - b));
    expect(order.map((i) => QUEUE_SELECTION.images[i].name)).toEqual(ranked);

    const walked: number[] = [];
    for (let at = stepTarget(order, -1, 1); at !== null; at = stepTarget(order, at, 1)) {
      walked.push(at);
    }
    expect(walked).toEqual(order);
  });
});

describe("jumpTarget (counter box)", () => {
  it("maps a 1-based position among the kept images to an index, clamped", () => {
    expect(jumpTarget([0, 1, 2, 3], 2)).toBe(1);
    expect(jumpTarget([0, 1, 2, 3], 99)).toBe(3);
    expect(jumpTarget([0, 1, 2, 3], 0)).toBe(0);
    expect(jumpTarget([2, 5], 2)).toBe(5);
    expect(jumpTarget([], 1)).toBeNull();
  });
});

describe("filteredIndices", () => {
  it("keeps the images whose state the filter names, an unknown state only under all", () => {
    const list = RAW_SELECTION.images;
    // The produced states: the first negative, the second partial, the fifth complete.
    const { states } = ANNOTATE_ANSWERS.marked_states;
    // The fourth photograph's state left unstated, as one the answer does not know.
    const stated = Object.fromEntries(
      Object.entries(states).filter(([name]) => name !== list[3].name),
    );
    expect(filteredIndices(list, stated, "all", [])).toEqual([0, 1, 2, 3, 4]);
    expect(filteredIndices(list, stated, "unannotated", [])).toEqual([2]);
    expect(filteredIndices(list, stated, "in_progress", [])).toEqual([1]);
    expect(filteredIndices(list, stated, "complete", [])).toEqual([0, 4]);
  });
});

describe("rankedNames", () => {
  it("names the completed queue's images for the capture it ranked, and none for another", () => {
    const { completed, running } = QUEUE_ANSWERS;
    const ranked = completed.ranked?.queue.map((e) => e.image);
    expect(rankedNames(completed, QUEUE_SELECTION)).toEqual(ranked);
    expect(rankedNames(completed, QUEUE_ANSWERS.later_selection.selection)).toBeNull();
    expect(rankedNames(running, QUEUE_SELECTION)).toBeNull();
    expect(rankedNames(null, QUEUE_SELECTION)).toBeNull();
  });
});
