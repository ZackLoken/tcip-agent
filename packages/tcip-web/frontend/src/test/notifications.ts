import { screen, within } from "@testing-library/react";

import { useStore } from "@/store";
import { type ConditionName, type NotificationEntry, textOf } from "@/store/slices/notifications";

/** Every listed entry whose text is `text`, oldest first. */
export function entriesSaying(text: string): NotificationEntry[] {
  return useStore.getState().notifications.filter((e) => textOf(e) === text);
}

/** The rendered notifications area, queried within. */
export const notificationsArea = () =>
  within(screen.getByRole("region", { name: "Notifications" }));

/** Every entry the rendered notifications area lists, newest first. */
export const areaEntries = (): HTMLElement[] => notificationsArea().queryAllByRole("listitem");

/** The one entry the rendered notifications area lists; throws when it lists another number. */
export function onlyAreaEntry(): HTMLElement {
  const items = areaEntries();
  if (items.length !== 1) throw new Error(`the area lists ${items.length} entries, not one`);
  return items[0];
}

/** Every entry the rendered notifications area lists whose text holds `text`, newest first. */
export const areaEntriesSaying = (text: string): HTMLElement[] =>
  areaEntries().filter((item) => item.textContent?.includes(text));

/** The one entry the rendered notifications area lists whose text holds `text`. */
export function areaEntrySaying(text: string): HTMLElement {
  const items = areaEntriesSaying(text);
  if (items.length !== 1) throw new Error(`the area lists ${items.length} entries saying ${text}`);
  return items[0];
}

/** The line the notifications area shows under the entry `item`'s text, as the person reads it:
 *  its level, kind, target, time and what follows the time. */
export function metaLine(item: HTMLElement): string {
  const line = item.querySelector("time")?.parentElement;
  if (!line) throw new Error("the item shows no time, so it is no entry of the area");
  return line.textContent ?? "";
}

/** Every text the live region `region` has been given since it began being watched: each node
 *  added to it, and each text rewritten in place. */
export function announcements(region: HTMLElement): string[] {
  const told: string[] = [];
  new MutationObserver((records) => {
    for (const record of records) {
      if (record.type === "characterData") told.push(record.target.textContent ?? "");
      record.addedNodes.forEach((node) => told.push(node.textContent ?? ""));
    }
  }).observe(region, { childList: true, subtree: true, characterData: true });
  return told;
}

/** The sentence of the listed entry `condition`'s current occurrence raised, null while the
 *  condition does not hold or its entry was dismissed. */
export function holdingSentence(condition: ConditionName): string | null {
  const { holding, notifications } = useStore.getState();
  return notifications.find((e) => e.id === holding[condition])?.sentence ?? null;
}
