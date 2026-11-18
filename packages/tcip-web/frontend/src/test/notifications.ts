import { useStore } from "@/store";
import { type ConditionName, type NotificationEntry, textOf } from "@/store/slices/notifications";

/** Every listed entry whose text is `text`, oldest first. */
export function entriesSaying(text: string): NotificationEntry[] {
  return useStore.getState().notifications.filter((e) => textOf(e) === text);
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
