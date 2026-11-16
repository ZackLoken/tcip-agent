import { useStore } from "@/store";
import { type ConditionName, type NotificationEntry, textOf } from "@/store/slices/notifications";

/** Every listed entry whose text is `text`, oldest first. */
export function entriesSaying(text: string): NotificationEntry[] {
  return useStore.getState().notifications.filter((e) => textOf(e) === text);
}

/** The sentence of the listed entry `condition`'s current occurrence raised, null while the
 *  condition does not hold or its entry was dismissed. */
export function holdingSentence(condition: ConditionName): string | null {
  const { holding, notifications } = useStore.getState();
  return notifications.find((e) => e.id === holding[condition])?.sentence ?? null;
}
