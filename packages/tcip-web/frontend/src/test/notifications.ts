import { useStore } from "@/store";
import type { ConditionName } from "@/store/slices/notifications";

/** The sentence of the listed entry `condition`'s current occurrence raised, null while the
 *  condition does not hold or its entry was dismissed. */
export function holdingSentence(condition: ConditionName): string | null {
  const { holding, notifications } = useStore.getState();
  return notifications.find((e) => e.id === holding[condition])?.sentence ?? null;
}
