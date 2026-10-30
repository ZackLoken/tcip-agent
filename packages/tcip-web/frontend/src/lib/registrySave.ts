import { committedOf, messageOf } from "@/api/http";
import { subjectsApi } from "@/api/subjects";
import type { RegistrySave, RegistrySubjects } from "@/api/types.generated";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";

/** Grow the open dataset's subject registry to `next`: adopted locally at once, saved against the
 *  version this browser holds, and reloaded from the server when the save is refused. Answers
 *  whether the save landed; a refusal toasts `failure` with the server's reason. */
export async function saveRegistry(next: RegistrySubjects, failure: string): Promise<boolean> {
  const { registry, setRegistry, pushToast, gui, user } = useStore.getState();
  setRegistry(next, registry.version, registry.discovered);
  if (!selectProjectRoot(useStore.getState()) || !gui.dataset) return false;
  const { dataset_root: root, date } = gui.dataset;
  try {
    const saved = await subjectsApi.save({
      subjects: next,
      dataset_root: root,
      version: registry.version,
      user,
    });
    setRegistry(next, saved.version, registry.discovered);
    return true;
  } catch (e) {
    const message = messageOf(e);
    const saved = committedOf<RegistrySave>(e);
    if (saved) {
      setRegistry(next, saved.version, registry.discovered);
      pushToast(message);
      return true;
    }
    pushToast(`${failure}: ${message}`);
    try {
      const fresh = await subjectsApi.load(root, date);
      setRegistry(fresh.subjects ?? {}, fresh.version, fresh.discovered);
    } catch {
      /* the reload itself failing leaves the optimistic registry in place */
    }
    return false;
  }
}
