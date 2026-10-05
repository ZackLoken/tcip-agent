import { committedOf } from "@/api/http";
import { subjectsApi, type Registry } from "@/api/subjects";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";

/** Grow the open dataset's subject registry to `next`: adopted locally at once, saved against the
 *  version this browser holds, and reloaded from the server when the save is refused. Answers
 *  whether the save landed; a refusal toasts `failure` with the server's reason. */
export async function saveRegistry(next: Registry, failure: string): Promise<boolean> {
  const { registry, setRegistry, pushToast, gui, user } = useStore.getState();
  const { dataset_root: root, date } = gui.dataset;
  setRegistry(next, registry.version, registry.discovered);
  if (!selectProjectRoot(useStore.getState()) || !root || !date) return false;
  try {
    const saved = await subjectsApi.save(next, root, registry.version, user);
    setRegistry(next, saved.version, registry.discovered);
    return true;
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    const saved = committedOf<{ version: string }>(e);
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
