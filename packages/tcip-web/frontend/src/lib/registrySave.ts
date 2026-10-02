import { committedOf } from "@/api/http";
import { subjectsApi, type Registry } from "@/api/subjects";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";

/** Grow the open dataset's subject registry to `next`: adopted locally at once, saved against the
 *  version this browser holds, and reloaded from the server when the save is refused. Answers
 *  whether the save landed; a refusal toasts `failure` with the server's reason. */
export async function saveRegistry(next: Registry, failure: string): Promise<boolean> {
  const { registry, setRegistry, pushToast, gui } = useStore.getState();
  const root = gui.dataset.dataset_root;
  setRegistry(next, registry.version);
  if (!selectProjectRoot(useStore.getState()) || !root) return false;
  try {
    const saved = await subjectsApi.save(next, root, registry.version);
    setRegistry(next, saved.version);
    return true;
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    const saved = committedOf<{ version: string }>(e);
    if (saved) {
      setRegistry(next, saved.version);
      pushToast(message);
      return true;
    }
    pushToast(`${failure}: ${message}`);
    try {
      const fresh = await subjectsApi.load(root, gui.dataset.annotations_dir);
      setRegistry(fresh.subjects, fresh.version);
    } catch {
      /* the reload itself failing leaves the optimistic registry in place */
    }
    return false;
  }
}
