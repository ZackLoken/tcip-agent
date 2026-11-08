import { committedOf, messageOf } from "@/api/http";
import { subjectsApi } from "@/api/subjects";
import {
  type RegistrySave,
  type RegistrySubjects,
  STORED,
  type SubjectsLoad,
} from "@/api/types.generated";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";

/** Hold the subject registry the load answer `loaded` found: a stored one's subjects at its
 *  version, or none declared over no registry. */
export function adoptLoad(loaded: SubjectsLoad): void {
  const { setRegistry } = useStore.getState();
  if (loaded.registry.kind === STORED) {
    const { subjects, ...version } = loaded.registry;
    setRegistry(subjects, version, loaded.discovered);
  } else setRegistry({}, loaded.registry, loaded.discovered);
}

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
    setRegistry(next, { kind: STORED, version: saved.version }, registry.discovered);
    return true;
  } catch (e) {
    const message = messageOf(e);
    const gap = committedOf<RegistrySave>(e);
    if (gap?.committed && gap.what !== null) {
      setRegistry(next, { kind: STORED, version: gap.what.version }, registry.discovered);
      pushToast(message);
      return true;
    }
    pushToast(`${failure}: ${message}`);
    try {
      adoptLoad(await subjectsApi.load(root, date));
    } catch {
      /* the reload itself failing leaves the optimistic registry in place */
    }
    return false;
  }
}
