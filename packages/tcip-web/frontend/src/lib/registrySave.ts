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

/** The key a refused registry save made from `control` over the dataset at `datasetRoot` stands
 *  on, so one dataset's refusal never shows against another's. */
export const refusedRegistryKey = (
  control: "attributes" | "new subject",
  datasetRoot: string,
): string => `registry:${control}:${datasetRoot}`;

/** The id of the Annotate toolbar's subject control, where focus returns once a refused new
 *  subject is dismissed. */
export const SUBJECT_CONTROL_ID = "annotate-subject-control";

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
 *  whether the save landed. A refusal is a failure stating `failure` with the server's reason,
 *  standing on `key` until a later save through the same key lands, which resolves it; a save
 *  that landed without its audit line is a warning notice. */
export async function saveRegistry(
  next: RegistrySubjects,
  failure: string,
  key: string,
): Promise<boolean> {
  const { registry, setRegistry, gui, user, openProject } = useStore.getState();
  setRegistry(next, registry.version, registry.discovered);
  if (!selectProjectRoot(useStore.getState()) || !gui.dataset) return false;
  const { dataset_root: root, date } = gui.dataset;
  const context = { project: openProject?.id, target: date };
  const { raiseNotification, resolveNotification } = useStore.getState();
  try {
    const saved = await subjectsApi.save({
      subjects: next,
      dataset_root: root,
      version: registry.version,
      user,
    });
    setRegistry(next, { kind: STORED, version: saved.version }, registry.discovered);
    resolveNotification(key);
    return true;
  } catch (e) {
    const message = messageOf(e);
    const saved = committedOf<RegistrySave>(e);
    if (saved) {
      setRegistry(next, { kind: STORED, version: saved.version }, registry.discovered);
      resolveNotification(key);
      raiseNotification({ kind: "notice", level: "warning", text: { sentence: message }, context });
      return true;
    }
    raiseNotification({
      kind: "failure",
      level: "error",
      text: { sentence: `${failure}: ${message}` },
      context,
      key,
    });
    try {
      adoptLoad(await subjectsApi.load(root, date));
    } catch {
      /* the reload itself failing leaves the optimistic registry in place */
    }
    return false;
  }
}
