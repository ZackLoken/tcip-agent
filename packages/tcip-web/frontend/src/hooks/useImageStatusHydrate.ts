import { useEffect } from "react";

import { classesApi } from "@/api/classes";
import { reconcileImageStatuses } from "@/lib/imageStatus";
import { useStore } from "@/store";

export interface ImageStatusHydrateParams {
  projectRoot: string | null;
  subject: string | null;
  datasetRoot: string | null;
  datasetDate: string | null;
  annotationsDir: string | null;
  imageList: string[];
}

/**
 * Loads the stored per-image status for the selected subject and reconciles it against what the
 * label files derive to now (`reconcileImageStatuses`). An unconfirmed name heals freely and is
 * written back through the bulk route; a confirmed name (complete/negative) whose derived token
 * now disagrees is never rewritten, only added to `staleMarks` for the breeder to re-confirm.
 *
 * Runs once per dataset selection (this hook's own dependencies): a label file edited afterward,
 * in this session or by the agent, is not checked again until the next selection of this dataset.
 *
 * Skips entirely with no subject selected: nothing to scope image status to yet. Either way,
 * `staleMarks` is cleared first, so a mark left over from a previously selected dataset can never
 * be read against a same-named image in this one.
 */
export function useImageStatusHydrate({
  projectRoot,
  subject,
  datasetRoot,
  datasetDate,
  annotationsDir,
  imageList,
}: ImageStatusHydrateParams): void {
  useEffect(() => {
    useStore.getState().clearStaleMarks();
    if (!projectRoot || !subject || imageList.length === 0) return;
    let cancelled = false;
    void (async () => {
      try {
        const saved = await classesApi.loadImageStatus(
          projectRoot,
          subject,
          datasetDate,
          datasetRoot,
          annotationsDir,
        );
        const stored = saved.statuses ?? {};
        const confirmed = imageList.filter(
          (name) => stored[name] === "complete" || stored[name] === "negative",
        );
        const derivedRes = await classesApi.deriveImageStatus({
          project_root: projectRoot,
          annotations_dir: annotationsDir,
          subject,
          image_list: imageList,
          complete_override: confirmed,
        });
        const derived = derivedRes.statuses ?? {};
        const { writes, staleMarks } = reconcileImageStatuses(stored, derived, confirmed);
        if (cancelled) return;
        if (derivedRes.unreadable.length) {
          useStore
            .getState()
            .pushToast(
              `${derivedRes.unreadable.length} label file(s) could not be read: ${derivedRes.unreadable.join(", ")}`,
            );
        }
        if (Object.keys(writes).length) {
          await classesApi.setImageStatusBulk(
            projectRoot,
            writes,
            subject,
            datasetDate,
            datasetRoot,
            annotationsDir,
            useStore.getState().user || undefined,
          );
        }
        if (cancelled) return;
        useStore.getState().setImageStatuses({ ...stored, ...writes }, staleMarks);
        if (staleMarks.length) {
          useStore
            .getState()
            .pushToast(
              `${staleMarks.length} image(s) need re-confirmation for ${subject}.`,
              "info",
            );
        }
      } catch (err) {
        if (cancelled) return;
        console.warn("image-status hydrate failed", err);
        useStore.getState().pushToast("Could not load the image status for this project.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [projectRoot, subject, datasetRoot, datasetDate, annotationsDir, imageList]);
}
