import { useEffect } from "react";

import { classesApi } from "@/api/classes";
import { DatasetPicker } from "@/components/DatasetPicker";
import { HelpOverlay } from "@/components/HelpOverlay";
import { StatusBar } from "@/components/StatusBar";
import { TopBar } from "@/components/TopBar";
import { stateSocket } from "@/api/ws";
import { useStore } from "@/store";
import { AnnotateTab } from "@/tabs/AnnotateTab";
import { InferenceTab } from "@/tabs/InferenceTab";
import { ResultsTab } from "@/tabs/ResultsTab";
import { ReviewTab } from "@/tabs/ReviewTab";
import { TrainingTab } from "@/tabs/TrainingTab";
import { TuningTab } from "@/tabs/TuningTab";

function App() {
  const activeTab = useStore((s) => s.gui.active_tab);
  const datasetReady = useStore((s) => !!s.gui.dataset.dataset_root && !!s.gui.dataset.date);
  const projectRoot = useStore((s) => s.gui.dataset.project_root);
  const datasetKey = useStore(
    (s) =>
      `${s.gui.dataset.dataset_root ?? ""}::${s.gui.dataset.annotation_type ?? ""}::${s.gui.dataset.date ?? ""}`,
  );
  const imageList = useStore((s) => s.gui.dataset.image_list);
  const annDetectDir = useStore((s) => s.gui.dataset.annotations_detect_dir);
  const annSegDir = useStore((s) => s.gui.dataset.annotations_segment_dir);
  const setClasses = useStore((s) => s.setClasses);
  const setImageStatuses = useStore((s) => s.setImageStatuses);

  useEffect(() => {
    stateSocket.connect();
    return () => stateSocket.close();
  }, []);

  // Hydrate classes + per-image status whenever the dataset selection changes.
  useEffect(() => {
    if (!projectRoot || imageList.length === 0) return;
    void (async () => {
      try {
        const reg = await classesApi.load(projectRoot);
        setClasses(reg.classes);

        const saved = await classesApi.loadImageStatus(projectRoot);
        const savedMap = saved.statuses ?? {};
        // For any image not yet in savedMap, derive status from label files.
        const missing = imageList.filter((name) => !(name in savedMap));
        let derived: Record<string, string> = {};
        if (missing.length) {
          const derivedRes = await classesApi.deriveImageStatus({
            project_root: projectRoot,
            annotations_detect_dir: annDetectDir,
            annotations_segment_dir: annSegDir,
            image_list: missing,
          });
          derived = derivedRes.statuses ?? {};
          if (Object.keys(derived).length) {
            await classesApi.setImageStatusBulk(
              projectRoot,
              derived as Record<string, "complete" | "partial" | "unannotated">,
            );
          }
        }
        setImageStatuses({
          ...savedMap,
          ...(derived as Record<string, "complete" | "partial" | "unannotated">),
        });
      } catch (err) {
        console.warn("class / image-status hydrate failed", err);
      }
    })();
  }, [projectRoot, datasetKey, imageList, annDetectDir, annSegDir, setClasses, setImageStatuses]);

  return (
    <div className="h-full flex flex-col bg-tcip-bg text-tcip-fg">
      <TopBar />
      {datasetReady ? (
        <>
          {activeTab === "annotate" && <AnnotateTab />}
          {activeTab === "review" && <ReviewTab />}
          {activeTab === "training" && <TrainingTab />}
          {activeTab === "tuning" && <TuningTab />}
          {activeTab === "inference" && <InferenceTab />}
          {activeTab === "results" && <ResultsTab />}
        </>
      ) : (
        <DatasetPicker />
      )}
      <StatusBar />
      <HelpOverlay activeTab={activeTab} />
    </div>
  );
}

export default App;
