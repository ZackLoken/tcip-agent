import { useEffect, useRef } from "react";

import { classesApi } from "@/api/classes";
import { sessionsApi } from "@/api/sessions";
import { DatasetPicker } from "@/components/DatasetPicker";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { HelpOverlay } from "@/components/HelpOverlay";
import { StatusBar } from "@/components/StatusBar";
import { TopBar } from "@/components/TopBar";
import { stateSocket } from "@/api/ws";
import { useStore } from "@/store";
import { AnnotateTab } from "@/tabs/AnnotateTab";
import { InferenceTab } from "@/tabs/InferenceTab";
import { MetaTab } from "@/tabs/MetaTab";
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
  const endedSessionForRoot = useRef<string | null>(null);

  useEffect(() => {
    stateSocket.connect();
    return () => stateSocket.close();
  }, []);

  // Session start / end.  Browser identifier is "web" since we don't have
  // OS user; the backend can rewrite this from a header later if needed.
  useEffect(() => {
    if (!projectRoot) return;
    const user = localStorage.getItem("tcip.user") || "web";
    void sessionsApi.start(projectRoot, user);
    endedSessionForRoot.current = null;

    function endSession() {
      if (endedSessionForRoot.current === projectRoot) return;
      endedSessionForRoot.current = projectRoot;

      const payload = JSON.stringify({ project_root: projectRoot });
      try {
        if (navigator.sendBeacon) {
          const blob = new Blob([payload], { type: "application/json" });
          navigator.sendBeacon("/api/sessions/end", blob);
          return;
        }
      } catch {
        // Fall through to fetch keepalive.
      }

      try {
        void fetch("/api/sessions/end", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
          keepalive: true,
        });
      } catch {
        // Best-effort telemetry only.
      }
    }

    window.addEventListener("pagehide", endSession);
    window.addEventListener("beforeunload", endSession);
    return () => {
      window.removeEventListener("pagehide", endSession);
      window.removeEventListener("beforeunload", endSession);
      endSession();
    };
  }, [projectRoot]);

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
      {/* Only Annotate / Review / Results need an imagery dataset+date; the rest
          (Training / Tuning / Inference / Meta) are reachable without one — being
          forced to pick a dataset just to read agent reports or watch a run was a
          usability trap. Dataset-dependent tabs show the picker until one is set. */}
      <ErrorBoundary resetKey={activeTab}>
        {activeTab === "annotate" && (datasetReady ? <AnnotateTab /> : <DatasetPicker />)}
        {activeTab === "review" && (datasetReady ? <ReviewTab /> : <DatasetPicker />)}
        {activeTab === "results" && (datasetReady ? <ResultsTab /> : <DatasetPicker />)}
        {activeTab === "training" && <TrainingTab />}
        {activeTab === "tuning" && <TuningTab />}
        {activeTab === "inference" && <InferenceTab />}
        {activeTab === "meta" && <MetaTab />}
      </ErrorBoundary>
      <StatusBar />
      <HelpOverlay activeTab={activeTab} />
    </div>
  );
}

export default App;
