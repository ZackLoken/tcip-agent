import { useEffect } from "react";

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

  useEffect(() => {
    stateSocket.connect();
    return () => stateSocket.close();
  }, []);

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
