/**
 * Modal-ish dataset picker shown when no dataset is loaded.
 * Browser asks the user for paths; backend validates and populates state.
 */

import { useEffect, useState } from "react";

import { api } from "@/api/client";
import { useStore } from "@/store";

export function DatasetPicker() {
  const [projectRoot, setProjectRoot] = useState(() =>
    localStorage.getItem("tcip.project_root") ?? "",
  );
  const [datasetRoot, setDatasetRoot] = useState(() =>
    localStorage.getItem("tcip.dataset_root") ?? "",
  );
  const [date, setDate] = useState(() => localStorage.getItem("tcip.date") ?? "");
  const [annotationType, setAnnotationType] = useState(() =>
    localStorage.getItem("tcip.annotation_type") ?? "",
  );
  const [modelName, setModelName] = useState(() => localStorage.getItem("tcip.model_name") ?? "");

  const [tree, setTree] = useState<{
    dates_with_images: string[];
    annotation_types: string[];
    model_names: string[];
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const patchGui = useStore((s) => s.patchGui);

  useEffect(() => {
    if (!datasetRoot) {
      setTree(null);
      return;
    }
    setLoading(true);
    api.dataset
      .tree(datasetRoot)
      .then((t) => {
        setTree({
          dates_with_images: t.dates_with_images,
          annotation_types: t.annotation_types,
          model_names: t.model_names,
        });
        setError(null);
      })
      .catch((e) => {
        setTree(null);
        setError(String(e));
      })
      .finally(() => setLoading(false));
  }, [datasetRoot]);

  async function submit() {
    if (!projectRoot || !datasetRoot) {
      setError("Project root and dataset root are required.");
      return;
    }
    setLoading(true);
    try {
      const res = await api.dataset.select({
        project_root: projectRoot,
        dataset_root: datasetRoot,
        annotation_type: annotationType || null,
        date: date || null,
        model_name: modelName || null,
      });
      patchGui({ dataset: res.selection });
      localStorage.setItem("tcip.project_root", projectRoot);
      localStorage.setItem("tcip.dataset_root", datasetRoot);
      localStorage.setItem("tcip.date", date);
      localStorage.setItem("tcip.annotation_type", annotationType);
      localStorage.setItem("tcip.model_name", modelName);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-full w-full flex items-center justify-center bg-tcip-bg">
      <div className="tcip-panel w-full max-w-lg p-6 flex flex-col gap-3">
        <div className="text-lg font-semibold">Open a dataset</div>
        <div className="text-[11px] text-tcip-muted">
          Enter absolute paths. Project root is where <code className="font-mono">.tcip/</code>
          lives; dataset root is the folder containing <code className="font-mono">images/</code>,
          <code className="font-mono"> annotations/</code>, <code className="font-mono">models/</code>.
        </div>

        <label className="text-[11px] text-tcip-muted">Project root</label>
        <input
          className="tcip-input"
          value={projectRoot}
          onChange={(e) => setProjectRoot(e.target.value)}
          placeholder="C:\Users\exx\Documents\GitHub\tcip-agent"
        />

        <label className="text-[11px] text-tcip-muted">Dataset root</label>
        <input
          className="tcip-input"
          value={datasetRoot}
          onChange={(e) => setDatasetRoot(e.target.value)}
          placeholder="…\Valley_Farm"
        />

        <div className="grid grid-cols-3 gap-2">
          <div className="flex flex-col gap-1">
            <label className="text-[11px] text-tcip-muted">Annotation type</label>
            <select
              className="tcip-select"
              value={annotationType}
              onChange={(e) => setAnnotationType(e.target.value)}
            >
              <option value="">—</option>
              {tree?.annotation_types.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-[11px] text-tcip-muted">Date</label>
            <select
              className="tcip-select"
              value={date}
              onChange={(e) => setDate(e.target.value)}
            >
              <option value="">—</option>
              {tree?.dates_with_images.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>

          <div className="flex flex-col gap-1">
            <label className="text-[11px] text-tcip-muted">Model (for preds)</label>
            <select
              className="tcip-select"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
            >
              <option value="">—</option>
              {tree?.model_names.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>

        {error && <div className="text-[12px] text-tcip-fp">{error}</div>}

        <button className="tcip-btn-primary mt-2" onClick={submit} disabled={loading}>
          {loading ? "Loading…" : "Open dataset"}
        </button>
      </div>
    </div>
  );
}
