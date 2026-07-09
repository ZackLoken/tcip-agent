/**
 * Shown when no dataset is loaded. A browser can't open a native folder dialog, so the
 * user types absolute paths and the backend validates them. The UI makes that explicit:
 * live scan feedback on the dataset root, an image count for the chosen date, and a
 * button whose label always says what's still needed.
 */

import { useEffect, useState } from "react";

import { api } from "@/api/client";
import { FolderBrowser } from "@/components/FolderBrowser";
import { cleanPath } from "@/lib/paths";
import { useStore } from "@/store";

interface Tree {
  dates_with_images: string[];
  annotation_types: string[];
  model_names: string[];
}

function friendlyScanError(msg: string): string {
  if (/40[34]/.test(msg)) {
    return "Folder not found. Check the path is exact — in File Explorer, click the address bar and copy the full path.";
  }
  return msg;
}

export function DatasetPicker() {
  const [projectRoot, setProjectRoot] = useState(
    () => localStorage.getItem("tcip.project_root") ?? "",
  );
  const [datasetRoot, setDatasetRoot] = useState(
    () => localStorage.getItem("tcip.dataset_root") ?? "",
  );
  const [date, setDate] = useState(() => localStorage.getItem("tcip.date") ?? "");
  const [annotationType, setAnnotationType] = useState(
    () => localStorage.getItem("tcip.annotation_type") ?? "",
  );
  const [modelName, setModelName] = useState(() => localStorage.getItem("tcip.model_name") ?? "");

  const [tree, setTree] = useState<Tree | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanError, setScanError] = useState<string | null>(null);
  const [imageCount, setImageCount] = useState<number | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [browsing, setBrowsing] = useState<"project" | "dataset" | null>(null);
  const patchGui = useStore((s) => s.patchGui);

  // Scan the dataset root whenever it changes → populates the dropdowns.
  useEffect(() => {
    if (!datasetRoot) {
      setTree(null);
      setScanError(null);
      return;
    }
    let cancelled = false;
    setScanning(true);
    setScanError(null);
    api.dataset
      .tree(datasetRoot)
      .then((t) => {
        if (cancelled) return;
        setTree(t);
      })
      .catch((e) => {
        if (cancelled) return;
        setTree(null);
        setScanError(friendlyScanError(String(e)));
      })
      .finally(() => {
        if (!cancelled) setScanning(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetRoot]);

  // How many images the chosen date holds (confirms the session will have something to show).
  useEffect(() => {
    if (!datasetRoot || !date) {
      setImageCount(null);
      return;
    }
    let cancelled = false;
    api.dataset
      .listImages(datasetRoot, date)
      .then((r) => {
        if (!cancelled) setImageCount(r.count);
      })
      .catch(() => {
        if (!cancelled) setImageCount(null);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetRoot, date]);

  const canOpen = Boolean(projectRoot && datasetRoot && tree && date && !submitting);
  const buttonLabel = submitting
    ? "Opening…"
    : !projectRoot || !datasetRoot
      ? "Enter both folder paths"
      : scanning
        ? "Scanning…"
        : !tree
          ? "Fix the dataset root path"
          : !date
            ? "Choose a date"
            : "Open dataset";

  async function submit() {
    if (!canOpen) return;
    setSubmitting(true);
    setSubmitError(null);
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
    } catch (e) {
      setSubmitError(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  const noImagesHere = tree !== null && tree.dates_with_images.length === 0;

  return (
    <div className="h-full w-full flex items-center justify-center bg-tcip-bg p-4">
      <div className="tcip-panel rounded-lg w-full max-w-xl p-6 flex flex-col gap-4">
        <div className="text-lg font-semibold">Open a dataset</div>

        {/* Paths */}
        <div className="flex flex-col gap-1">
          <label
            className="text-[11px] font-medium"
            title="Folder that holds .tcip\ — often the same as the dataset root"
          >
            Project root <span className="text-tcip-fp">*</span>
          </label>
          <div className="flex gap-2">
            <input
              className="tcip-input flex-1"
              value={projectRoot}
              onChange={(e) => setProjectRoot(cleanPath(e.target.value))}
              placeholder="e.g. C:\path\to\project"
              spellCheck={false}
            />
            <button
              className="tcip-btn text-[11px] whitespace-nowrap"
              onClick={() => setBrowsing("project")}
            >
              Browse…
            </button>
          </div>
        </div>

        <div className="flex flex-col gap-1">
          <label
            className="text-[11px] font-medium"
            title="Folder containing images/, annotations/, models/"
          >
            Dataset root <span className="text-tcip-fp">*</span>
          </label>
          <div className="flex gap-2">
            <input
              className="tcip-input flex-1"
              value={datasetRoot}
              onChange={(e) => setDatasetRoot(cleanPath(e.target.value))}
              placeholder="e.g. C:\path\to\Valley_Farm"
              spellCheck={false}
            />
            <button
              className="tcip-btn text-[11px] whitespace-nowrap"
              onClick={() => setBrowsing("dataset")}
            >
              Browse…
            </button>
          </div>
          {datasetRoot && scanning && (
            <span className="text-[11px] text-tcip-muted">Scanning…</span>
          )}
          {datasetRoot && !scanning && scanError && (
            <span className="text-[11px] text-tcip-fp">{scanError}</span>
          )}
          {datasetRoot && !scanning && tree && !noImagesHere && (
            <span className="text-[11px] text-tcip-tp">
              ✓ {tree.dates_with_images.length} date(s), {tree.annotation_types.length} type(s),{" "}
              {tree.model_names.length} model(s)
            </span>
          )}
          {datasetRoot && !scanning && noImagesHere && (
            <span className="text-[11px] text-tcip-fp">
              No <code className="font-mono">images/</code> folders here — wrong dataset root?
            </span>
          )}
        </div>

        {/* Selection */}
        <div className="grid grid-cols-3 gap-2">
          <div className="flex flex-col gap-1">
            <label className="text-[11px] font-medium">
              Date <span className="text-tcip-fp">*</span>
            </label>
            <select
              className="tcip-select"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              disabled={!tree}
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
            <label className="text-[11px] font-medium" title="Needed to view/edit labels">
              Annotation type
            </label>
            <select
              className="tcip-select"
              value={annotationType}
              onChange={(e) => setAnnotationType(e.target.value)}
              disabled={!tree}
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
            <label className="text-[11px] font-medium" title="Needed to review predictions">
              Model
            </label>
            <select
              className="tcip-select"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              disabled={!tree}
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
        {date && imageCount !== null && (
          <span className="text-[11px] text-tcip-muted -mt-2">
            {date}: {imageCount} image(s)
          </span>
        )}

        {submitError && <div className="text-[12px] text-tcip-fp">{submitError}</div>}

        <button className="tcip-btn-primary mt-1" onClick={submit} disabled={!canOpen}>
          {buttonLabel}
        </button>
      </div>

      {browsing && (
        <FolderBrowser
          title={browsing === "project" ? "Select project root" : "Select dataset root"}
          initialPath={browsing === "project" ? projectRoot : datasetRoot}
          onSelect={(p) => {
            if (browsing === "project") setProjectRoot(p);
            else setDatasetRoot(p);
            setBrowsing(null);
          }}
          onClose={() => setBrowsing(null)}
        />
      )}
    </div>
  );
}
