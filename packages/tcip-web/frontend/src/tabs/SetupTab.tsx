import { useCallback, useEffect, useState } from "react";

import { committedOf, StructuredRefusalError } from "@/api/http";
import { resultsApi, type PlantMappingSummary } from "@/api/inference";
import type { MatchTolerance } from "@/api/types.generated";
import { TabHeading } from "@/components/TabHeading";
import { TraitRevisionPanel } from "@/components/TraitRevisionPanel";
import { useStore } from "@/store";

// resolve_nn_tolerance_m's own three sources, in the breeder's words; a source string this
// map does not know renders as its own raw string.
const TOLERANCE_SOURCE_PHRASES: Record<string, string> = {
  grid_pitch: "derived from the plot's grid pitch",
  stated: "the stated value",
  stated_capped: "the stated value, capped to the grid pitch",
};

function toleranceSourceText(source: string): string {
  return TOLERANCE_SOURCE_PHRASES[source] ?? source;
}

function PlantMappingPanel({ datasetRoot }: { datasetRoot: string | null }) {
  const [mappingName, setMappingName] = useState("");
  // Every mapping name already persisted under the open project, for the picker below; a name
  // typed here that isn't in the list is a new mapping this build will create.
  const [mappingNames, setMappingNames] = useState<string[]>([]);
  const [plantRegistry, setPlantRegistry] = useState("");
  const [nnTolerance, setNnTolerance] = useState<number | "">("");
  // Off by default: a rebuild a delivery event still cites answers 409 unless this is sent true.
  const [supersedeMapping, setSupersedeMapping] = useState(false);
  const [buildSummary, setBuildSummary] = useState<PlantMappingSummary | null>(null);
  const [buildTolerance, setBuildTolerance] = useState<MatchTolerance | null>(null);
  const [buildMaxMatchDistance, setBuildMaxMatchDistance] = useState<number | null>(null);
  const [buildMsg, setBuildMsg] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);

  const refreshMappingNames = useCallback(() => {
    void resultsApi
      .listPlantMappings()
      .then((res) => setMappingNames(res.names))
      .catch(() => setMappingNames([]));
  }, []);

  useEffect(() => {
    refreshMappingNames();
  }, [refreshMappingNames]);

  // Selecting an already-built mapping by name shows its own summary and tolerance, the same way
  // a fresh build does.
  async function loadMapping(name: string) {
    try {
      const res = await resultsApi.loadPlantMapping(name);
      setBuildSummary("per_date" in res.summary ? (res.summary as PlantMappingSummary) : null);
      setBuildTolerance(res.nn_tolerance_m);
      setBuildMaxMatchDistance(res.max_match_distance_m);
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Load mapping failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  async function buildMapping() {
    if (!datasetRoot) return;
    if (!mappingName) {
      setBuildMsg("Name the mapping before building.");
      return;
    }
    if (!plantRegistry) {
      setBuildMsg("Name a registered plant registry before building.");
      return;
    }
    setBuilding(true);
    setBuildMsg(null);
    setBuildSummary(null);
    setBuildTolerance(null);
    setBuildMaxMatchDistance(null);
    try {
      const res = await resultsApi.buildPlantMapping({
        name: mappingName,
        images_root: `${datasetRoot}/images`,
        plant_registry: plantRegistry,
        supersede: supersedeMapping,
        ...(nnTolerance === "" ? {} : { nn_tolerance_m: nnTolerance }),
      });
      setBuildSummary(res.summary);
      setBuildTolerance(res.nn_tolerance_m);
      setBuildMaxMatchDistance(res.max_match_distance_m);
      setBuildMsg(`Mapping built + saved as ${mappingName}`);
      refreshMappingNames();
    } catch (e) {
      const committed = committedOf<Awaited<ReturnType<typeof resultsApi.buildPlantMapping>>>(e);
      if (committed) {
        setBuildSummary(committed.summary);
        setBuildTolerance(committed.nn_tolerance_m);
        setBuildMaxMatchDistance(committed.max_match_distance_m);
        refreshMappingNames();
        setBuildMsg(
          `Mapping built + saved as ${mappingName}. ` +
            (e instanceof Error ? e.message : String(e)),
        );
        return;
      }
      if (e instanceof StructuredRefusalError && e.status === 409) {
        // A rebuild cited by a delivery event, or an audit-gap refusal with no mapping saved.
        setBuildMsg(e.message);
      } else {
        useStore
          .getState()
          .pushToast(`Build mapping failed: ${e instanceof Error ? e.message : String(e)}`);
        setBuildMsg(null);
      }
    } finally {
      setBuilding(false);
    }
  }

  return (
    <div className="tcip-panel p-4">
      <div className="tcip-heading mb-3">Plant mapping</div>
      <div className="grid grid-cols-[1fr_1fr] gap-3">
        <div className="flex flex-col gap-1">
          <label className="tcip-label">
            Mapping name (pick one already built under this project, or type a new one)
          </label>
          <input
            className="tcip-input"
            value={mappingName}
            onChange={(e) => {
              const name = e.target.value;
              setMappingName(name);
              if (mappingNames.includes(name)) {
                void loadMapping(name);
              } else {
                setBuildSummary(null);
                setBuildTolerance(null);
                setBuildMaxMatchDistance(null);
              }
            }}
            placeholder="valley-2026"
            list="plant-mapping-names"
          />
          <datalist id="plant-mapping-names">
            {mappingNames.map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
          <label className="tcip-label mt-1">
            Plant registry name (registered via register_plant_registry)
          </label>
          <input
            className="tcip-input"
            value={plantRegistry}
            onChange={(e) => setPlantRegistry(e.target.value)}
            placeholder="valley-plants"
          />
          <label className="tcip-label mt-1 flex items-center gap-2">
            <input
              type="checkbox"
              checked={supersedeMapping}
              onChange={(e) => setSupersedeMapping(e.target.checked)}
            />
            Supersede a mapping a delivery event still cites
          </label>
        </div>
        <div className="flex flex-col gap-2">
          <label className="tcip-label">Match tolerance (m)</label>
          <input
            className="tcip-input"
            type="number"
            step="1"
            min="0"
            placeholder="derived from grid pitch"
            value={nnTolerance}
            onChange={(e) =>
              setNnTolerance(e.target.value === "" ? "" : parseFloat(e.target.value) || 0)
            }
          />
          <button
            className="tcip-btn-primary"
            onClick={buildMapping}
            disabled={building || !datasetRoot}
          >
            {building ? "Building…" : "Build + save mapping"}
          </button>
          {buildMsg && <div className="text-[11px] text-tcip-muted">{buildMsg}</div>}
        </div>
      </div>
      {buildSummary && (
        <div className="mt-2 text-[11px] text-tcip-muted tabular-nums">
          {buildTolerance && buildMaxMatchDistance !== null && (
            <div className="mb-1">
              {`Match tolerance ${buildTolerance.value.toFixed(2)} m (${toleranceSourceText(buildTolerance.source)}); matches accepted out to ${buildMaxMatchDistance.toFixed(2)} m`}
            </div>
          )}
          {Object.entries(buildSummary.per_date).map(([d, s]) => (
            <div key={d}>
              {`${d}: mapped ${s.n_mapped} of ${s.n_images}, ${s.n_unattributed} attributed to no plant · avg ${s.avg_distance_m === null ? "no distances" : `${s.avg_distance_m.toFixed(1)} m`}`}
            </div>
          ))}
          {buildSummary.totals.n_unattributed > 0 && (
            <div className="mt-1">
              {`${buildSummary.totals.n_unattributed} captures across this mapping's dates are attributed to no plant (no readable position, a raster capture, or beyond the accepted match distance)`}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/** What a project is set up with before anything is measured: each trait's revisions for the
 *  breeder to confirm, and the plant mapping a phenology delivery reads. */
export function SetupTab() {
  const projectRoot = useStore((s) => s.gui.dataset.project_root);
  const datasetRoot = useStore((s) => s.gui.dataset.dataset_root);
  if (!projectRoot) return null;
  return (
    <div className="flex-1 overflow-auto p-4 flex flex-col gap-4">
      <TabHeading tab="setup" />
      <TraitRevisionPanel projectRoot={projectRoot} />
      <PlantMappingPanel key={projectRoot} datasetRoot={datasetRoot} />
    </div>
  );
}
