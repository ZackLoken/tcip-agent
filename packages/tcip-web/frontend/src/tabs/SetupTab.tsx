import { useCallback, useEffect, useState } from "react";

import { api } from "@/api/client";
import { StructuredRefusalError } from "@/api/http";
import { resultsApi, type PlantMappingSummary } from "@/api/inference";
import type { CaptureInvariances, MatchRadius } from "@/api/types.generated";
import { TabHeading } from "@/components/TabHeading";
import { TraitRevisionPanel } from "@/components/TraitRevisionPanel";
import { useStore } from "@/store";

// A date's match radius as the breeder reads it, from the record's own source.
function radiusPhrase(radius: MatchRadius): string {
  const where =
    radius.source === "h_pos_err"
      ? "the largest positioning error its photographs report"
      : `stated: ${radius.rationale}`;
  return `matched within ${radius.value.toFixed(2)} m (${where})`;
}

function PlantMappingPanel({ datasetRoot }: { datasetRoot: string | null }) {
  const [mappingName, setMappingName] = useState("");
  // Every mapping name already persisted under the open project, for the picker below; a name
  // typed here that isn't in the list is a new mapping this build will create.
  const [mappingNames, setMappingNames] = useState<string[]>([]);
  const [plantRegistry, setPlantRegistry] = useState("");
  const [statedRadius, setStatedRadius] = useState<number | "">("");
  const [radiusRationale, setRadiusRationale] = useState("");
  // Off by default: a rebuild a delivery event still cites answers 409 unless this is sent true.
  const [supersedeMapping, setSupersedeMapping] = useState(false);
  const [buildSummary, setBuildSummary] = useState<PlantMappingSummary | null>(null);
  const [buildRadius, setBuildRadius] = useState<Record<string, MatchRadius>>({});
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

  // Selecting an already-built mapping by name shows its own summary and radii, the same way a
  // fresh build does.
  async function loadMapping(name: string) {
    try {
      const res = await resultsApi.loadPlantMapping(name);
      setBuildSummary(res.summary);
      setBuildRadius(res.match_radius);
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
    setBuildRadius({});
    try {
      const res = await resultsApi.buildPlantMapping({
        name: mappingName,
        images_root: `${datasetRoot}/images`,
        plant_registry: plantRegistry,
        supersede: supersedeMapping,
        ...(statedRadius === ""
          ? {}
          : { match_radius: { value: statedRadius, rationale: radiusRationale } }),
        user: useStore.getState().user,
      });
      setBuildSummary(res.summary);
      setBuildRadius(res.match_radius);
      setBuildMsg(`Mapping built + saved as ${mappingName}`);
      refreshMappingNames();
    } catch (e) {
      if (e instanceof StructuredRefusalError && e.status === 409) {
        // A rebuild cited by a delivery event, or a receipt that could not be written.
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
          <label className="tcip-label" htmlFor="plant-mapping-name">
            Mapping name (pick one already built under this project, or type a new one)
          </label>
          <input
            id="plant-mapping-name"
            className="tcip-input"
            value={mappingName}
            onChange={(e) => {
              const name = e.target.value;
              setMappingName(name);
              if (mappingNames.includes(name)) {
                void loadMapping(name);
              } else {
                setBuildSummary(null);
                setBuildRadius({});
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
          <label className="tcip-label mt-1" htmlFor="plant-registry-name">
            Plant registry name (registered via register_plant_registry)
          </label>
          <input
            id="plant-registry-name"
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
          <label className="tcip-label" htmlFor="plant-mapping-radius">
            Match radius (m)
          </label>
          <input
            id="plant-mapping-radius"
            className="tcip-input"
            type="number"
            step="0.1"
            min="0"
            placeholder="each photograph's own reported positioning error"
            value={statedRadius}
            onChange={(e) =>
              setStatedRadius(e.target.value === "" ? "" : parseFloat(e.target.value) || 0)
            }
          />
          {statedRadius !== "" && (
            <>
              <label className="tcip-label" htmlFor="plant-mapping-radius-rationale">
                How the radius is known
              </label>
              <input
                id="plant-mapping-radius-rationale"
                className="tcip-input"
                value={radiusRationale}
                onChange={(e) => setRadiusRationale(e.target.value)}
                placeholder="the receiver's spec sheet"
              />
            </>
          )}
          <button
            className="tcip-btn-primary"
            onClick={() => void buildMapping()}
            disabled={building || !datasetRoot}
          >
            {building ? "Building…" : "Build + save mapping"}
          </button>
          {buildMsg && <div className="text-[13px] text-tcip-muted">{buildMsg}</div>}
        </div>
      </div>
      {buildSummary && (
        <div className="mt-2 text-[13px] text-tcip-muted tabular-nums">
          {Object.entries(buildSummary.per_date).map(([d, s]) => (
            <div key={d}>
              {`${d}: mapped ${s.n_mapped} of ${s.n_images}, ${s.n_unattributed} attributed to no plant · avg ${s.avg_distance_m === null ? "no distances" : `${s.avg_distance_m.toFixed(1)} m`}${d in buildRadius ? ` · ${radiusPhrase(buildRadius[d])}` : ""}`}
            </div>
          ))}
          {buildSummary.totals.n_unattributed > 0 && (
            <div className="mt-1">
              {`${buildSummary.totals.n_unattributed} captures across this mapping's dates are attributed to no plant (no readable position, a raster capture, or beyond the date's match radius)`}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

type Answer = [keyof CaptureInvariances, boolean | number];

// An invariance as the breeder reads it, named by its record field.
function invarianceName(name: keyof CaptureInvariances): string {
  return name.replaceAll("_", " ");
}

// An invariance answer as the breeder reads it: yes or no, or a rotation in degrees.
function answerPhrase(value: boolean | number): string {
  return typeof value === "boolean" ? (value ? "yes" : "no") : `${value} degrees`;
}

/** The capture invariances the open project's record carries, read from the project listing and
 *  corrected as one block through the record's correction door. */
function InvariancesPanel({ projectId }: { projectId: string }) {
  const user = useStore((s) => s.user);
  const [recorded, setRecorded] = useState<CaptureInvariances | null>(null);
  const [draft, setDraft] = useState<CaptureInvariances | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    void api.projects
      .list()
      .then((res) => {
        const invariances = res.projects.find((p) => p.id === projectId)?.invariances ?? null;
        setRecorded(invariances);
        setDraft(invariances);
      })
      .catch((e: unknown) => setProblem(e instanceof Error ? e.message : String(e)));
  }, [projectId]);

  if (!recorded || !draft) {
    return problem ? (
      <div className="tcip-panel p-4 text-[13px] tcip-problem">{problem}</div>
    ) : null;
  }
  const answers = Object.entries(draft) as Answer[];
  const unchanged = answers.every(([name, value]) => recorded[name] === value);

  async function correct(invariances: CaptureInvariances) {
    setSubmitting(true);
    setProblem(null);
    try {
      const res = await api.projects.invariances({ id: projectId, invariances, user });
      const changed = (Object.entries(res.invariances) as Answer[])
        .filter(([name, value]) => res.previous_invariances[name] !== value)
        .map(
          ([name, value]) =>
            `${invarianceName(name)} ${answerPhrase(res.previous_invariances[name])} to ${answerPhrase(value)}`,
        );
      setRecorded(res.invariances);
      setDraft(res.invariances);
      useStore
        .getState()
        .pushToast(`Capture invariances corrected: ${changed.join(", ")}.`, "success");
    } catch (e) {
      setProblem(e instanceof Error ? e.message : String(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="tcip-panel p-4">
      <div className="tcip-heading mb-1">Capture invariances</div>
      <p className="mb-3 text-[13px] text-tcip-muted">
        Whether each transform of an image leaves the trait&apos;s measurement what it was; every
        augmentation a training run states is judged against these answers.
      </p>
      <div className="grid grid-cols-3 gap-3">
        {answers.map(([name, value]) => {
          const id = `invariance-${name}`;
          const label = invarianceName(name);
          return (
            <div key={name} className="flex flex-col gap-1">
              <label className="tcip-label" htmlFor={id}>
                {typeof value === "boolean" ? label : `${label} (largest angle, degrees)`}
              </label>
              {typeof value === "boolean" ? (
                <select
                  id={id}
                  className="tcip-input"
                  value={answerPhrase(value)}
                  onChange={(e) =>
                    setDraft({ ...draft, [name]: e.target.value === answerPhrase(true) })
                  }
                >
                  {[true, false].map((answer) => (
                    <option key={answerPhrase(answer)}>{answerPhrase(answer)}</option>
                  ))}
                </select>
              ) : (
                <input
                  id={id}
                  className="tcip-input"
                  type="number"
                  value={Number.isNaN(value) ? "" : value}
                  onChange={(e) => setDraft({ ...draft, [name]: e.target.valueAsNumber })}
                />
              )}
            </div>
          );
        })}
      </div>
      <button
        className="tcip-btn-primary mt-3"
        onClick={() => void correct(draft)}
        disabled={unchanged || submitting}
      >
        {submitting ? "Correcting…" : "Correct invariances"}
      </button>
      {problem && (
        <div className="mt-2 text-[13px] tcip-problem" aria-live="polite">
          {problem}
        </div>
      )}
    </div>
  );
}

/** What a project is set up with before anything is measured: each trait's revisions for the
 *  breeder to confirm, the capture invariances augmentations are judged against, and the plant
 *  mapping a phenology delivery reads. */
export function SetupTab() {
  const projectId = useStore((s) => s.openProject?.id ?? null);
  const datasetRoot = useStore((s) => s.gui.dataset.dataset_root);
  if (!projectId) return null;
  return (
    <div className="flex-1 overflow-auto p-4 flex flex-col gap-4">
      <TabHeading tab="setup" />
      <TraitRevisionPanel key={projectId} />
      <InvariancesPanel key={`invariances-${projectId}`} projectId={projectId} />
      <PlantMappingPanel key={`mapping-${projectId}`} datasetRoot={datasetRoot} />
    </div>
  );
}
