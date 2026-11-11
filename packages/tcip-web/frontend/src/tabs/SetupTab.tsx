import { useCallback, useEffect, useState } from "react";

import { api } from "@/api/client";
import { messageOf, refusalOf, StructuredRefusalError } from "@/api/http";
import { resultsApi } from "@/api/inference";
import {
  AUDIT_ENTRY_NOT_WRITTEN,
  type AuditGapRefusal,
  type CaptureInvariances,
  CROP,
  type MappingNames,
  type MatchRadius,
  READ,
  REFUSED,
  REGISTERED,
  type RootIdentity,
  type ServedMapping,
  UNREGISTERED,
} from "@/api/types.generated";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { TabHeading } from "@/components/TabHeading";
import { TraitRevisionPanel } from "@/components/TraitRevisionPanel";
import { answerPhrase, invarianceName } from "@/lib/invariances";
import { useStore } from "@/store";
import { selectDatasetRoot } from "@/store/slices/gui";

// A date's match radius as the breeder reads it, from the record's own source.
function radiusPhrase(radius: MatchRadius): string {
  const source =
    radius.source === "h_pos_err"
      ? "from the photographs' positioning error"
      : `stated: ${radius.rationale}`;
  return `radius ${radius.value.toFixed(2)} m ${source}`;
}

function PlantMappingPanel({ datasetRoot }: { datasetRoot: string | null }) {
  const [mappingName, setMappingName] = useState("");
  // Every mapping name already persisted under the open project, for the picker below; a name
  // typed here that isn't in the list is a new mapping this build will create.
  const [mappings, setMappings] = useState<MappingNames | null>(null);
  const [plantRegistry, setPlantRegistry] = useState("");
  const [statedRadius, setStatedRadius] = useState<number | "">("");
  const [radiusRationale, setRadiusRationale] = useState("");
  // Off by default: a rebuild a delivery event still cites answers 409 unless this is sent true.
  const [supersedeMapping, setSupersedeMapping] = useState(false);
  const [mapping, setMapping] = useState<ServedMapping | null>(null);
  const [buildMsg, setBuildMsg] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);

  const refreshMappingNames = useCallback(() => {
    void resultsApi
      .listPlantMappings()
      .then(setMappings)
      .catch(() => setMappings(null));
  }, []);

  useEffect(() => {
    refreshMappingNames();
  }, [refreshMappingNames]);

  // Selecting an already-built mapping by name shows its own summary and radii, the same way a
  // fresh build does.
  async function loadMapping(name: string) {
    try {
      setMapping(await resultsApi.loadPlantMapping({ name }));
    } catch (e) {
      useStore.getState().pushToast(`Could not load the mapping: ${messageOf(e)}`);
    }
  }

  async function buildMapping() {
    if (!datasetRoot) return;
    if (!mappingName) {
      setBuildMsg("Name the mapping.");
      return;
    }
    if (!plantRegistry) {
      setBuildMsg("Name the plant registry.");
      return;
    }
    setBuilding(true);
    setBuildMsg(null);
    setMapping(null);
    try {
      const built = await resultsApi.buildPlantMapping({
        name: mappingName,
        dataset_root: datasetRoot,
        plant_registry: plantRegistry,
        dates: null,
        supersede: supersedeMapping,
        match_radius:
          statedRadius === ""
            ? null
            : { source: "stated", value: statedRadius, rationale: radiusRationale },
        user: useStore.getState().user,
      });
      setMapping(built);
      setBuildMsg(`Built ${mappingName}.`);
      refreshMappingNames();
    } catch (e) {
      if (e instanceof StructuredRefusalError && e.status === 409) {
        // A rebuild cited by a delivery event, or a receipt that could not be written.
        setBuildMsg(e.message);
      } else {
        useStore.getState().pushToast(`Could not build the mapping: ${messageOf(e)}`);
        setBuildMsg(null);
      }
    } finally {
      setBuilding(false);
    }
  }

  return (
    <div className="tcip-panel p-5">
      <div className="tcip-heading mb-3">Plant mapping</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-3">
        <div className="flex flex-col gap-1">
          <label className="tcip-label" htmlFor="plant-mapping-name">
            Mapping
          </label>
          <input
            id="plant-mapping-name"
            className="tcip-input"
            value={mappingName}
            onChange={(e) => {
              const name = e.target.value;
              setMappingName(name);
              if (mappings?.names.includes(name)) void loadMapping(name);
              else setMapping(null);
            }}
            list="plant-mapping-names"
          />
          <datalist id="plant-mapping-names">
            {(mappings?.names ?? []).map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
        </div>
        <div className="flex flex-col gap-1">
          <label className="tcip-label" htmlFor="plant-registry-name">
            Plant registry
          </label>
          <input
            id="plant-registry-name"
            className="tcip-input"
            value={plantRegistry}
            onChange={(e) => setPlantRegistry(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="tcip-label" htmlFor="plant-mapping-radius">
            Match radius (m)
          </label>
          <input
            id="plant-mapping-radius"
            className="tcip-input"
            type="number"
            step="0.1"
            min="0"
            value={statedRadius}
            onChange={(e) =>
              setStatedRadius(e.target.value === "" ? "" : parseFloat(e.target.value) || 0)
            }
          />
        </div>
        {statedRadius !== "" && (
          <div className="flex flex-col gap-1">
            <label className="tcip-label" htmlFor="plant-mapping-radius-rationale">
              Radius source
            </label>
            <input
              id="plant-mapping-radius-rationale"
              className="tcip-input"
              value={radiusRationale}
              onChange={(e) => setRadiusRationale(e.target.value)}
            />
          </div>
        )}
      </div>
      <label className="mt-3 flex items-center gap-2">
        <input
          type="checkbox"
          checked={supersedeMapping}
          onChange={(e) => setSupersedeMapping(e.target.checked)}
        />
        Replace a mapping a delivery used
      </label>
      <div className="mt-4 flex items-center gap-3">
        <button
          className="tcip-btn-primary"
          onClick={() => void buildMapping()}
          disabled={building || !datasetRoot}
        >
          {building ? "Building…" : "Build"}
        </button>
        {buildMsg && <span className="text-tcip-muted">{buildMsg}</span>}
      </div>
      {mapping && (
        <div className="mt-3 text-tcip-muted tabular-nums">
          {Object.entries(mapping.per_date).map(([d, s]) => (
            <div key={d}>
              {`${d}: ${s.n_mapped} of ${s.n_images} mapped, ${s.n_unattributed} unmatched, ${s.avg_distance_m === null ? "no distances" : `avg ${s.avg_distance_m.toFixed(1)} m`}${d in mapping.match_radius ? `, ${radiusPhrase(mapping.match_radius[d])}` : ""}`}
            </div>
          ))}
          {mapping.images_unattributed > 0 && (
            <div className="mt-1">{`${mapping.images_unattributed} ${mapping.images_unattributed === 1 ? "capture" : "captures"} matched no plant.`}</div>
          )}
        </div>
      )}
    </div>
  );
}

/** What changing a dataset's crop refuses, shown before the change is confirmed. */
const CROP_CHANGE_CONSEQUENCE =
  "A mapped delivery refuses while its plant registry's crop is not the dataset's, and an " +
  "orthomosaic count is checked against the registry it names when it is delivered.";

/** The selected dataset root's identity as its document reads, registered from here at a crop of
 *  the served vocabulary; a registered crop changes only after a confirmation listing the open
 *  project's mappings built over the dataset with each one's plant registry crop. */
function DatasetIdentityPanel({ datasetRoot }: { datasetRoot: string | null }) {
  const user = useStore((s) => s.user);
  const [read, setRead] = useState<RootIdentity | null>(null);
  const [crop, setCrop] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  // The sentence of an id the registration restored, kept across the reread that follows it.
  const [repaired, setRepaired] = useState<string | null>(null);

  const refresh = useCallback(() => {
    if (!datasetRoot) return;
    void api.dataset
      .identity(datasetRoot)
      .then((answer) => {
        setRead(answer);
        setCrop(answer.identity.kind === REGISTERED ? answer.identity.identity.crop : "");
      })
      .catch((e: unknown) =>
        useStore.getState().pushToast(`Could not read the dataset identity: ${messageOf(e)}`),
      );
  }, [datasetRoot]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  if (!datasetRoot || !read) return null;
  const state = read.identity;
  const stored = state.kind === REGISTERED ? state.identity.crop : null;

  async function register(confirmedCrop: string | null) {
    if (!datasetRoot) return;
    setSubmitting(true);
    setRepaired(null);
    try {
      const answer = await api.dataset.register({
        dataset_root: datasetRoot,
        crop,
        user,
        confirmed_crop: confirmedCrop,
      });
      setRepaired(answer.repaired);
      useStore.getState().pushToast(`Registered at crop ${answer.crop}.`, "success");
    } catch (e) {
      // A registration whose audit line was lost still committed, so its message is no failure.
      const gap = refusalOf<AuditGapRefusal>(e, AUDIT_ENTRY_NOT_WRITTEN);
      useStore
        .getState()
        .pushToast(gap ? gap.message : `Could not register the dataset: ${messageOf(e)}`);
    } finally {
      setSubmitting(false);
      setConfirming(false);
      refresh();
    }
  }

  return (
    <div className="tcip-panel p-5">
      <div className="tcip-heading mb-3">Dataset identity</div>
      {state.kind === REGISTERED && (
        <p className="text-tcip-muted">{`Registered at crop ${state.identity.crop}, id ${state.identity.id}.`}</p>
      )}
      {state.kind === UNREGISTERED && (
        <p className="text-tcip-muted">No identity document is stored at this root.</p>
      )}
      {state.kind === REFUSED && <p className="tcip-problem">{state.reason}</p>}
      <div className="mt-3 flex items-end gap-3">
        <div className="flex flex-col gap-1">
          <label className="tcip-label" htmlFor="dataset-crop">
            Crop
          </label>
          <select
            id="dataset-crop"
            className="tcip-input"
            value={crop}
            onChange={(e) => setCrop(e.target.value)}
          >
            {stored === null && <option value="">Choose a crop</option>}
            {read.crops.map((name) => (
              <option key={name}>{name}</option>
            ))}
          </select>
        </div>
        {stored === null ? (
          <button
            className="tcip-btn-primary"
            onClick={() => void register(null)}
            disabled={!crop || submitting}
          >
            {submitting ? "Registering…" : "Register"}
          </button>
        ) : (
          <button
            className="tcip-btn-primary"
            onClick={() => setConfirming(true)}
            disabled={crop === stored || submitting}
          >
            Change crop
          </button>
        )}
      </div>
      {repaired && <p className="mt-2 text-tcip-muted">{repaired}</p>}
      {confirming && state.kind === REGISTERED && (
        <ConfirmDialog
          heading="Change the dataset's crop"
          onClose={() => setConfirming(false)}
          busy={submitting}
        >
          <div className="flex flex-col gap-3 text-[13px]">
            <p>{`Current crop: ${state.identity.crop}. Chosen crop: ${crop}.`}</p>
            <p className="text-tcip-muted">{CROP_CHANGE_CONSEQUENCE}</p>
            <ul aria-label="The open project's mappings over this dataset">
              {state.mappings.placed.map((m) => (
                <li key={m.name}>
                  {`${m.name}: plant registry ${m.plant_registry}, ${m.registry.kind === CROP ? `crop ${m.registry.crop}` : m.registry.refusal}`}
                </li>
              ))}
              {state.mappings.unplaced.map((m) => (
                <li key={m.name}>{`${m.name} could not be placed: ${m.refusal}`}</li>
              ))}
            </ul>
            {state.mappings.placed.length + state.mappings.unplaced.length === 0 && (
              <p className="text-tcip-muted">
                No plant mapping of the open project is built over this dataset.
              </p>
            )}
            <div className="flex gap-2">
              <button
                className="tcip-btn-primary flex-1"
                disabled={submitting}
                onClick={() => void register(state.identity.crop)}
              >
                {submitting ? "Changing…" : "Change crop"}
              </button>
              <button className="tcip-btn flex-1" onClick={() => setConfirming(false)}>
                Cancel
              </button>
            </div>
          </div>
        </ConfirmDialog>
      )}
    </div>
  );
}

type Answer = [keyof CaptureInvariances, boolean | number];

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
        const invariances =
          res.projects.filter((p) => p.kind === READ).find((p) => p.record.id === projectId)?.record
            .invariances ?? null;
        setRecorded(invariances);
        setDraft(invariances);
      })
      .catch((e: unknown) => setProblem(messageOf(e)));
  }, [projectId]);

  if (!recorded || !draft) {
    return problem ? <div className="tcip-panel p-5 tcip-problem">{problem}</div> : null;
  }
  const answers = Object.entries(draft) as Answer[];
  const unchanged = answers.every(([name, value]) => recorded[name] === value);

  async function correct(invariances: CaptureInvariances) {
    setSubmitting(true);
    setProblem(null);
    try {
      const { record, previous } = await api.projects.invariances({
        id: projectId,
        invariances,
        user,
      });
      const changed = (Object.entries(record.invariances) as Answer[])
        .filter(([name, value]) => previous.invariances[name] !== value)
        .map(
          ([name, value]) =>
            `${invarianceName(name)} ${answerPhrase(previous.invariances[name])} to ${answerPhrase(value)}`,
        );
      setRecorded(record.invariances);
      setDraft(record.invariances);
      useStore.getState().pushToast(`Saved: ${changed.join(", ")}.`, "success");
    } catch (e) {
      setProblem(messageOf(e));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="tcip-panel p-5">
      <div className="tcip-heading mb-3">Variation across photos</div>
      <div className="grid grid-cols-3 gap-4">
        {answers.map(([name, value]) => {
          const id = `invariance-${name}`;
          const label = invarianceName(name);
          return (
            <div key={name} className="flex flex-col gap-1">
              <label className="tcip-label" htmlFor={id}>
                {typeof value === "boolean" ? label : `${label} (degrees)`}
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
        className="tcip-btn-primary mt-4"
        onClick={() => void correct(draft)}
        disabled={unchanged || submitting}
      >
        {submitting ? "Saving…" : unchanged ? "Saved" : "Save"}
      </button>
      {problem && (
        <div className="mt-2 tcip-problem" aria-live="polite">
          {problem}
        </div>
      )}
    </div>
  );
}

/** What a project is set up with before anything is measured: each trait's revisions for the
 *  breeder to confirm, the capture invariances augmentations are judged against, the selected
 *  dataset's identity, and the plant mapping a phenology delivery reads. */
export function SetupTab() {
  const projectId = useStore((s) => s.openProject?.id ?? null);
  const datasetRoot = useStore(selectDatasetRoot);
  if (!projectId) return null;
  return (
    <div className="flex-1 overflow-auto p-5">
      <div className="mx-auto flex max-w-4xl flex-col gap-5">
        <TabHeading tab="setup" />
        <TraitRevisionPanel key={projectId} />
        <InvariancesPanel key={`invariances-${projectId}`} projectId={projectId} />
        <DatasetIdentityPanel
          key={`identity-${projectId}-${datasetRoot}`}
          datasetRoot={datasetRoot}
        />
        <PlantMappingPanel key={`mapping-${projectId}`} datasetRoot={datasetRoot} />
      </div>
    </div>
  );
}
