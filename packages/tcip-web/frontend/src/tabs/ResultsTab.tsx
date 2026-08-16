import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { api } from "@/api/client";
import { StructuredRefusalError } from "@/api/http";
import {
  operationalizationRefusalOf,
  resultsApi,
  type OperationalizationRecord,
  type OperationalizationRefusal,
  type PhenologyRequest,
  type OnsetRow,
  type PerPlantRow,
  type PlantMappingSummary,
} from "@/api/inference";
import { DisclosureChevron } from "@/components/CollapsibleSection";
import { useEditableAgentRequest } from "@/hooks/useEditableAgentRequest";
import { useStore } from "@/store";
import { CHART, CHART_LINE_COLORS } from "@/tabs/chartTheme";

interface DateRow {
  date: string;
  [plantId: string]: number | string | null;
}

/** Breeder-facing wording only; the served `statement_fields` decides what a row shows, and a
 *  field without an entry here renders under its own name rather than disappearing. */
const STATEMENT_FIELD_LABELS: Record<string, string> = {
  statement: "What the number means",
  mechanism: "How the platform decides it",
  measured_subject: "Measured subject",
  delivered_phenotypes: "Delivered phenotypes",
  delivered_value_keys: "Delivered value keys",
  stated_by: "Recorded through",
  stated_at: "Recorded at",
  relayed_note: "Relayed note",
};

/** One record, addressed by the pair its store is keyed on. */
function recordKey(record: { trait: string; delivery_kind: string }): string {
  return `${record.trait}::${record.delivery_kind}`;
}

function statementFieldText(record: OperationalizationRecord, field: string): string {
  const values: Record<string, unknown> = record;
  const value = values[field];
  if (Array.isArray(value)) return value.length > 0 ? value.join(", ") : "none";
  if (typeof value !== "string" || value.trim() === "") return "none";
  return value;
}

function supersededValueText(value: unknown): string {
  if (value === null || value === undefined) return "nothing";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

/**
 * The record a 409 carries, or null when the failure is anything else.
 *
 * The confirm route answers a moved record with what is on file now, so the row can re-render the
 * text the breeder has to read before their click means anything.
 */
function movedRecordFrom(e: unknown): OperationalizationRecord | null {
  if (!(e instanceof StructuredRefusalError) || e.status !== 409) return null;
  const record = (e.detail as { record?: unknown }).record;
  return record && typeof (record as { record_seen?: unknown }).record_seen === "string"
    ? (record as OperationalizationRecord)
    : null;
}

// The agent states, the breeder confirms; a correction is a message that proposes no meaning.
function correctionRequest(record: OperationalizationRecord): string {
  return (
    `The operationalization on record for the "${record.trait}" trait's ${record.delivery_kind} ` +
    `delivery is not what I mean. On record: ${record.statement} Decided by: ${record.mechanism} ` +
    `Measured subject: ${record.measured_subject}. What should change: `
  );
}

function OperationalizationRow({
  record,
  statementFields,
  refused,
  confirming,
  withdrawing,
  note,
  onConfirm,
  onWithdraw,
}: {
  record: OperationalizationRecord;
  statementFields: string[];
  refused: boolean;
  confirming: boolean;
  withdrawing: boolean;
  note: string | null;
  onConfirm: (record: OperationalizationRecord) => void;
  onWithdraw: (record: OperationalizationRecord) => void;
}) {
  const { request, setRequest } = useEditableAgentRequest(correctionRequest(record));
  const [correcting, setCorrecting] = useState(false);

  const delivers =
    record.delivers.map((d) => `${d.name}: ${d.definition}`).join("; ") ||
    "nothing this project's crop vocabulary defines";

  return (
    <li
      className={`rounded border p-3 ${refused ? "border-tcip-fp" : "border-tcip-border"}`}
      data-testid={`operationalization-${recordKey(record)}`}
    >
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[12px]">{record.trait}</span>
        <span className="text-[11px] text-tcip-muted">{record.delivery_kind}</span>
        <span
          className={`ml-auto text-[11px] ${record.confirmed_current ? "text-tcip-muted" : "text-tcip-fp"}`}
        >
          {record.confirmed_by && record.confirmed_current
            ? `Confirmed by ${record.confirmed_by} on ${record.confirmed_at}`
            : record.confirmed_by
              ? `Confirmed by ${record.confirmed_by} on ${record.confirmed_at}, no longer current`
              : "Not confirmed"}
        </span>
      </div>
      {record.confirmed_by && (
        <div className="text-[11px] text-tcip-muted">
          {record.identity_from_request
            ? "That name came with the confirming request."
            : "That name came from the backend's own environment, not from the confirming request."}
        </div>
      )}
      <p className="mt-1 text-[11px] text-tcip-muted">Delivers {delivers}</p>
      <dl className="mt-2 grid grid-cols-[170px_1fr] gap-x-3 gap-y-1 text-[11px]">
        {statementFields.map((field) => (
          <Fragment key={field}>
            <dt className="text-tcip-muted">{STATEMENT_FIELD_LABELS[field] ?? field}</dt>
            <dd>{statementFieldText(record, field)}</dd>
          </Fragment>
        ))}
      </dl>
      {record.superseded.length > 0 && (
        <div className="mt-2 rounded border border-tcip-fp/40 p-2 text-[11px] text-tcip-fp">
          <div>Changed since this was confirmed:</div>
          <ul className="mt-1 list-disc pl-4">
            {record.superseded.map((s) => (
              <li key={s.field}>
                {s.field}: confirmed as {supersededValueText(s.confirmed_value)}, now{" "}
                {supersededValueText(s.current_value)}
              </li>
            ))}
          </ul>
        </div>
      )}
      {note && <div className="mt-2 text-[11px] text-tcip-fp">{note}</div>}
      <div className="mt-2 flex items-center gap-2">
        {!record.confirmed_current && (
          <button
            className="tcip-btn-primary text-[11px]"
            onClick={() => onConfirm(record)}
            disabled={confirming || withdrawing}
          >
            {confirming ? "Confirming…" : "Confirm this record"}
          </button>
        )}
        {record.confirmed_by && (
          <button
            className="tcip-btn text-[11px]"
            onClick={() => onWithdraw(record)}
            disabled={confirming || withdrawing}
          >
            {withdrawing ? "Withdrawing…" : "Withdraw this confirmation"}
          </button>
        )}
        <button
          className="tcip-btn text-[11px]"
          aria-expanded={correcting}
          onClick={() => setCorrecting((open) => !open)}
        >
          <DisclosureChevron open={correcting} />
          Send a correction to the agent
        </button>
      </div>
      {correcting && (
        <div className="mt-2 flex flex-col gap-1">
          <textarea
            className="tcip-input h-20 w-full resize-none text-[11px] leading-4"
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            spellCheck={true}
            aria-label={`Correction for ${record.trait}, ${record.delivery_kind}`}
          />
          <button
            className="tcip-btn self-start text-[11px]"
            onClick={() => useStore.getState().sendToAgentTerminal(request)}
            disabled={!request.trim()}
          >
            Send to the agent
          </button>
        </div>
      )}
    </li>
  );
}

function OperationalizationPanel({
  records,
  statementFields,
  loadError,
  refusal,
  confirmingKey,
  withdrawingKey,
  notes,
  onConfirm,
  onWithdraw,
}: {
  records: OperationalizationRecord[];
  statementFields: string[];
  loadError: string | null;
  refusal: OperationalizationRefusal | null;
  confirmingKey: string | null;
  withdrawingKey: string | null;
  notes: Record<string, string>;
  onConfirm: (record: OperationalizationRecord) => void;
  onWithdraw: (record: OperationalizationRecord) => void;
}) {
  return (
    <div className="tcip-panel p-4">
      <div className="tcip-heading mb-1">What the delivered numbers mean</div>
      <p className="mb-3 text-[11px] text-tcip-muted">
        The agent records what each delivered number means and how the platform decides it. A
        delivery waits until you confirm the record it would ship under. Read what is on file, then
        confirm it or send the agent a correction. A confirmation you gave stands until you withdraw
        it.
      </p>
      {refusal && (
        <div className="mb-3 rounded border border-tcip-fp/40 p-2 text-[11px] text-tcip-fp">
          <div>
            A delivery for the {refusal.trait} trait's {refusal.delivery_kind} number is waiting on
            a confirmed record of what that number means.
          </div>
          <div className="mt-1 text-tcip-muted">{refusal.message}</div>
        </div>
      )}
      {loadError && <div className="mb-3 text-[11px] text-tcip-fp">{loadError}</div>}
      {records.length === 0 ? (
        <div className="text-[11px] text-tcip-muted">
          Nothing is recorded for this project yet. The agent records one before a delivery can ship
          under it.
        </div>
      ) : (
        <ul className="flex flex-col gap-2">
          {records.map((record) => (
            <OperationalizationRow
              key={recordKey(record)}
              record={record}
              statementFields={statementFields}
              refused={
                refusal !== null &&
                refusal.trait === record.trait &&
                refusal.delivery_kind === record.delivery_kind
              }
              confirming={confirmingKey === recordKey(record)}
              withdrawing={withdrawingKey === recordKey(record)}
              note={notes[recordKey(record)] || null}
              onConfirm={onConfirm}
              onWithdraw={onWithdraw}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

/**
 * Parse an ISO date (`YYYY-MM-DD`) into a sortable integer. Mirrors the backend `_date_key`
 * in results.py so the chart's date order matches the server-computed onset table.
 */
function dateKey(date: string): number {
  const parts = date.split("-");
  if (parts.length !== 3) return 0;
  // Match the backend _date_key (Python int()): reject junk-suffixed parts like "15b"
  // so the chart's date order agrees with the server-computed onset ordering.
  if (!parts.every((p) => /^\d+$/.test(p))) return 0;
  const [y, m, d] = parts.map((x) => parseInt(x, 10));
  return y * 10000 + m * 100 + d;
}

export function ResultsTab() {
  const dataset = useStore((s) => s.gui.dataset);
  const projectRoot = dataset.project_root;
  const datasetRoot = dataset.dataset_root;

  const [mappingPath, setMappingPath] = useState(
    projectRoot ? `${projectRoot}/.tcip/state/plant_mapping.json` : "",
  );
  // True unless a computed run reported that its predictions carried no positive-state class.
  const [positiveClassUnassessed, setPositiveClassUnassessed] = useState(false);

  // Dataset tree (dates + which models actually have predictions per date) drives the structured
  // per-date picker below, never a hand-edited JSON blob; models_with_predictions is the same
  // primitive the backend already computes this from, via api.dataset.tree.
  const [dates, setDates] = useState<string[]>([]);
  const [modelsByDate, setModelsByDate] = useState<Record<string, string[]>>({});
  const [predictionDirs, setPredictionDirs] = useState<Record<string, Record<string, string>>>({});
  const [datesError, setDatesError] = useState<string | null>(null);
  // The model picked per date; "" means "skip this date" (dropped before compute()).
  const [dateModel, setDateModel] = useState<Record<string, string>>({});

  // Plant-mapping build inputs.
  const [plantCsvText, setPlantCsvText] = useState("");
  const [nnTolerance, setNnTolerance] = useState(10);
  const [buildSummary, setBuildSummary] = useState<PlantMappingSummary | null>(null);
  const [buildMsg, setBuildMsg] = useState<string | null>(null);
  const [building, setBuilding] = useState(false);

  const [curves, setCurves] = useState<PerPlantRow[]>([]);
  const [onset, setOnset] = useState<OnsetRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  // The exact request the displayed numbers came from: the CSV door recomputes from these inputs
  // rather than being handed the rows, so export and screen share one producer.
  const [lastRequest, setLastRequest] = useState<PhenologyRequest | null>(null);
  // Reconciled evidence for what is currently displayed. `provisional` is true whenever a dimension
  // lacked on-disk backing, so the tables can say so instead of rendering a phenology date as "valid".
  const [provisional, setProvisional] = useState(false);
  const [validity, setValidity] = useState<Record<string, string>>({});
  const [unvalidatedRefusal, setUnvalidatedRefusal] = useState<string | null>(null);

  // Listed rather than selected: records are keyed by trait plus kind, including uncomputed kinds.
  const [operationalizations, setOperationalizations] = useState<OperationalizationRecord[]>([]);
  // Which fields a confirmation covers, and their order, as the record's own module names them.
  const [statementFields, setStatementFields] = useState<string[]>([]);
  const [operationalizationsError, setOperationalizationsError] = useState<string | null>(null);
  const [operationalizationRefusal, setOperationalizationRefusal] =
    useState<OperationalizationRefusal | null>(null);
  const [confirmingKey, setConfirmingKey] = useState<string | null>(null);
  const [withdrawingKey, setWithdrawingKey] = useState<string | null>(null);
  const [confirmNotes, setConfirmNotes] = useState<Record<string, string>>({});

  // The trait a delivery is computed for, resolved from this project's own registered traits
  // (never assumed): auto-selected when there is exactly one, left blank (with an explicit
  // error, not a silent guess) when there are zero, offered as a choice when there are several.
  const [availableTraits, setAvailableTraits] = useState<string[]>([]);
  // Each trait's declared milestone fractions, straight from its spec: what this tab can compute
  // for the selected trait follows from them, not from a category the server assigned.
  const [milestoneFractionsByTrait, setMilestoneFractionsByTrait] = useState<
    Record<string, number[]>
  >({});
  const [trait, setTrait] = useState("");
  const [traitError, setTraitError] = useState<string | null>(null);
  // Trait specs that failed to load, so a breeder can tell "nothing registered" from
  // "something is registered but broken" instead of the two looking identical.
  const [invalidSpecs, setInvalidSpecs] = useState<{ file: string; reason: string }[]>([]);

  useEffect(() => {
    if (!projectRoot) return;
    setTrait("");
    setTraitError(null);
    void resultsApi
      .traits(projectRoot)
      .then((res) => {
        setAvailableTraits(res.traits);
        setMilestoneFractionsByTrait(res.milestone_fractions_by_trait);
        setInvalidSpecs(res.invalid_specs);
        if (res.traits.length === 0) {
          setTraitError("No trait is registered for this project yet.");
        } else if (res.traits.length === 1) {
          setTrait(res.traits[0]);
        }
      })
      .catch((e) => {
        setAvailableTraits([]);
        setMilestoneFractionsByTrait({});
        setInvalidSpecs([]);
        setTraitError(
          `Could not load this project's registered traits: ${e instanceof Error ? e.message : String(e)}`,
        );
      });
  }, [projectRoot]);

  useEffect(() => {
    if (!projectRoot) return;
    void resultsApi
      .operationalizations(projectRoot)
      .then((res) => {
        setOperationalizations(res.records);
        setStatementFields(res.statement_fields);
        setOperationalizationsError(null);
      })
      .catch((e) => {
        setOperationalizations([]);
        setStatementFields([]);
        setOperationalizationsError(
          `Could not load what this project's delivered numbers mean: ${e instanceof Error ? e.message : String(e)}`,
        );
      });
  }, [projectRoot]);

  const replaceOperationalization = useCallback((record: OperationalizationRecord) => {
    setOperationalizations((prev) =>
      prev.map((r) => (recordKey(r) === recordKey(record) ? record : r)),
    );
  }, []);

  // Withdrawal is the same door with confirmed false, so one writer serves both directions.
  const writeConfirmation = useCallback(
    async (record: OperationalizationRecord, confirmed: boolean) => {
      if (!projectRoot) return;
      const key = recordKey(record);
      const setPending = confirmed ? setConfirmingKey : setWithdrawingKey;
      setPending(key);
      setConfirmNotes((prev) => ({ ...prev, [key]: "" }));
      try {
        await resultsApi.confirmOperationalization({
          project_root: projectRoot,
          trait: record.trait,
          delivery_kind: record.delivery_kind,
          record_seen: record.record_seen,
          confirmed,
          user: useStore.getState().user || undefined,
        });
        replaceOperationalization(
          await resultsApi.operationalization(projectRoot, record.trait, record.delivery_kind),
        );
      } catch (e) {
        const moved = movedRecordFrom(e);
        if (moved) {
          replaceOperationalization(moved);
          setConfirmNotes((prev) => ({
            ...prev,
            [key]: confirmed
              ? "This record changed since it was shown. Read what is on file above, then confirm that."
              : "This record changed since it was shown. Read what is on file above, then withdraw that.",
          }));
        } else {
          setConfirmNotes((prev) => ({
            ...prev,
            [key]: `${confirmed ? "Could not confirm" : "Could not withdraw"}: ${
              e instanceof Error ? e.message : String(e)
            }`,
          }));
        }
      } finally {
        setPending(null);
      }
    },
    [projectRoot, replaceOperationalization],
  );

  const refreshDatasetTree = useCallback(() => {
    if (!datasetRoot) return;
    void api.dataset
      .tree(datasetRoot)
      .then((t) => {
        setDates(t.dates_with_images);
        setModelsByDate(t.models_by_date);
        setPredictionDirs(t.prediction_dirs);
        // Default each date to its first model with predictions; a date with none stays "" (skip).
        setDateModel(
          Object.fromEntries(t.dates_with_images.map((d) => [d, t.models_by_date[d]?.[0] ?? ""])),
        );
        setDatesError(null);
      })
      .catch((e) => {
        setDatesError(
          `Could not load this dataset's dates: ${e instanceof Error ? e.message : String(e)}`,
        );
      });
  }, [datasetRoot]);

  useEffect(() => {
    refreshDatasetTree();
  }, [refreshDatasetTree]);

  // The dir the backend itself says a model's predictions for a date live in, looked up from the
  // tree response. A path assembled here would only agree with the writers by coincidence.
  function predDirFor(date: string, model: string): string {
    return (model && predictionDirs[date]?.[model]) || "";
  }

  async function buildMapping() {
    if (!datasetRoot) return;
    const paths = plantCsvText
      .split(/[\n,]/)
      .map((s) => s.trim())
      .filter(Boolean);
    if (paths.length === 0) {
      setBuildMsg("Add at least one plant CSV path.");
      return;
    }
    setBuilding(true);
    setBuildMsg(null);
    setBuildSummary(null);
    try {
      const res = await resultsApi.buildPlantMapping({
        images_root: `${datasetRoot}/images`,
        plant_csv_paths: paths,
        nn_tolerance_m: nnTolerance,
        persist_path: mappingPath || undefined,
      });
      setBuildSummary(res.summary);
      setBuildMsg(`Mapping built + saved to ${mappingPath}`);
    } catch (e) {
      useStore
        .getState()
        .pushToast(`Build mapping failed: ${e instanceof Error ? e.message : String(e)}`);
      setBuildMsg(null);
    } finally {
      setBuilding(false);
    }
  }

  async function compute(acknowledgeUnvalidated = false) {
    if (!projectRoot) return;
    if (!trait) {
      setError(traitError ?? "Pick a trait before computing.");
      return;
    }
    setLoading(true);
    setError(null);
    setOperationalizationRefusal(null);
    try {
      const predsMap: Record<string, string> = {};
      for (const d of dates) {
        const dir = predDirFor(d, dateModel[d] ?? "");
        if (dir) predsMap[d] = dir;
      }
      const request = {
        project_root: projectRoot,
        mapping_path: mappingPath,
        predictions_by_date: predsMap,
        trait,
        acknowledge_unvalidated: acknowledgeUnvalidated,
      };
      setLastRequest(request);
      const curveRes = await resultsApi.perPlantCurves(request);
      // The numbers and the evidence that qualifies them arrive together, so the tables below can
      // never render an unvalidated phenology measurement as though it were a delivery.
      setProvisional(curveRes.provisional);
      setValidity(curveRes.validated);
      setUnvalidatedRefusal(null);
      const unclassified = curveRes.positive_class_assessed === false;
      setPositiveClassUnassessed(unclassified);
      setCurves(curveRes.rows ?? []);
      if (unclassified) {
        // No positive-state class → the fraction is not a phenology measurement. Don't derive
        // milestones from it at all, so there is nothing to export (belt-and-braces with the
        // disabled export buttons + the compute_phenology MCP tool's hard refusal).
        setOnset([]);
      } else {
        // Same inputs, not the curve rows: the server recomputes rather than trusting a table the
        // client hands back, so a milestone date and the curve it was read off cannot disagree.
        const onsetRes = await resultsApi.onsetDates(request);
        setOnset(onsetRes.rows ?? []);
      }
    } catch (e) {
      // A refusal naming its own kind is routed by that kind, before any prose is read.
      const refusal = operationalizationRefusalOf(e);
      if (refusal) {
        setOperationalizationRefusal(refusal);
        setCurves([]);
        setOnset([]);
        return;
      }
      // The server refuses unvalidated evidence by default. Surface why, plus the one-click way to
      // see the numbers anyway (clearly marked provisional), so an uncalibrated operating point is
      // a signposted next step rather than a dead end.
      const detail = e instanceof Error ? e.message : String(e);
      if (!acknowledgeUnvalidated && /unvalidated|not validated/i.test(detail)) {
        setUnvalidatedRefusal(detail);
        setCurves([]);
        setOnset([]);
      } else {
        setError(detail);
      }
    } finally {
      setLoading(false);
    }
  }

  async function downloadCsv(payload: "curves" | "milestones", filename: string) {
    if (!lastRequest) return;
    try {
      const blob = await resultsApi.exportCsv(lastRequest, payload, filename);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      // The download door refuses through the same structured family, so it lands in the panel.
      const refusal = operationalizationRefusalOf(e);
      if (refusal) {
        setOperationalizationRefusal(refusal);
        return;
      }
      useStore
        .getState()
        .pushToast(`CSV export failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  // Measurement-integrity guard: never export a phenology CSV built on predictions that carry no
  // positive-state class, or on provisional evidence. Mirrors compute_phenology, which
  // hard-refuses both, so the GUI and the agent surface behave identically (see CLAUDE.md
  // invariant). The server refuses either case regardless; these keep the button from promising
  // what it can't do.
  const exportBlocked = positiveClassUnassessed || provisional;
  const downloadOnsetCsv = () => {
    if (exportBlocked) return;
    void downloadCsv("milestones", `${trait}_phenology.csv`);
  };
  const downloadCurvesCsv = () => {
    if (exportBlocked) return;
    // A curve export is the same delivered phenology measurement as the milestone one, just
    // un-summarised: same producer, same gate.
    void downloadCsv("curves", `${trait}_curves.csv`);
  };

  const chartData: DateRow[] = useMemo(() => {
    const byDate: Record<string, DateRow> = {};
    for (const r of curves) {
      byDate[r.date] ??= { date: r.date };
      byDate[r.date][r.plant_id] = r.ratio;
    }
    return Object.values(byDate).sort((a, b) => dateKey(a.date) - dateKey(b.date));
  }, [curves]);

  const plantKeys = useMemo(() => {
    const set = new Set<string>();
    curves.forEach((r) => set.add(r.plant_id));
    return Array.from(set);
  }, [curves]);

  // Which delivery dimensions the reconciled evidence actually failed on, reused to make the
  // agent hand-off below specific to what's missing rather than a generic "go calibrate" ask.
  const unvalidatedDims = useMemo(
    () =>
      Object.entries(validity)
        .filter(([, state]) => state === "false")
        .map(([dim]) => dim),
    [validity],
  );

  // What a breeder can't act on themselves: the backend refuses to deliver phenology until a
  // calibrated export_predictions + calibrate_classifier_operating_point stand behind it (see
  // results.py's _refusal). Hand that off to the agent instead of leaving the tool names on
  // screen with no next step.
  function calibrationRequest(detail: string | null): string {
    const dims = unvalidatedDims.length > 0 ? unvalidatedDims.join(", ") : "the operating point";
    const subject = trait ? `the "${trait}" trait` : "this trait";
    return (
      `Phenology delivery for ${subject} is blocked: ${dims} not validated on disk. ` +
      "Please produce the predictions via a calibrated export_predictions and calibrate the " +
      "classifier via calibrate_classifier_operating_point so this validates, then let me know " +
      "when it's ready so I can recompute here." +
      (detail ? ` Details from the app: ${detail}` : "")
    );
  }

  // The breeder can't author a trait spec from the GUI, so a trait with no milestone fractions
  // needs a way forward rather than an empty tab.
  function milestoneAbsenceRequest(): string {
    return (
      `The Results tab has nothing to compute for the "${trait}" trait: its spec declares no ` +
      "milestone fractions. Please tell me what this trait's measurement delivers and how I get " +
      "it, and update the spec if milestones are part of it."
    );
  }

  // Milestone columns are read generically off whatever the (threaded) trait's spec returned,
  // never hardcoded to one trait's own column names, so a different trait's rows render instead
  // of showing empty.
  const milestoneColumns = useMemo(() => {
    const known = new Set([
      "plant_id",
      "accession",
      "n_datapoints",
      "n_dates_unclassified",
      "n_dates_missing_images",
      "n_observed_dates",
    ]);
    const cols = new Set<string>();
    onset.forEach((r) => {
      Object.keys(r).forEach((k) => {
        // `_date` only: each milestone's `*_date_bound` is rendered beside its own date below
        // rather than as a column of its own.
        if (!known.has(k) && k.endsWith("_date")) cols.add(k);
      });
    });
    return Array.from(cols).sort();
  }, [onset]);

  // Curves and milestones are only meaningful for a trait whose spec declares the fractions they
  // are read off. Nothing else on this tab depends on it.
  const hasMilestones = (milestoneFractionsByTrait[trait] ?? []).length > 0;

  return (
    <div className="flex-1 overflow-auto p-4 flex flex-col gap-4">
      {traitError && <div className="tcip-panel p-3 text-[11px] text-tcip-fp">{traitError}</div>}
      {invalidSpecs.length > 0 && (
        <div className="tcip-panel p-3 text-[11px] text-tcip-fp">
          <div>
            {invalidSpecs.length} trait spec{invalidSpecs.length > 1 ? "s" : ""} failed to load and
            {invalidSpecs.length > 1 ? " are" : " is"} not registered:
          </div>
          <ul className="mt-1 list-disc pl-4">
            {invalidSpecs.map((s) => (
              <li key={s.file}>
                {s.file}: {s.reason}
              </li>
            ))}
          </ul>
        </div>
      )}
      {availableTraits.length > 1 && (
        <div className="tcip-panel p-3 flex items-center gap-2">
          <label className="tcip-label">Trait</label>
          <select
            className="tcip-input w-auto"
            value={trait}
            onChange={(e) => setTrait(e.target.value)}
          >
            <option value="" disabled>
              Choose a trait…
            </option>
            {availableTraits.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
      )}
      {/* Plant mapping: build (from geolocated images + plant CSVs) or point at an existing file */}
      <div className="tcip-panel p-4">
        <div className="tcip-heading mb-3">Plant mapping</div>
        <div className="grid grid-cols-[1fr_1fr] gap-3">
          <div className="flex flex-col gap-1">
            <label className="tcip-label">
              Mapping file (built here, or an existing one to load)
            </label>
            <input
              className="tcip-input"
              value={mappingPath}
              onChange={(e) => setMappingPath(e.target.value)}
              placeholder="…/.tcip/state/plant_mapping.json"
            />
            <label className="tcip-label mt-1">Plant CSV path(s), one per line</label>
            <textarea
              className="tcip-input h-16 font-mono text-[11px] leading-4"
              value={plantCsvText}
              onChange={(e) => setPlantCsvText(e.target.value)}
              placeholder="…/plants_block_A.csv"
              spellCheck={false}
            />
          </div>
          <div className="flex flex-col gap-2">
            <label className="tcip-label">NN tolerance (m)</label>
            <input
              className="tcip-input"
              type="number"
              step="1"
              min="0"
              value={nnTolerance}
              onChange={(e) => setNnTolerance(parseFloat(e.target.value) || 0)}
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
            {Object.entries(buildSummary).map(([d, s]) => (
              <div key={d}>
                {d}: {s.n_mapped}/{s.n_images} mapped · avg {s.avg_distance_m.toFixed(1)} m
              </div>
            ))}
          </div>
        )}
      </div>

      <OperationalizationPanel
        records={operationalizations}
        statementFields={statementFields}
        loadError={operationalizationsError}
        refusal={operationalizationRefusal}
        confirmingKey={confirmingKey}
        withdrawingKey={withdrawingKey}
        notes={confirmNotes}
        onConfirm={(record) => void writeConfirmation(record, true)}
        onWithdraw={(record) => void writeConfirmation(record, false)}
      />

      {trait && !hasMilestones && (
        <div className="tcip-panel p-4 flex flex-col gap-2">
          <div className="tcip-heading">Nothing to compute here for {trait}</div>
          <p className="text-[11px] text-tcip-muted">
            This trait's spec declares no milestone fractions, so there are no curves or milestones
            to compute for it.
          </p>
          <button
            className="tcip-btn-primary text-[11px] self-start"
            onClick={() => useStore.getState().sendToAgentTerminal(milestoneAbsenceRequest())}
          >
            Ask the agent what this trait delivers
          </button>
        </div>
      )}

      {hasMilestones && (
        <>
          <div className="tcip-panel p-4">
            <div className="tcip-heading mb-3">Per-plant phenology curves</div>
            <div className="grid grid-cols-[1fr_180px] gap-3">
              <div className="flex flex-col gap-1">
                <label className="tcip-label">Predictions by date</label>
                {datesError && (
                  <div className="text-[11px] text-tcip-fp mb-1">
                    {datesError}{" "}
                    <button className="tcip-btn text-[11px] ml-1" onClick={refreshDatasetTree}>
                      Retry
                    </button>
                  </div>
                )}
                {dates.length === 0 ? (
                  !datesError && (
                    <div className="text-[11px] text-tcip-muted">No dates in this dataset yet.</div>
                  )
                ) : (
                  <div className="max-h-40 overflow-auto rounded border border-tcip-border">
                    <table className="w-full text-[11px]">
                      <tbody>
                        {dates.map((d) => {
                          const opts = modelsByDate[d] ?? [];
                          return (
                            <tr key={d} className="border-t border-tcip-border first:border-t-0">
                              <td className="py-1 pl-2 pr-2 font-mono tabular-nums">{d}</td>
                              <td className="py-1 pr-2">
                                <select
                                  className="tcip-select text-[11px] w-full"
                                  value={dateModel[d] ?? ""}
                                  onChange={(e) =>
                                    setDateModel((prev) => ({ ...prev, [d]: e.target.value }))
                                  }
                                  disabled={opts.length === 0}
                                  title={
                                    opts.length === 0
                                      ? "No model has predictions for this date"
                                      : "Model whose predictions to use for this date"
                                  }
                                >
                                  <option value="">
                                    {opts.length === 0 ? "no predictions" : "(skip)"}
                                  </option>
                                  {opts.map((m) => (
                                    <option key={m} value={m}>
                                      {m}
                                    </option>
                                  ))}
                                </select>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
              <div className="flex flex-col gap-2">
                <p className="text-[11px] text-tcip-muted">
                  The positive-state fraction is the share of a plant's detected objects that are in
                  the trait's positive state. That state is a class from the validated classifier,
                  not a bbox measurement; predictions must be classified for it.
                </p>
                {positiveClassUnassessed && (
                  <div className="text-[11px] text-tcip-fp border border-tcip-fp/40 rounded p-2">
                    These predictions carry no positive-state class, so the curves below are not a
                    valid phenology measurement and CSV export is disabled. Run the classifier
                    first.
                  </div>
                )}
                {unvalidatedRefusal && (
                  <div className="text-[11px] text-tcip-fp border border-tcip-fp/40 rounded p-2 flex flex-col gap-2">
                    <div>
                      These predictions have no validated operating point on disk, so this is not
                      yet a deliverable phenology measurement. Calibrate first, or look at the
                      numbers as provisional, which will not let you export them.
                    </div>
                    <div className="text-tcip-muted">{unvalidatedRefusal}</div>
                    <div className="flex gap-2">
                      <button
                        className="tcip-btn text-[11px] self-start"
                        onClick={() => void compute(true)}
                        disabled={loading}
                      >
                        Show provisional numbers
                      </button>
                      <button
                        className="tcip-btn-primary text-[11px] self-start"
                        onClick={() =>
                          useStore
                            .getState()
                            .sendToAgentTerminal(calibrationRequest(unvalidatedRefusal))
                        }
                      >
                        Ask the agent to calibrate this
                      </button>
                    </div>
                  </div>
                )}
                {provisional && (
                  <div className="text-[11px] text-tcip-fp border border-tcip-fp/40 rounded p-2 flex flex-col gap-2">
                    <div>
                      Provisional: shown for inspection only, not a deliverable phenotype.
                      Unvalidated: {unvalidatedDims.join(", ") || "unknown"}. CSV export stays
                      disabled until both dimensions are validated on disk.
                    </div>
                    <button
                      className="tcip-btn-primary text-[11px] self-start"
                      onClick={() =>
                        useStore.getState().sendToAgentTerminal(calibrationRequest(null))
                      }
                    >
                      Ask the agent to calibrate this
                    </button>
                  </div>
                )}
                <button
                  className="tcip-btn-primary"
                  onClick={() => void compute()}
                  disabled={loading}
                >
                  {loading ? "Computing…" : "Compute curves + milestone dates"}
                </button>
                <p className="text-[10px] text-tcip-muted">
                  One computed measurement in two shapes: every (plant, date) point, or the
                  milestone dates read off it.
                </p>
                <div className="flex gap-1">
                  <button
                    className="tcip-btn flex-1 text-[11px]"
                    onClick={downloadCurvesCsv}
                    disabled={curves.length === 0 || exportBlocked}
                  >
                    Curves CSV
                  </button>
                  <button
                    className="tcip-btn flex-1 text-[11px]"
                    onClick={downloadOnsetCsv}
                    disabled={onset.length === 0 || exportBlocked}
                  >
                    Milestones CSV
                  </button>
                </div>
              </div>
            </div>
            {error && <div className="mt-2 text-[11px] text-tcip-fp">{error}</div>}
          </div>

          <div className="tcip-panel p-4 h-80">
            <div className="tcip-heading mb-3">
              Positive-state fraction over time, per plant
              {plantKeys.length > 30
                ? ` (showing 30 of ${plantKeys.length} plants, the milestones table below has all)`
                : ` (${plantKeys.length} plants)`}
            </div>
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="90%">
                <LineChart data={chartData}>
                  <CartesianGrid stroke={CHART.grid} strokeDasharray="3 3" />
                  <XAxis dataKey="date" stroke={CHART.axis} style={{ fontSize: 11 }} />
                  <YAxis stroke={CHART.axis} domain={[0, 1]} style={{ fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{
                      background: CHART.tooltipBg,
                      border: `1px solid ${CHART.tooltipBorder}`,
                      borderRadius: 4,
                      fontSize: 11,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11, color: CHART.legendText }} />
                  {plantKeys.slice(0, 30).map((pid, i) => (
                    <Line
                      key={pid}
                      type="monotone"
                      dataKey={pid}
                      stroke={CHART_LINE_COLORS[i % CHART_LINE_COLORS.length]}
                      dot={false}
                      strokeWidth={1}
                      isAnimationActive={false}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-tcip-muted text-[12px]">
                No data. Configure mapping + predictions above, then compute.
              </div>
            )}
          </div>

          <div className="tcip-panel p-4">
            <div className="tcip-heading mb-3">
              Phenology milestones for {trait || "the selected trait"} (the date each declared
              fraction is crossed, per plant): {onset.length} rows
            </div>
            {onset.length > 0 ? (
              <div className="overflow-auto max-h-96">
                <table className="w-full text-[11px]">
                  <thead className="sticky top-0 bg-tcip-panel">
                    <tr className="border-b border-tcip-border">
                      <th className="tcip-th">Plant ID</th>
                      <th className="tcip-th">Accession</th>
                      <th className="tcip-th">N points</th>
                      <th className="tcip-th">Validity</th>
                      {milestoneColumns.map((c) => (
                        <th key={c} className="tcip-th">
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {onset.map((r) => {
                      // Gate the derivation itself (matching the setOnset([]) pattern used above)
                      // rather than a banner, so a plant with any unclassified/missing date shows as
                      // such, not silently blank milestone cells with no explanation.
                      const rowValid =
                        r.n_dates_unclassified === 0 && r.n_dates_missing_images === 0;
                      // "Valid" alone doesn't distinguish real detection data from a plant that was fully
                      // classified/observed but never had a single detection (before emergence, or a
                      // genuinely empty scene): that reads as no observations, not blank cells next
                      // to a reassuring "valid".
                      const neverObserved = rowValid && r.n_observed_dates === 0;
                      return (
                        <tr
                          key={r.plant_id}
                          className="border-t border-tcip-border first:border-t-0"
                        >
                          <td className="py-1.5 pr-3 font-mono">{r.plant_id}</td>
                          <td className="pr-3">{r.accession ?? "—"}</td>
                          <td className="pr-3 tabular-nums">{r.n_dates}</td>
                          <td className="pr-3">
                            {neverObserved ? (
                              <span
                                className="text-tcip-muted"
                                title="Fully classified and fully observed, but no detections on any date, so there is nothing to derive milestones from."
                              >
                                no observations
                              </span>
                            ) : rowValid && provisional ? (
                              // Coverage is complete, but the measurement behind these dates has no
                              // validated operating point. The banner announcing that sits two panels
                              // up and scrolls out of view, so the row must say so where it is read:
                              // a phenology date beside a plain "valid" would be an unearned precision claim.
                              <span
                                className="text-tcip-fp"
                                title="Coverage is complete, but the operating point behind these dates is not validated on disk: provisional, not a deliverable phenotype."
                              >
                                provisional
                              </span>
                            ) : rowValid ? (
                              <span className="text-tcip-muted">valid</span>
                            ) : (
                              <span
                                className="text-tcip-fp"
                                title={`${r.n_dates_unclassified} unclassified date(s), ${r.n_dates_missing_images} missing-image date(s)`}
                              >
                                incomplete
                              </span>
                            )}
                          </td>
                          {milestoneColumns.map((c) => {
                            const date = r[c] as string | null;
                            const bound = r[`${c}_bound`] as string | null;
                            // A left-censored crossing means the first observation already met the
                            // target, so the true date is only an upper bound; a right-censored one
                            // means the last observation still hadn't, so the true date (if any) is
                            // after this one, a lower bound. Rendering either as a plain date is a
                            // precision claim the data does not support.
                            const marker =
                              bound === "left_censored"
                                ? {
                                    symbol: "≤",
                                    className: "text-tcip-fp",
                                    title:
                                      "Left-censored: the first observation already met this target, so the true date is at or before this one.",
                                  }
                                : bound === "right_censored"
                                  ? {
                                      symbol: ">",
                                      className: "text-tcip-fp",
                                      title:
                                        "Right-censored: the last observation still hadn't met this target, so the true date, if any, is after this one.",
                                    }
                                  : bound === "interpolated"
                                    ? {
                                        symbol: "~",
                                        className: "text-tcip-muted",
                                        title: "Interpolated between two observed dates.",
                                      }
                                    : null;
                            return (
                              <td key={c} className="pr-3 tabular-nums">
                                {date ?? "—"}
                                {date && marker && (
                                  <span className={`ml-1 ${marker.className}`} title={marker.title}>
                                    {marker.symbol}
                                  </span>
                                )}
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="text-[11px] text-tcip-muted">Run the compute step to populate.</div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
