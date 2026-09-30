import { type ReactNode, useCallback, useEffect, useMemo, useState } from "react";
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
import { committedOf } from "@/api/http";
import {
  deliveryGateRefusalOf,
  operationalizationRefusalOf,
  resultsApi,
  type DeliveryEventRecord,
  type DeliveryGateRefusal,
  type ExportCountCsvHeaders,
  type OperationalizationRefusal,
  type OnsetRow,
  type PerPlantRow,
  type ServedTraitRecord,
} from "@/api/inference";
import type {
  ExportCountCsvPayload,
  ExportCsvPayload,
  PhenologyPayload,
} from "@/api/types.generated";
import { DeliveryEventsPanel } from "@/components/DeliveryEventsPanel";
import { TabHeading } from "@/components/TabHeading";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { CHART, CHART_LINE_COLORS } from "@/tabs/chartTheme";

interface DateRow {
  date: string;
  [plantId: string]: number | string | null;
}

/** A note whose remedy is on the Setup tab, with the one action that opens it. */
function SetupTabNote({ children }: { children: ReactNode }) {
  return (
    <div className="tcip-panel p-3 text-[11px] text-tcip-fp flex items-center gap-2">
      <div>{children}</div>
      <button
        className="tcip-btn text-[11px]"
        onClick={() => useStore.getState().setActiveTab("setup")}
      >
        Open the Setup tab
      </button>
    </div>
  );
}

/** The message for a delivery written to `savedPath` whose audit line was not recorded. */
function auditGapExportMessage(savedPath: string, detail: string): string {
  return (
    `The file is already written at ${savedPath}, but its delivery is unrecorded. Exporting ` +
    "again is a second delivery with its own event; supersede_delivery is the remedy for the " +
    `first. ${detail}`
  );
}

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
  const projectRoot = useStore(selectProjectRoot);
  const datasetRoot = dataset.dataset_root;
  // An acknowledged export is refused server-side with no user set; read reactively so the
  // acknowledged-export buttons disable themselves before a breeder types a reason for nothing.
  const user = useStore((s) => s.user);

  // The mapping this measurement reads, picked from those built on the Setup tab.
  const [mappingName, setMappingName] = useState("");
  const [mappingNames, setMappingNames] = useState<string[]>([]);
  // True unless a computed run reported that its predictions carried no positive-state class.
  const [positiveClassUnassessed, setPositiveClassUnassessed] = useState(false);

  // Dataset tree (dates + which models actually have predictions per date) drives the structured
  // per-date picker below, never a hand-edited JSON blob; models_with_predictions is the same
  // primitive the backend already computes this from, via api.dataset.tree.
  const [dates, setDates] = useState<string[]>([]);
  const [modelsByDate, setModelsByDate] = useState<Record<string, string[]>>({});
  const [predictionDirs, setPredictionDirs] = useState<Record<string, Record<string, string>>>({});
  const [datesError, setDatesError] = useState<string | null>(null);
  const [labelProblem, setLabelProblem] = useState<string | null>(null);
  // The model picked per date; "" means "skip this date" (dropped before compute()).
  const [dateModel, setDateModel] = useState<Record<string, string>>({});
  // The population, typed by the breeder: one plant id per line or comma. The mapping names
  // every plot its plant CSVs carry, which is never the same list, so nothing fills this in.
  const [plantsText, setPlantsText] = useState("");
  const plants = plantsText
    .split(/[\n,]/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  const [curves, setCurves] = useState<PerPlantRow[]>([]);
  const [onset, setOnset] = useState<OnsetRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  // The exact request the displayed numbers came from: the CSV door recomputes from these inputs
  // rather than being handed the rows, so export and screen share one producer.
  const [lastRequest, setLastRequest] = useState<PhenologyPayload | null>(null);
  // Reconciled evidence for what is currently displayed. `unvalidated` is true whenever a dimension
  // lacked on-disk backing, so the tables can say so instead of rendering a phenology date as "valid".
  const [unvalidated, setUnvalidated] = useState(false);
  const [validity, setValidity] = useState<Record<string, string>>({});
  // What the mapping's own delivery-time check could not verify, shown beside the numbers rather
  // than only in the exported CSV.
  const [capturesUnverified, setCapturesUnverified] = useState<string[]>([]);
  const [plantCsvsUnverified, setPlantCsvsUnverified] = useState<string[]>([]);
  const [datesDelivered, setDatesDelivered] = useState<string[]>([]);
  const [imagesUnattributed, setImagesUnattributed] = useState(0);
  const [unvalidatedRefusal, setUnvalidatedRefusal] = useState<string | null>(null);
  // The reason field for acknowledging and exporting an unvalidated measurement; shown once the
  // breeder opts into the acknowledged-export flow, cleared on every fresh compute.
  const [showAckExport, setShowAckExport] = useState(false);
  const [ackReason, setAckReason] = useState("");

  const [operationalizationRefusal, setOperationalizationRefusal] =
    useState<OperationalizationRefusal | null>(null);

  // What has shipped from this project: read-only, no confirm/withdraw state to carry.
  const [deliveryEvents, setDeliveryEvents] = useState<DeliveryEventRecord[]>([]);
  const [deliveryEventsError, setDeliveryEventsError] = useState<string | null>(null);

  // Count export: per_image_count and orthomosaic_plant_counts, the two delivery kinds this
  // route serves, reachable here only (an MCP tool call builds no acknowledgment for either).
  const [countKind, setCountKind] = useState<"per_image_count" | "orthomosaic_plant_counts">(
    "per_image_count",
  );
  const [countDate, setCountDate] = useState("");
  const [countModel, setCountModel] = useState("");
  const [countTrait, setCountTrait] = useState("");
  const [countRasterPath, setCountRasterPath] = useState("");
  const [countPlantRegistry, setCountPlantRegistry] = useState("");
  const [countDeliveredPhenotype, setCountDeliveredPhenotype] = useState("");
  const [countCrop, setCountCrop] = useState("");
  const [countPipelineVersion, setCountPipelineVersion] = useState("");
  const [countCanopySubject, setCountCanopySubject] = useState("");
  const [countFilename, setCountFilename] = useState("");
  const [countExporting, setCountExporting] = useState(false);
  const [countError, setCountError] = useState<string | null>(null);
  const [countGateRefusal, setCountGateRefusal] = useState<DeliveryGateRefusal | null>(null);
  const [countOperationalizationRefusal, setCountOperationalizationRefusal] =
    useState<OperationalizationRefusal | null>(null);
  const [countAckReason, setCountAckReason] = useState("");
  const [countShowAck, setCountShowAck] = useState(false);
  const [countResultHeaders, setCountResultHeaders] = useState<ExportCountCsvHeaders | null>(null);

  // The trait a delivery is computed for, resolved from this project's own registered traits
  // (never assumed): auto-selected when there is exactly one, left blank (with an explicit
  // error, not a silent guess) when there are zero, offered as a choice when there are several.
  const [traitRecords, setTraitRecords] = useState<ServedTraitRecord[]>([]);
  const availableTraits = traitRecords.map((r) => r.trait);
  const [trait, setTrait] = useState("");
  const [traitError, setTraitError] = useState<string | null>(null);
  // Trait records that failed to load, so a breeder can tell "nothing registered" from
  // "something is registered but broken" instead of the two looking identical.
  const [unreadableTraits, setUnreadableTraits] = useState<{ trait: string; reason: string }[]>([]);

  useEffect(() => {
    if (!projectRoot) return;
    setTrait("");
    setTraitError(null);
    void resultsApi
      .traits()
      .then((res) => {
        setTraitRecords(res.traits);
        setUnreadableTraits(res.unreadable);
        if (res.traits.length === 0) {
          setTraitError("No trait is registered for this project yet.");
        } else if (res.traits.length === 1) {
          setTrait(res.traits[0].trait);
        }
      })
      .catch((e) => {
        setTraitRecords([]);
        setUnreadableTraits([]);
        setTraitError(
          `Could not load this project's registered traits: ${e instanceof Error ? e.message : String(e)}`,
        );
      });
  }, [projectRoot]);

  useEffect(() => {
    if (!projectRoot) return;
    void resultsApi
      .listPlantMappings()
      .then((res) => setMappingNames(res.names))
      .catch(() => setMappingNames([]));
  }, [projectRoot]);

  useEffect(() => {
    if (!projectRoot) return;
    void resultsApi
      .deliveryEvents()
      .then((res) => {
        setDeliveryEvents(res.records);
        setDeliveryEventsError(null);
      })
      .catch((e) => {
        setDeliveryEvents([]);
        setDeliveryEventsError(
          `Could not load what this project has shipped: ${e instanceof Error ? e.message : String(e)}`,
        );
      });
  }, [projectRoot]);

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
        setLabelProblem(t.label_problem);
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

  async function compute(showUnvalidated = false) {
    if (!projectRoot) return;
    if (!trait) {
      setError(traitError ?? "Pick a trait before computing.");
      return;
    }
    if (plants.length === 0) {
      setError("Name the plants to measure (one plant id per line) before computing.");
      return;
    }
    setLoading(true);
    setError(null);
    setOperationalizationRefusal(null);
    setShowAckExport(false);
    setAckReason("");
    try {
      const predsMap: Record<string, string> = {};
      for (const d of dates) {
        const dir = predDirFor(d, dateModel[d] ?? "");
        if (dir) predsMap[d] = dir;
      }
      const request = {
        mapping_name: mappingName,
        predictions_by_date: predsMap,
        trait,
        plants,
        show_unvalidated: showUnvalidated,
      };
      setLastRequest(request);
      // One request computes both projections from one server-side measurement: a milestone
      // date and the curve it was read off cannot disagree.
      const res = await resultsApi.phenologyMeasurement(request);
      // The numbers and the evidence that qualifies them arrive together, so the tables below can
      // never render an unvalidated phenology measurement as though it were a delivery.
      setUnvalidated(res.has_unvalidated_dimensions);
      setValidity(res.validated);
      setCapturesUnverified(res.captures_unverified ?? []);
      setPlantCsvsUnverified(res.plant_csvs_unverified ?? []);
      setDatesDelivered(res.dates_delivered ?? []);
      setImagesUnattributed(res.images_unattributed ?? 0);
      setUnvalidatedRefusal(null);
      const unclassified = res.positive_class_assessed === false;
      setPositiveClassUnassessed(unclassified);
      setCurves(res.curves.rows ?? []);
      // No positive-state class: not a phenology measurement, so the milestones projection is
      // dropped rather than shown (belt-and-braces with the disabled export buttons).
      setOnset(unclassified ? [] : (res.milestones.rows ?? []));
    } catch (e) {
      // A refusal naming its own kind is routed by that kind, before any prose is read.
      const refusal = operationalizationRefusalOf(e);
      if (refusal) {
        setOperationalizationRefusal(refusal);
        setCurves([]);
        setOnset([]);
        return;
      }
      // The server refuses unvalidated evidence by default. Surface why, plus the one-click way
      // to see the numbers anyway, marked with their unvalidated dimensions.
      const detail = e instanceof Error ? e.message : String(e);
      if (!showUnvalidated && /unvalidated|not validated/i.test(detail)) {
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
    if (unvalidated && !ackReason.trim()) return;
    try {
      const body: ExportCsvPayload = {
        mapping_name: lastRequest.mapping_name,
        predictions_by_date: lastRequest.predictions_by_date,
        trait: lastRequest.trait,
        plants: lastRequest.plants,
        payload,
        filename,
        user: useStore.getState().user || undefined,
        acknowledgment: unvalidated ? { reason: ackReason.trim() } : null,
      };
      const blob = await resultsApi.downloadCsv(body);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      const committed = committedOf<{ saved_path: string }>(e);
      if (committed) {
        useStore
          .getState()
          .pushToast(
            auditGapExportMessage(committed.saved_path, e instanceof Error ? e.message : String(e)),
          );
        return;
      }
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

  const countPredictionsDir = (countModel && predictionDirs[countDate]?.[countModel]) || "";
  // Each kind's own required fields, beside the bucket and filename every kind needs.
  const countKindFieldsMissing =
    countKind === "per_image_count"
      ? !countTrait
      : !countRasterPath.trim() || !countPlantRegistry.trim() || !countDeliveredPhenotype.trim();

  async function exportCountCsv() {
    if (!projectRoot || !countPredictionsDir || !countFilename.trim()) return;
    if (countKindFieldsMissing) return;
    if (countShowAck && !countAckReason.trim()) return;
    setCountExporting(true);
    setCountError(null);
    setCountGateRefusal(null);
    setCountOperationalizationRefusal(null);
    setCountResultHeaders(null);
    try {
      const delivery: ExportCountCsvPayload["delivery"] =
        countKind === "per_image_count"
          ? { kind: "per_image_count", predictions_dir: countPredictionsDir, trait: countTrait }
          : {
              kind: "orthomosaic_plant_counts",
              predictions_dir: countPredictionsDir,
              raster_path: countRasterPath,
              plant_registry: countPlantRegistry,
              delivered_phenotype: countDeliveredPhenotype,
              crop: countCrop || undefined,
              pipeline_version: countPipelineVersion || undefined,
              canopy_subject: countCanopySubject || undefined,
            };
      const body: ExportCountCsvPayload = {
        delivery,
        filename: countFilename.trim(),
        user: useStore.getState().user || undefined,
        acknowledgment: countShowAck ? { reason: countAckReason.trim() } : null,
      };
      const { blob, headers } = await resultsApi.downloadCountCsv(body);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = countFilename.trim();
      a.click();
      URL.revokeObjectURL(url);
      setCountResultHeaders(headers);
      setCountShowAck(false);
    } catch (e) {
      const committed = committedOf<{ saved_path: string }>(e);
      if (committed) {
        setCountError(
          auditGapExportMessage(committed.saved_path, e instanceof Error ? e.message : String(e)),
        );
        return;
      }
      const opRefusal = operationalizationRefusalOf(e);
      if (opRefusal) {
        setCountOperationalizationRefusal(opRefusal);
        return;
      }
      const gateRefusal = deliveryGateRefusalOf(e);
      if (gateRefusal) {
        setCountGateRefusal(gateRefusal);
        setCountShowAck(true);
        return;
      }
      setCountError(e instanceof Error ? e.message : String(e));
    } finally {
      setCountExporting(false);
    }
  }

  // Never export a CSV built on predictions with no positive-state class, mirroring
  // deliver_phenology_milestones. An unvalidated measurement exports once acknowledged (below).
  const exportBlocked = positiveClassUnassessed;
  const downloadOnsetCsv = () => {
    if (exportBlocked) return;
    void downloadCsv("milestones", `${trait}_phenology.csv`);
  };
  const downloadCurvesCsv = () => {
    if (exportBlocked) return;
    // A curve export is the same delivered phenology measurement as the milestone one, just
    // un-summarized: same producer, same gate.
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
  // calibrated run_inference + calibrate_classifier_operating_point stand behind it (see
  // results.py's _refusal). Hand that off to the agent instead of leaving the tool names on
  // screen with no next step.
  function calibrationRequest(detail: string | null): string {
    const dims = unvalidatedDims.length > 0 ? unvalidatedDims.join(", ") : "the operating point";
    const subject = trait ? `the "${trait}" trait` : "this trait";
    return (
      `Phenology delivery for ${subject} is blocked: ${dims} not validated on disk. ` +
      "Please produce the predictions via a calibrated run_inference and calibrate the " +
      "classifier via calibrate_classifier_operating_point so this validates, then let me know " +
      "when it's ready so I can recompute here." +
      (detail ? ` Details from the app: ${detail}` : "")
    );
  }

  // The breeder can't propose a trait revision from the GUI, so a trait with no milestone
  // fractions needs a way forward rather than an empty tab.
  function milestoneAbsenceRequest(): string {
    return (
      `The Results tab has nothing to compute for the "${trait}" trait: its confirmed revision ` +
      "declares no milestone fractions. Please tell me what this trait's measurement delivers " +
      "and how I get it, and propose a revision if milestones are part of it."
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

  // The entry a delivery reads for the selected trait: its latest confirmed revision's, or null.
  const selectedRecord = traitRecords.find((r) => r.trait === trait);
  const confirmedEntry =
    selectedRecord?.revisions.find((r) => r.number === selectedRecord.latest_confirmed)?.entry ??
    null;
  // Curves and milestones are only meaningful for a trait whose confirmed revision declares the
  // fractions they are read off. Nothing else on this tab depends on it.
  const hasMilestones = (confirmedEntry?.milestone_fractions ?? []).length > 0;

  return (
    <div className="flex-1 overflow-auto p-4 flex flex-col gap-4">
      <TabHeading tab="results" />
      {traitError && <div className="tcip-panel p-3 text-[11px] text-tcip-fp">{traitError}</div>}
      {unreadableTraits.length > 0 && (
        <SetupTabNote>
          These traits' records will not read: {unreadableTraits.map((u) => u.trait).join(", ")}.
        </SetupTabNote>
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
      {trait && selectedRecord && !confirmedEntry && (
        <SetupTabNote>
          No revision of {trait} is confirmed, so nothing delivers under it.
        </SetupTabNote>
      )}

      <DeliveryEventsPanel records={deliveryEvents} loadError={deliveryEventsError} />

      <div className="tcip-panel p-4">
        <div className="tcip-heading mb-3">Count export</div>
        <p className="text-[10px] text-tcip-muted mb-2">
          A per-image count (a bucket, no plant identity) or a per-plant count through the
          orthomosaic composition (a bucket plus a registered plant registry). The ordinal and
          regression aggregates (their per-plant strategy is an agent choice) and the per-plant
          walked-capture count (its across-dates strategy is not yet a structured, breeder-confirmed
          field) have no route here; deliver those through the agent.
        </p>
        <div className="grid grid-cols-[1fr_1fr] gap-3">
          <div className="flex flex-col gap-2">
            <label className="tcip-label">Kind</label>
            <select
              className="tcip-select"
              value={countKind}
              onChange={(e) => {
                setCountKind(e.target.value as typeof countKind);
                setCountGateRefusal(null);
                setCountOperationalizationRefusal(null);
                setCountShowAck(false);
              }}
            >
              <option value="per_image_count">Per-image count</option>
              <option value="orthomosaic_plant_counts">Per-plant count (orthomosaic)</option>
            </select>

            <label className="tcip-label">Prediction bucket</label>
            <select
              className="tcip-select"
              value={countDate && countModel ? `${countDate} ${countModel}` : ""}
              onChange={(e) => {
                const [d, m] = e.target.value.split(" ");
                setCountDate(d ?? "");
                setCountModel(m ?? "");
              }}
            >
              <option value="">Choose a bucket…</option>
              {dates.flatMap((d) =>
                (modelsByDate[d] ?? []).map((m) => (
                  <option key={`${d} ${m}`} value={`${d} ${m}`}>
                    {`${d} (${m})`}
                  </option>
                )),
              )}
            </select>

            {countKind === "per_image_count" ? (
              <>
                <label className="tcip-label">Trait</label>
                <select
                  className="tcip-select"
                  value={countTrait}
                  onChange={(e) => setCountTrait(e.target.value)}
                >
                  <option value="">Choose a trait…</option>
                  {availableTraits.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </>
            ) : (
              <>
                <label className="tcip-label">Raster path</label>
                <input
                  className="tcip-input"
                  value={countRasterPath}
                  onChange={(e) => setCountRasterPath(e.target.value)}
                  placeholder="the georeferenced raster the bucket was produced from"
                />
                <label className="tcip-label">Plant registry (registered by name)</label>
                <input
                  className="tcip-input"
                  value={countPlantRegistry}
                  onChange={(e) => setCountPlantRegistry(e.target.value)}
                />
                <label className="tcip-label">Delivered phenotype</label>
                <input
                  className="tcip-input"
                  value={countDeliveredPhenotype}
                  onChange={(e) => setCountDeliveredPhenotype(e.target.value)}
                />
                <div className="grid grid-cols-3 gap-2">
                  <input
                    className="tcip-input"
                    value={countCrop}
                    onChange={(e) => setCountCrop(e.target.value)}
                    placeholder="crop"
                  />
                  <input
                    className="tcip-input"
                    value={countPipelineVersion}
                    onChange={(e) => setCountPipelineVersion(e.target.value)}
                    placeholder="pipeline_version"
                  />
                  <input
                    className="tcip-input"
                    value={countCanopySubject}
                    onChange={(e) => setCountCanopySubject(e.target.value)}
                    placeholder="canopy_subject (optional)"
                  />
                </div>
              </>
            )}

            <label className="tcip-label">Filename</label>
            <input
              className="tcip-input"
              value={countFilename}
              onChange={(e) => setCountFilename(e.target.value)}
              placeholder="stem_count.csv"
            />
          </div>

          <div className="flex flex-col gap-2">
            {countError && <div className="text-[11px] text-tcip-fp">{countError}</div>}
            {countOperationalizationRefusal && (
              <SetupTabNote>{countOperationalizationRefusal.message}</SetupTabNote>
            )}
            {countGateRefusal && (
              <div className="text-[11px] text-tcip-fp border border-tcip-fp/40 rounded p-2 flex flex-col gap-2">
                <div>{countGateRefusal.message}</div>
                {!user.trim() && (
                  <div>
                    Set your name on the workspace page before delivering an acknowledged export.
                  </div>
                )}
                <input
                  className="tcip-input text-[11px]"
                  placeholder="Reason for delivering unvalidated"
                  value={countAckReason}
                  onChange={(e) => setCountAckReason(e.target.value)}
                />
              </div>
            )}
            {countResultHeaders && (
              <div className="text-[11px] text-tcip-muted">
                Saved to {countResultHeaders.savedTo}.
                {countResultHeaders.unvalidatedDimensions
                  ? ` Unvalidated: ${countResultHeaders.unvalidatedDimensions}.`
                  : ""}
                {countResultHeaders.acknowledgedBy
                  ? ` Acknowledged by ${countResultHeaders.acknowledgedBy}.`
                  : ""}
              </div>
            )}
            <button
              className="tcip-btn-primary"
              onClick={() => void exportCountCsv()}
              disabled={
                countExporting ||
                !countPredictionsDir ||
                !countFilename.trim() ||
                countKindFieldsMissing ||
                (countShowAck && (!countAckReason.trim() || !user.trim()))
              }
            >
              {countExporting ? "Exporting…" : countShowAck ? "Acknowledge and export" : "Export"}
            </button>
          </div>
        </div>
      </div>

      {trait && confirmedEntry && !hasMilestones && (
        <div className="tcip-panel p-4 flex flex-col gap-2">
          <div className="tcip-heading">Nothing to compute here for {trait}</div>
          <p className="text-[11px] text-tcip-muted">
            This trait's confirmed revision declares no milestone fractions, so there are no curves
            or milestones to compute for it.
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
                {!datesError && labelProblem && (
                  <div className="text-[11px] text-tcip-fp mb-1">{labelProblem}</div>
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
                <label className="tcip-label" htmlFor="phenology-mapping">
                  Plant mapping (built on the Setup tab)
                </label>
                <select
                  id="phenology-mapping"
                  className="tcip-select text-[11px]"
                  value={mappingName}
                  onChange={(e) => setMappingName(e.target.value)}
                >
                  <option value="">Choose a mapping…</option>
                  {mappingNames.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
                <label className="tcip-label" htmlFor="phenology-plants">
                  Plants to measure (one id per line)
                </label>
                <textarea
                  id="phenology-plants"
                  className="tcip-input text-[11px] font-mono h-20"
                  value={plantsText}
                  onChange={(e) => setPlantsText(e.target.value)}
                  placeholder={"PLOT-01\nPLOT-02"}
                  title="The plant ids this measurement delivers, one row each; the mapping's own plot list is not the population"
                />
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
                      At least one dimension behind this measurement (the count operating point, the
                      positive-state classifier, or the tile scale; see detail below) has no
                      validated evidence on disk, so this is not yet a deliverable phenology
                      measurement. Calibrate first, or look at the numbers with their unvalidated
                      dimensions marked and acknowledge and export them anyway.
                    </div>
                    <div className="text-tcip-muted">{unvalidatedRefusal}</div>
                    <div className="flex gap-2">
                      <button
                        className="tcip-btn text-[11px] self-start"
                        onClick={() => void compute(true)}
                        disabled={loading}
                      >
                        Show unvalidated numbers
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
                {unvalidated && (
                  <div className="text-[11px] text-tcip-fp border border-tcip-fp/40 rounded p-2 flex flex-col gap-2">
                    <div>
                      Unvalidated dimensions {unvalidatedDims.join(", ") || "unknown"}. Calibrate to
                      deliver a validated phenotype, or acknowledge and export it unvalidated below.
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
                {datesDelivered.length > 0 && (
                  <div className="text-[11px] text-tcip-muted">
                    {`Delivered dates ${datesDelivered.join(", ")}: ${imagesUnattributed} attributed to no plant`}
                  </div>
                )}
                {(capturesUnverified.length > 0 || plantCsvsUnverified.length > 0) && (
                  <div className="text-[11px] text-tcip-muted border border-tcip-border rounded p-2">
                    Not verified at delivery time:
                    {capturesUnverified.length > 0 && ` ${capturesUnverified.join(", ")}`}
                    {capturesUnverified.length > 0 && plantCsvsUnverified.length > 0 && ";"}
                    {plantCsvsUnverified.length > 0 &&
                      ` plant CSV(s) ${plantCsvsUnverified.join(", ")} (path not found)`}
                  </div>
                )}
                <button
                  className="tcip-btn-primary"
                  onClick={() => void compute()}
                  disabled={loading}
                >
                  {loading
                    ? "Verifying the mapping's captures…"
                    : "Compute curves + milestone dates"}
                </button>
                <p className="text-[10px] text-tcip-muted">
                  One computed measurement in two shapes: every (plant, date) point, or the
                  milestone dates read off it.
                </p>
                <p className="text-[10px] text-tcip-muted">
                  A phenology milestone delivery here, and a per-image or per-plant-orthomosaic
                  count in the section below, can each be acknowledged and exported unvalidated.
                  Every other count kind (ordinal, regression, a per-plant walked-capture count) has
                  no acknowledged route yet and needs a validated operating point before it can be
                  delivered at all.
                </p>
                {!unvalidated || !showAckExport ? (
                  <div className="flex gap-1">
                    {unvalidated ? (
                      <button
                        className="tcip-btn flex-1 text-[11px]"
                        onClick={() => setShowAckExport(true)}
                        disabled={exportBlocked}
                      >
                        Acknowledge and export
                      </button>
                    ) : (
                      <>
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
                      </>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col gap-1">
                    {!user.trim() && (
                      <div className="text-[11px] text-tcip-fp">
                        Set your name on the workspace page before delivering an acknowledged
                        export: the recorded name is the one this session states, not an
                        authenticated identity.
                      </div>
                    )}
                    <input
                      className="tcip-input text-[11px]"
                      placeholder="Reason for delivering unvalidated"
                      value={ackReason}
                      onChange={(e) => setAckReason(e.target.value)}
                    />
                    <div className="flex gap-1">
                      <button
                        className="tcip-btn flex-1 text-[11px]"
                        onClick={downloadCurvesCsv}
                        disabled={curves.length === 0 || !ackReason.trim() || !user.trim()}
                        title={
                          !user.trim() ? "set your name on the workspace page first" : undefined
                        }
                      >
                        Curves CSV
                      </button>
                      <button
                        className="tcip-btn flex-1 text-[11px]"
                        onClick={downloadOnsetCsv}
                        disabled={onset.length === 0 || !ackReason.trim() || !user.trim()}
                        title={
                          !user.trim() ? "set your name on the workspace page first" : undefined
                        }
                      >
                        Milestones CSV
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
            {error && <div className="mt-2 text-[11px] text-tcip-fp">{error}</div>}
            {operationalizationRefusal && (
              <div className="mt-2">
                <SetupTabNote>{operationalizationRefusal.message}</SetupTabNote>
              </div>
            )}
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
                          <td className="pr-3">{r.accession ?? UNSET_GLYPH}</td>
                          <td className="pr-3 tabular-nums">{r.n_dates}</td>
                          <td className="pr-3">
                            {neverObserved ? (
                              <span
                                className="text-tcip-muted"
                                title="Fully classified and fully observed, but no detections on any date, so there is nothing to derive milestones from."
                              >
                                no observations
                              </span>
                            ) : rowValid && unvalidated ? (
                              // Coverage is complete, but the measurement behind these dates has no
                              // validated operating point. The banner announcing that sits two panels
                              // up and scrolls out of view, so the row must say so where it is read:
                              // a phenology date beside a plain "valid" would be an unearned precision claim.
                              <span
                                className="text-tcip-fp"
                                title="Coverage is complete, but the operating point behind these dates is not validated on disk: unvalidated, not a deliverable phenotype."
                              >
                                unvalidated
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
                                {date ?? UNSET_GLYPH}
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
