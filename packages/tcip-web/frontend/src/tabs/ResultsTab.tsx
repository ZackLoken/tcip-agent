import { useCallback, useEffect, useMemo, useState } from "react";
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
  deliveryRefusalOf,
  operationalizationRefusalOf,
  resultsApi,
  type DeliveryEventRecord,
  type DeliveryRefusal,
  type ExportCountCsvHeaders,
  type MilestoneColumn,
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
import { UNSET_GLYPH } from "@/lib/glyphs";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";
import { CHART, CHART_LINE_COLORS } from "@/tabs/chartTheme";

interface DateRow {
  date: string;
  [plantId: string]: number | string | null;
}

/** The message for a delivery written to `savedPath` whose audit line was not recorded. */
function auditGapExportMessage(savedPath: string, detail: string): string {
  return (
    `The file is already written at ${savedPath}, but its delivery is unrecorded. Exporting ` +
    `again is a second delivery with its own event. ${detail}`
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

  // The mapping this measurement reads, picked from those built on the Setup tab.
  const [mappingName, setMappingName] = useState("");
  const [mappingNames, setMappingNames] = useState<string[]>([]);
  // True unless a computed run reported that its predictions carried no positive-state class.
  const [positiveClassUnassessed, setPositiveClassUnassessed] = useState(false);

  // Dataset tree (dates + the buckets published over each) drives the structured per-date picker
  // below, never a hand-edited JSON blob.
  const [dates, setDates] = useState<string[]>([]);
  const [bucketsByDate, setBucketsByDate] = useState<Record<string, string[]>>({});
  // The name of each bucket published over ``date`` under the dataset root.
  const bucketsFor = (date: string): string[] => bucketsByDate[date] ?? [];
  const [datesError, setDatesError] = useState<string | null>(null);
  const [labelProblem, setLabelProblem] = useState<string | null>(null);
  // The bucket name the breeder picked per date; a date with none picked is not delivered.
  const [dateBucket, setDateBucket] = useState<Record<string, string>>({});
  // The population, typed by the breeder: one plant id per line or comma. The mapping names
  // every plot its plant CSVs carry, which is never the same list, so nothing fills this in.
  const [plantsText, setPlantsText] = useState("");
  const plants = plantsText
    .split(/[\n,]/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  const [curves, setCurves] = useState<PerPlantRow[]>([]);
  const [onset, setOnset] = useState<OnsetRow[]>([]);
  // The milestone columns the trait's own entry names, as the server's producer writes them.
  const [milestoneColumns, setMilestoneColumns] = useState<MilestoneColumn[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  // The exact request the displayed numbers came from: the CSV door recomputes from these inputs
  // rather than being handed the rows, so export and screen share one producer.
  const [lastRequest, setLastRequest] = useState<PhenologyPayload | null>(null);
  // Whether an assessment answers for what is currently displayed, and the gate's sentence when
  // not, so the tables can say so instead of rendering a phenology date as "valid".
  const [unvalidated, setUnvalidated] = useState(false);
  const [unvalidatedReason, setUnvalidatedReason] = useState<string | null>(null);
  // Each projection's digest of the unvalidated result shown, which the breeder's acknowledgment
  // of exporting it binds to.
  const [resultSha256, setResultSha256] = useState<Record<"curves" | "milestones", string> | null>(
    null,
  );
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
  // route serves; an acknowledgment of either is recorded here, never by an MCP tool call.
  const [countKind, setCountKind] = useState<"per_image_count" | "orthomosaic_plant_counts">(
    "per_image_count",
  );
  const [countBucket, setCountBucket] = useState("");
  const [countTrait, setCountTrait] = useState("");
  const [countPlantRegistry, setCountPlantRegistry] = useState("");
  const [countDeliveredPhenotype, setCountDeliveredPhenotype] = useState("");
  const [countCrop, setCountCrop] = useState("");
  const [countPipelineVersion, setCountPipelineVersion] = useState("");
  const [countCanopySubject, setCountCanopySubject] = useState("");
  const [countPositionErrorM, setCountPositionErrorM] = useState("");
  const [countFilename, setCountFilename] = useState("");
  const [countExporting, setCountExporting] = useState(false);
  const [countError, setCountError] = useState<string | null>(null);
  const [countGateRefusal, setCountGateRefusal] = useState<DeliveryRefusal | null>(null);
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
        setBucketsByDate(t.buckets_by_date);
        setDateBucket({});
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

  async function compute(showUnvalidated = false) {
    if (!projectRoot || !datasetRoot) return;
    if (!trait) {
      setError(traitError ?? "Choose a trait.");
      return;
    }
    if (plants.length === 0) {
      setError("Name the plants.");
      return;
    }
    setLoading(true);
    setError(null);
    setOperationalizationRefusal(null);
    setShowAckExport(false);
    setAckReason("");
    try {
      const request: PhenologyPayload = {
        mapping_name: mappingName,
        dataset_root: datasetRoot,
        buckets: Object.values(dateBucket).filter((name) => name),
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
      setUnvalidated(!res.validated);
      setUnvalidatedReason(res.unvalidated_reason);
      setResultSha256(res.result_sha256);
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
      setMilestoneColumns(res.milestones.columns);
    } catch (e) {
      // A refusal naming its own kind is routed by that kind, before any prose is read.
      const refusal = operationalizationRefusalOf(e);
      if (refusal) {
        setOperationalizationRefusal(refusal);
        setCurves([]);
        setOnset([]);
        return;
      }
      // The server refuses an unassessed delivery by default. Surface why, plus the one-click way
      // to see the numbers anyway, marked unvalidated.
      const delivery = deliveryRefusalOf(e);
      if (delivery) {
        setUnvalidatedRefusal(delivery.message);
        setCurves([]);
        setOnset([]);
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setLoading(false);
    }
  }

  async function downloadCsv(payload: "curves" | "milestones", filename: string) {
    if (!lastRequest) return;
    if (unvalidated && (!ackReason.trim() || !resultSha256)) return;
    try {
      const body: ExportCsvPayload = {
        mapping_name: lastRequest.mapping_name,
        dataset_root: lastRequest.dataset_root,
        buckets: lastRequest.buckets,
        trait: lastRequest.trait,
        plants: lastRequest.plants,
        payload,
        filename,
        user: useStore.getState().user,
        acknowledgment:
          unvalidated && resultSha256
            ? { reason: ackReason.trim(), result_sha256: resultSha256[payload] }
            : null,
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

  // Each kind's own required fields, beside the bucket and filename every kind needs.
  const countKindFieldsMissing =
    countKind === "per_image_count"
      ? !countTrait
      : plants.length === 0 || !countPlantRegistry.trim() || !countDeliveredPhenotype.trim();

  async function exportCountCsv() {
    if (!projectRoot || !datasetRoot || !countBucket || !countFilename.trim()) return;
    if (countKindFieldsMissing) return;
    if (countShowAck && !countAckReason.trim()) return;
    setCountExporting(true);
    setCountError(null);
    setCountOperationalizationRefusal(null);
    setCountResultHeaders(null);
    try {
      const delivery: ExportCountCsvPayload["delivery"] =
        countKind === "per_image_count"
          ? {
              kind: "per_image_count",
              dataset_root: datasetRoot,
              bucket: countBucket,
              trait: countTrait,
            }
          : {
              kind: "orthomosaic_plant_counts",
              dataset_root: datasetRoot,
              bucket: countBucket,
              plant_registry: countPlantRegistry,
              delivered_phenotype: countDeliveredPhenotype,
              plants,
              crop: countCrop || undefined,
              pipeline_version: countPipelineVersion || undefined,
              ...(countCanopySubject
                ? {
                    canopy_subject: countCanopySubject,
                    position_error_m:
                      countPositionErrorM === "" ? undefined : Number(countPositionErrorM),
                  }
                : {}),
            };
      const body: ExportCountCsvPayload = {
        delivery,
        filename: countFilename.trim(),
        user: useStore.getState().user,
        acknowledgment:
          countShowAck && countGateRefusal?.result_sha256
            ? { reason: countAckReason.trim(), result_sha256: countGateRefusal.result_sha256 }
            : null,
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
      setCountGateRefusal(null);
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
      const gateRefusal = deliveryRefusalOf(e);
      if (gateRefusal) {
        setCountGateRefusal(gateRefusal);
        setCountShowAck(gateRefusal.result_sha256 !== null);
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
      {traitError && <div className="tcip-problem">{traitError}</div>}
      {unreadableTraits.length > 0 && (
        <div className="tcip-problem">
          These traits&apos; records will not read:{" "}
          {unreadableTraits.map((u) => u.trait).join(", ")}.
        </div>
      )}
      {availableTraits.length > 1 && (
        <div className="tcip-panel p-3 flex items-center gap-2">
          <label className="tcip-label" htmlFor="results-trait">
            Trait
          </label>
          <select
            id="results-trait"
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
        <div className="tcip-problem">No revision of {trait} is confirmed.</div>
      )}

      <DeliveryEventsPanel records={deliveryEvents} loadError={deliveryEventsError} />

      <div className="tcip-panel p-4">
        <h2 className="tcip-heading mb-3">Export</h2>
        <div className="grid grid-cols-[1fr_1fr] gap-3">
          <div className="flex flex-col gap-2">
            <label className="tcip-label" htmlFor="count-kind">
              Kind
            </label>
            <select
              id="count-kind"
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
              <option value="orthomosaic_plant_counts">Per-plant count</option>
            </select>

            <label className="tcip-label" htmlFor="count-bucket">
              Predictions
            </label>
            <select
              id="count-bucket"
              className="tcip-select"
              value={countBucket}
              onChange={(e) => setCountBucket(e.target.value)}
            >
              <option value="">Choose predictions</option>
              {dates.flatMap((d) =>
                bucketsFor(d).map((name) => (
                  <option key={name} value={name}>
                    {`${d} (${name})`}
                  </option>
                )),
              )}
            </select>

            {countKind === "per_image_count" ? (
              <>
                <label className="tcip-label" htmlFor="count-trait">
                  Trait
                </label>
                <select
                  id="count-trait"
                  className="tcip-select"
                  value={countTrait}
                  onChange={(e) => setCountTrait(e.target.value)}
                >
                  <option value="">Choose a trait</option>
                  {availableTraits.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </>
            ) : (
              <>
                <label className="tcip-label" htmlFor="count-plants">
                  Plants
                </label>
                <textarea
                  id="count-plants"
                  className="tcip-input font-mono h-20"
                  value={plantsText}
                  onChange={(e) => setPlantsText(e.target.value)}
                />
                <label className="tcip-label" htmlFor="count-plant-registry">
                  Plant registry
                </label>
                <input
                  id="count-plant-registry"
                  className="tcip-input"
                  value={countPlantRegistry}
                  onChange={(e) => setCountPlantRegistry(e.target.value)}
                />
                <label className="tcip-label" htmlFor="count-delivered-phenotype">
                  Phenotype
                </label>
                <input
                  id="count-delivered-phenotype"
                  className="tcip-input"
                  value={countDeliveredPhenotype}
                  onChange={(e) => setCountDeliveredPhenotype(e.target.value)}
                />
                <label className="tcip-label" htmlFor="count-crop">
                  Crop
                </label>
                <input
                  id="count-crop"
                  className="tcip-input"
                  value={countCrop}
                  onChange={(e) => setCountCrop(e.target.value)}
                />
                <label className="tcip-label" htmlFor="count-pipeline-version">
                  Pipeline version
                </label>
                <input
                  id="count-pipeline-version"
                  className="tcip-input"
                  value={countPipelineVersion}
                  onChange={(e) => setCountPipelineVersion(e.target.value)}
                />
                <label className="tcip-label" htmlFor="count-canopy-subject">
                  Canopy subject
                </label>
                <input
                  id="count-canopy-subject"
                  className="tcip-input"
                  value={countCanopySubject}
                  onChange={(e) => setCountCanopySubject(e.target.value)}
                />
                {countCanopySubject && (
                  <>
                    <label className="tcip-label" htmlFor="count-position-error">
                      Position error (m)
                    </label>
                    <input
                      id="count-position-error"
                      className="tcip-input"
                      type="number"
                      min={0}
                      step="any"
                      value={countPositionErrorM}
                      onChange={(e) => setCountPositionErrorM(e.target.value)}
                    />
                  </>
                )}
              </>
            )}

            <label className="tcip-label" htmlFor="count-filename">
              File name
            </label>
            <input
              id="count-filename"
              className="tcip-input"
              value={countFilename}
              onChange={(e) => setCountFilename(e.target.value)}
            />
          </div>

          <div className="flex flex-col gap-2">
            {countError && <div className="tcip-problem">{countError}</div>}
            {countOperationalizationRefusal && (
              <div className="tcip-problem">{countOperationalizationRefusal.message}</div>
            )}
            {countGateRefusal && (
              <div className="tcip-problem flex flex-col gap-2">
                <div>{countGateRefusal.message}</div>
                {countShowAck && (
                  <input
                    className="tcip-input"
                    placeholder="Reason for delivering unvalidated"
                    value={countAckReason}
                    onChange={(e) => setCountAckReason(e.target.value)}
                  />
                )}
              </div>
            )}
            {countResultHeaders && (
              <div className="text-tcip-muted">
                Saved to {countResultHeaders.savedTo}.
                {countResultHeaders.validated ? "" : " Not validated."}
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
                !countBucket ||
                !countFilename.trim() ||
                countKindFieldsMissing ||
                (countShowAck && !countAckReason.trim())
              }
            >
              {countShowAck ? "Acknowledge and export" : "Export"}
            </button>
          </div>
        </div>
      </div>

      {hasMilestones && (
        <>
          <div className="tcip-panel p-4">
            <h2 className="tcip-heading mb-3">Phenology</h2>
            <div className="grid grid-cols-[1fr_180px] gap-3">
              <div className="flex flex-col gap-1">
                <div className="tcip-label">Predictions by date</div>
                {datesError && (
                  <div className="tcip-problem mb-1">
                    {datesError}{" "}
                    <button className="tcip-btn ml-1" onClick={refreshDatasetTree}>
                      Retry
                    </button>
                  </div>
                )}
                {!datesError && labelProblem && (
                  <div className="tcip-problem mb-1">{labelProblem}</div>
                )}
                {dates.length === 0 ? (
                  !datesError && <div className="text-tcip-muted">No dates with images.</div>
                ) : (
                  <div className="max-h-40 overflow-auto rounded border border-tcip-border">
                    <table className="w-full text-[13px]">
                      <tbody>
                        {dates.map((d) => {
                          const opts = bucketsFor(d);
                          return (
                            <tr key={d} className="border-t border-tcip-border first:border-t-0">
                              <td className="py-1 pl-2 pr-2 font-mono tabular-nums">{d}</td>
                              <td className="py-1 pr-2">
                                <select
                                  className="tcip-select text-[13px] w-full"
                                  value={dateBucket[d] ?? ""}
                                  onChange={(e) =>
                                    setDateBucket((prev) => ({ ...prev, [d]: e.target.value }))
                                  }
                                  disabled={opts.length === 0}
                                  title={
                                    opts.length === 0
                                      ? "No bucket is published over this date"
                                      : "The bucket whose predictions to use for this date"
                                  }
                                >
                                  <option value="">
                                    {opts.length === 0 ? "no predictions" : "(skip)"}
                                  </option>
                                  {opts.map((name) => (
                                    <option key={name} value={name}>
                                      {name}
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
                  Plant mapping
                </label>
                <select
                  id="phenology-mapping"
                  className="tcip-select"
                  value={mappingName}
                  onChange={(e) => setMappingName(e.target.value)}
                >
                  <option value="">Choose a mapping</option>
                  {mappingNames.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
                <label className="tcip-label" htmlFor="phenology-plants">
                  Plants
                </label>
                <textarea
                  id="phenology-plants"
                  className="tcip-input font-mono h-20"
                  value={plantsText}
                  onChange={(e) => setPlantsText(e.target.value)}
                />
                {positiveClassUnassessed && (
                  <div className="tcip-problem">
                    These predictions carry no positive state, so the curves are not a measurement
                    and export is off.
                  </div>
                )}
                {unvalidatedRefusal && (
                  <div className="tcip-problem flex flex-col gap-2">
                    <div>No assessment answers for these predictions.</div>
                    <button
                      className="tcip-btn self-start"
                      onClick={() => void compute(true)}
                      disabled={loading}
                    >
                      Show unvalidated numbers
                    </button>
                  </div>
                )}
                {unvalidated && unvalidatedReason && (
                  <div className="tcip-problem">Not validated: {unvalidatedReason}</div>
                )}
                {datesDelivered.length > 0 && (
                  <div className="text-tcip-muted">
                    {`Delivered dates ${datesDelivered.join(", ")}: ${imagesUnattributed} attributed to no plant`}
                  </div>
                )}
                {(capturesUnverified.length > 0 || plantCsvsUnverified.length > 0) && (
                  <div className="text-tcip-muted">
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
                  Compute
                </button>
                {!unvalidated || !showAckExport ? (
                  <div className="flex gap-1">
                    {unvalidated ? (
                      <button
                        className="tcip-btn flex-1"
                        onClick={() => setShowAckExport(true)}
                        disabled={exportBlocked}
                      >
                        Acknowledge and export
                      </button>
                    ) : (
                      <>
                        <button
                          className="tcip-btn flex-1"
                          onClick={downloadCurvesCsv}
                          disabled={curves.length === 0 || exportBlocked}
                        >
                          Curves CSV
                        </button>
                        <button
                          className="tcip-btn flex-1"
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
                    <input
                      className="tcip-input"
                      placeholder="Reason for delivering unvalidated"
                      value={ackReason}
                      onChange={(e) => setAckReason(e.target.value)}
                    />
                    <div className="flex gap-1">
                      <button
                        className="tcip-btn flex-1"
                        onClick={downloadCurvesCsv}
                        disabled={curves.length === 0 || !ackReason.trim()}
                      >
                        Curves CSV
                      </button>
                      <button
                        className="tcip-btn flex-1"
                        onClick={downloadOnsetCsv}
                        disabled={onset.length === 0 || !ackReason.trim()}
                      >
                        Milestones CSV
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
            {error && <div className="mt-2 tcip-problem">{error}</div>}
            {operationalizationRefusal && (
              <div className="mt-2 tcip-problem">{operationalizationRefusal.message}</div>
            )}
          </div>

          <div className="tcip-panel p-4 h-80">
            <h2 className="tcip-heading mb-3">Positive fraction by date</h2>
            {plantKeys.length > 30 && (
              <div className="text-tcip-muted mb-1">30 of {plantKeys.length} plants shown</div>
            )}
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
              <div className="flex items-center justify-center h-full text-tcip-muted">
                No data.
              </div>
            )}
          </div>

          <div className="tcip-panel p-4">
            <h2 className="tcip-heading mb-3">
              Milestones
              {lastRequest && (
                <span className="text-tcip-muted font-normal">
                  {` ${onset.length} ${onset.length === 1 ? "row" : "rows"}`}
                </span>
              )}
            </h2>
            {onset.length > 0 ? (
              <div className="overflow-auto max-h-96">
                <table className="w-full text-[13px]">
                  <thead className="sticky top-0 bg-tcip-panel">
                    <tr className="border-b border-tcip-border">
                      <th className="tcip-th">Plant ID</th>
                      <th className="tcip-th">Accession</th>
                      <th className="tcip-th">N points</th>
                      <th className="tcip-th">Validity</th>
                      {milestoneColumns.map((c) => (
                        <th key={c.date} className="tcip-th">
                          {c.date}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {onset.map((r) => {
                      // A plant with any unassessed or missing date shows as such, never as blank cells.
                      const rowValid = r.complete;
                      // Every date assessed and observed with no detection reads as no observations.
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
                              <span className="text-tcip-muted" title="No detections on any date">
                                no observations
                              </span>
                            ) : rowValid && unvalidated ? (
                              <span className="tcip-problem" title="No assessment answers">
                                unvalidated
                              </span>
                            ) : rowValid ? (
                              <span className="text-tcip-muted">valid</span>
                            ) : (
                              <span
                                className="tcip-problem"
                                title={`${r.n_dates_unclassified} unclassified date(s), ${r.n_dates_missing_images} missing-image date(s)`}
                              >
                                incomplete
                              </span>
                            )}
                          </td>
                          {milestoneColumns.map((c) => {
                            const date = r[c.date] as string | null;
                            const bound = r[c.bound] as string | null;
                            // A censored crossing is a bound on the true date, never the date.
                            const marker =
                              bound === "left_censored"
                                ? {
                                    symbol: "≤",
                                    className: "tcip-problem",
                                    title: "Before first observation",
                                  }
                                : bound === "right_censored"
                                  ? {
                                      symbol: ">",
                                      className: "tcip-problem",
                                      title: "After last observation",
                                    }
                                  : bound === "interpolated"
                                    ? {
                                        symbol: "~",
                                        className: "text-tcip-muted",
                                        title: "Interpolated",
                                      }
                                    : null;
                            return (
                              <td key={c.date} className="pr-3 tabular-nums">
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
              <div className="text-tcip-muted">No rows.</div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
