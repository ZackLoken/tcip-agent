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
import { committedOf, messageOf, refusalOf } from "@/api/http";
import { resultsApi } from "@/api/inference";
import {
  type CountExport,
  type DatasetTree,
  DELIVERY,
  type DeliveryEvents,
  type DeliveryRefusal,
  type ExportCountCsvPayload,
  type ExportCsvPayload,
  type MappingNames,
  OPERATIONALIZATION,
  type OperationalizationRefusal,
  type PhenologyAnswer,
  type PhenologyPayload,
  type PlantMappingDisclosure,
  type Projection,
  type SavedExport,
  type TraitsListing,
} from "@/api/types.generated";
import { DeliveryEventsPanel } from "@/components/DeliveryEventsPanel";
import { TabHeading } from "@/components/TabHeading";
import { deliveredDatesLine } from "@/lib/disclosureLines";
import { UNSET_GLYPH } from "@/lib/glyphs";
import { bucketsForDate } from "@/lib/openProject";
import { useStore } from "@/store";
import { selectDatasetRoot, selectProjectRoot } from "@/store/slices/gui";
import { CHART, CHART_LINE_COLORS } from "@/tabs/chartTheme";

interface CaptureRow {
  capture: string;
  [plantId: string]: number | string | null;
}

/** The message for a delivery written to `savedPath` whose audit line was not recorded. */
function auditGapExportMessage(savedPath: string, detail: string): string {
  return (
    `The file is already written at ${savedPath}, but its delivery is unrecorded. Exporting ` +
    `again is a second delivery with its own event. ${detail}`
  );
}

/** What a measurement's plant mapping disclosed: its delivered dates and what it could not verify
 *  at delivery time. */
function MappingDisclosureLines({ disclosure }: { disclosure: PlantMappingDisclosure }) {
  const { dates_delivered, captures_unverified, plant_csvs_unverified } = disclosure;
  return (
    <>
      {dates_delivered.length > 0 && (
        <div className="text-tcip-muted">{deliveredDatesLine(disclosure)}</div>
      )}
      {(captures_unverified.length > 0 || plant_csvs_unverified.length > 0) && (
        <div className="text-tcip-muted">
          Not verified at delivery time:
          {captures_unverified.length > 0 && ` ${captures_unverified.join(", ")}`}
          {captures_unverified.length > 0 && plant_csvs_unverified.length > 0 && ";"}
          {plant_csvs_unverified.length > 0 &&
            ` plant CSV(s) ${plant_csvs_unverified.join(", ")} (path not found)`}
        </div>
      )}
    </>
  );
}

export function ResultsTab() {
  const projectRoot = useStore(selectProjectRoot);
  const datasetRoot = useStore(selectDatasetRoot);

  // The mapping this measurement reads, picked from those built on the Setup tab.
  const [mappingName, setMappingName] = useState("");
  const [mappings, setMappings] = useState<MappingNames | null>(null);

  // Dataset tree (dates + the buckets published over each) drives the structured per-date picker
  // below.
  const [tree, setTree] = useState<DatasetTree | null>(null);
  const [datesError, setDatesError] = useState<string | null>(null);
  // The bucket name the breeder picked per date; a date with none picked is not delivered.
  const [dateBucket, setDateBucket] = useState<Record<string, string>>({});
  // The population, typed by the breeder: one plant id per line or comma.
  const [plantsText, setPlantsText] = useState("");
  const plants = plantsText
    .split(/[\n,]/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  // The measurement shown, whole, and the inputs it was computed from, which the CSV door
  // recomputes from.
  const [answer, setAnswer] = useState<PhenologyAnswer | null>(null);
  const [lastInputs, setLastInputs] = useState<Omit<PhenologyPayload, "show_unvalidated"> | null>(
    null,
  );
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const curves = useMemo(() => answer?.curves ?? [], [answer]);
  // A run whose predictions carry no positive-state class is not a phenology measurement, so it
  // carries no milestones.
  const positiveClassUnassessed = answer !== null && !answer.positive_class_assessed;
  const onset = answer !== null && answer.positive_class_assessed ? answer.milestones : [];
  // The gate's refusal when no assessment answers for what is currently displayed, with each
  // projection's digest the breeder's acknowledgment of exporting it binds to.
  const shownUnvalidated = answer?.unvalidated ?? null;
  const unvalidated = shownUnvalidated !== null;
  const [unvalidatedRefusal, setUnvalidatedRefusal] = useState<DeliveryRefusal | null>(null);
  // The reason field for acknowledging and exporting an unvalidated measurement; shown once the
  // breeder opts into the acknowledged-export flow, cleared on every fresh compute.
  const [showAckExport, setShowAckExport] = useState(false);
  const [ackReason, setAckReason] = useState("");

  const [operationalizationRefusal, setOperationalizationRefusal] =
    useState<OperationalizationRefusal | null>(null);

  // What has shipped from this project: read-only, no confirm/withdraw state to carry.
  const [deliveryEvents, setDeliveryEvents] = useState<DeliveryEvents | null>(null);
  const [deliveryEventsError, setDeliveryEventsError] = useState<string | null>(null);

  // Count export: per_image_count and orthomosaic_plant_counts, the two delivery kinds this
  // route serves; an acknowledgment of either is recorded here.
  const [countKind, setCountKind] =
    useState<ExportCountCsvPayload["delivery"]["kind"]>("per_image_count");
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
  const [countExported, setCountExported] = useState<CountExport | null>(null);

  // The trait a delivery is computed for, from this project's registered traits: the one when
  // there is one, none with an error when there are none, and the breeder's choice of several.
  const [traitsListing, setTraitsListing] = useState<TraitsListing | null>(null);
  const traitRecords = traitsListing?.traits ?? [];
  const availableTraits = traitRecords.map((r) => r.trait);
  const [trait, setTrait] = useState("");
  const [traitError, setTraitError] = useState<string | null>(null);

  useEffect(() => {
    if (!projectRoot) return;
    setTrait("");
    setTraitError(null);
    void resultsApi
      .traits()
      .then((res) => {
        setTraitsListing(res);
        if (res.traits.length === 0) {
          setTraitError("No trait is registered for this project yet.");
        } else if (res.traits.length === 1) {
          setTrait(res.traits[0].trait);
        }
      })
      .catch((e) => {
        setTraitsListing(null);
        setTraitError(`Could not load this project's registered traits: ${messageOf(e)}`);
      });
  }, [projectRoot]);

  useEffect(() => {
    if (!projectRoot) return;
    void resultsApi
      .listPlantMappings()
      .then(setMappings)
      .catch(() => setMappings(null));
  }, [projectRoot]);

  useEffect(() => {
    if (!projectRoot) return;
    void resultsApi
      .deliveryEvents()
      .then((res) => {
        setDeliveryEvents(res);
        setDeliveryEventsError(null);
      })
      .catch((e) => {
        setDeliveryEvents(null);
        setDeliveryEventsError(`Could not load what this project has shipped: ${messageOf(e)}`);
      });
  }, [projectRoot]);

  const refreshDatasetTree = useCallback(() => {
    if (!datasetRoot) return;
    void api.dataset
      .tree(datasetRoot)
      .then((t) => {
        setTree(t);
        setDateBucket({});
        setDatesError(null);
      })
      .catch((e) => {
        setDatesError(`Could not load this dataset's dates: ${messageOf(e)}`);
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
      const inputs = {
        mapping_name: mappingName,
        dataset_root: datasetRoot,
        buckets: Object.values(dateBucket).filter((name) => name),
        trait,
        plants,
      };
      setLastInputs(inputs);
      setAnswer(
        await resultsApi.phenologyMeasurement({ ...inputs, show_unvalidated: showUnvalidated }),
      );
      setUnvalidatedRefusal(null);
    } catch (e) {
      setAnswer(null);
      // A refusal naming its own kind is routed by that kind, before any prose is read.
      const refusal = refusalOf<OperationalizationRefusal>(e, OPERATIONALIZATION);
      if (refusal) {
        setOperationalizationRefusal(refusal);
        return;
      }
      // The server refuses an unassessed delivery by default. Surface why, plus the one-click way
      // to see the numbers anyway, marked unvalidated.
      const delivery = refusalOf<DeliveryRefusal>(e, DELIVERY);
      if (delivery) {
        setUnvalidatedRefusal(delivery);
      } else {
        setError(messageOf(e));
      }
    } finally {
      setLoading(false);
    }
  }

  async function downloadCsv(payload: Projection) {
    if (!lastInputs) return;
    if (shownUnvalidated && !ackReason.trim()) return;
    try {
      const body: ExportCsvPayload = {
        ...lastInputs,
        payload,
        user: useStore.getState().user,
        acknowledgment: shownUnvalidated && {
          reason: ackReason.trim(),
          result_sha256: shownUnvalidated.digests[payload],
        },
      };
      const { blob, exported } = await resultsApi.downloadCsv(body);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = exported.filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      const committed = committedOf<SavedExport>(e);
      if (committed) {
        useStore.getState().pushToast(auditGapExportMessage(committed.saved_path, messageOf(e)));
        return;
      }
      // The download door refuses through the same structured family, so it lands in the panel.
      const refusal = refusalOf<OperationalizationRefusal>(e, OPERATIONALIZATION);
      if (refusal) {
        setOperationalizationRefusal(refusal);
        return;
      }
      useStore.getState().pushToast(`CSV export failed: ${messageOf(e)}`);
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
    setCountExported(null);
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
              crop: countCrop,
              pipeline_version: countPipelineVersion,
              canopy_subject: countCanopySubject,
              position_error_m:
                countCanopySubject && countPositionErrorM !== ""
                  ? Number(countPositionErrorM)
                  : null,
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
      const { blob, exported } = await resultsApi.downloadCountCsv(body);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = exported.filename;
      a.click();
      URL.revokeObjectURL(url);
      setCountExported(exported);
      setCountShowAck(false);
      setCountGateRefusal(null);
    } catch (e) {
      const committed = committedOf<SavedExport>(e);
      if (committed) {
        setCountError(auditGapExportMessage(committed.saved_path, messageOf(e)));
        return;
      }
      const opRefusal = refusalOf<OperationalizationRefusal>(e, OPERATIONALIZATION);
      if (opRefusal) {
        setCountOperationalizationRefusal(opRefusal);
        return;
      }
      const gateRefusal = refusalOf<DeliveryRefusal>(e, DELIVERY);
      if (gateRefusal) {
        setCountGateRefusal(gateRefusal);
        setCountShowAck(gateRefusal.result_sha256 !== null);
        return;
      }
      setCountError(messageOf(e));
    } finally {
      setCountExporting(false);
    }
  }

  // Predictions with no positive-state class export nothing; an unvalidated measurement exports
  // once acknowledged (below).
  const exportBlocked = positiveClassUnassessed;
  const downloadOnsetCsv = () => {
    if (exportBlocked) return;
    void downloadCsv("milestones");
  };
  const downloadCurvesCsv = () => {
    if (exportBlocked) return;
    // A curve export is the same delivered phenology measurement as the milestone one, unsummarized.
    void downloadCsv("curves");
  };

  const chartData: CaptureRow[] = useMemo(() => {
    const byCapture: Record<string, CaptureRow> = {};
    for (const r of curves) {
      byCapture[r.capture] ??= { capture: r.capture };
      byCapture[r.capture][r.plant_id] = r.ratio;
    }
    // In the order the tree lists its captures.
    return (tree?.captures ?? []).flatMap((c) => byCapture[c.name] ?? []);
  }, [curves, tree]);

  const plantKeys = useMemo(() => {
    const set = new Set<string>();
    curves.forEach((r) => set.add(r.plant_id));
    return Array.from(set);
  }, [curves]);

  // The entry a delivery reads for the selected trait: its latest confirmed revision's, or null.
  const selectedRecord = traitRecords.find((r) => r.trait === trait);
  const confirmedEntry = selectedRecord?.latest_confirmed?.entry ?? null;
  // Curves and milestones are only meaningful for a trait whose confirmed revision declares the
  // fractions they are read off. Nothing else on this tab depends on it.
  const hasMilestones = (confirmedEntry?.milestone_fractions ?? []).length > 0;

  return (
    <div className="flex-1 overflow-auto p-4 flex flex-col gap-4">
      <TabHeading tab="results" />
      {traitError && <div className="tcip-problem">{traitError}</div>}
      {traitsListing && traitsListing.unreadable.length > 0 && (
        <div className="tcip-problem">
          These traits&apos; records will not read:{" "}
          {traitsListing.unreadable.map((u) => u.trait).join(", ")}.
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

      <DeliveryEventsPanel
        records={deliveryEvents?.records ?? []}
        loadError={deliveryEventsError}
      />

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
              {tree?.captures.flatMap(({ name: d }) =>
                bucketsForDate(tree, d).map((name) => (
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
            {countExported && (
              <div className="text-tcip-muted">
                Saved to {countExported.saved_path}.
                {countExported.validated ? "" : " Not validated."}
                {countExported.acknowledged_by
                  ? ` Acknowledged by ${countExported.acknowledged_by}.`
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
                {!datesError && tree?.label_problem && (
                  <div className="tcip-problem mb-1">{tree.label_problem}</div>
                )}
                {!tree || tree.captures.length === 0 ? (
                  !datesError && <div className="text-tcip-muted">No dates with images.</div>
                ) : (
                  <div className="max-h-40 overflow-auto rounded border border-tcip-border">
                    <table className="w-full text-[13px]">
                      <tbody>
                        {tree.captures.map(({ name: d }) => {
                          const opts = bucketsForDate(tree, d);
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
                  {(mappings?.names ?? []).map((name) => (
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
                    <div>{unvalidatedRefusal.message}</div>
                    <button
                      className="tcip-btn self-start"
                      onClick={() => void compute(true)}
                      disabled={loading}
                    >
                      Show unvalidated numbers
                    </button>
                  </div>
                )}
                {shownUnvalidated && (
                  <div className="tcip-problem">Not validated: {shownUnvalidated.message}</div>
                )}
                {answer && <MappingDisclosureLines disclosure={answer.plant_mapping} />}
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
                  <XAxis dataKey="capture" stroke={CHART.axis} style={{ fontSize: 11 }} />
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
              {answer && (
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
                      {Object.keys(onset[0].milestones).map((key) => (
                        <th key={key} className="tcip-th">
                          {key}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {onset.map((r) => {
                      // A plant with any unassessed or missing date shows as such.
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
                              <span className="tcip-problem" title={shownUnvalidated?.message}>
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
                          {Object.entries(r.milestones).map(([key, crossing]) => {
                            const bound = crossing?.bound;
                            // A censored crossing is a bound on the true date.
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
                              <td key={key} className="pr-3 tabular-nums">
                                {crossing ? crossing.date : UNSET_GLYPH}
                                {crossing && marker && (
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
