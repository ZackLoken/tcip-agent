import { useEffect, useMemo, useState } from "react";

import { tuningApi, type Sweep, type SweepDetail, type SweepTrial } from "@/api/tuning";
import type { TensorboardLaunch } from "@/api/training";
import { DisclosureChevron } from "@/components/CollapsibleSection";
import { EmbeddedTool } from "@/components/EmbeddedTool";
import { LaunchPicker } from "@/components/LaunchPicker";
import { useEditableAgentRequest } from "@/hooks/useEditableAgentRequest";
import { TERMINAL_STATUSES } from "@/lib/runStatus";
import { useStore } from "@/store";
import { defaultSweepRequest } from "@/tabs/agentPrompts";
import { RunMonitorEmpty, RunMonitorLayout } from "@/tabs/RunMonitorLayout";

function hasContent(value: unknown): boolean {
  return typeof value === "object" && value !== null && Object.keys(value).length > 0;
}

/** A launch either serves a url straight away or says why it could not. */
function launchOutcome(launched: TensorboardLaunch): { url: string | null; error: string | null } {
  if (launched.url) return { url: launched.url, error: null };
  const error = launched.error
    ? launched.output
      ? `${launched.error}: ${launched.output}`
      : launched.error
    : "TensorBoard did not start.";
  return { url: null, error };
}

function messageOf(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function cellText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

export function TuningTab() {
  const datasetRoot = useStore((s) => s.gui.dataset.dataset_root);
  const { request, setRequest } = useEditableAgentRequest(defaultSweepRequest(datasetRoot));

  const [pickerOpen, setPickerOpen] = useState(false);
  const [sweeps, setSweeps] = useState<Sweep[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SweepDetail | null>(null);
  const [trials, setTrials] = useState<SweepTrial[]>([]);
  const [selectedTrialId, setSelectedTrialId] = useState<string | null>(null);
  const [trialMetrics, setTrialMetrics] = useState<Record<string, unknown>[]>([]);
  const [rayUrl, setRayUrl] = useState<string | null>(null);
  const [rayError, setRayError] = useState<string | null>(null);
  const [rayAttempt, setRayAttempt] = useState(0);
  const [sweepTb, setSweepTb] = useState<{ url: string | null; error: string | null }>({
    url: null,
    error: null,
  });
  const [sweepTbAttempt, setSweepTbAttempt] = useState(0);
  const [trialTb, setTrialTb] = useState<{ url: string | null; error: string | null }>({
    url: null,
    error: null,
  });
  const [trialTbAttempt, setTrialTbAttempt] = useState(0);

  async function refresh() {
    try {
      const r = await tuningApi.listSweeps();
      setSweeps(r.sweeps ?? []);
    } catch {
      /* ignore */
    }
  }

  useEffect(() => {
    void refresh();
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, []);

  // Load the selected sweep's detail and its trials, then poll while it's still running. Trials
  // are read for the expanded sweep only: a listing per sweep per tick would walk the whole HPO
  // root every few seconds. Keyed on the id so re-selecting the same sweep keeps what it loaded.
  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      setTrials([]);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const d = await tuningApi.getSweep(selectedId);
        if (cancelled) return;
        setDetail(d);
        try {
          const t = await tuningApi.listTrials(selectedId);
          if (!cancelled) setTrials(t.trials ?? []);
        } catch {
          // A sweep launched here has no trial directory until its first trial writes one.
          if (!cancelled) setTrials([]);
        }
        if (!cancelled && !TERMINAL_STATUSES.has(d.status)) timer = setTimeout(poll, 3000);
      } catch {
        /* ignore */
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [selectedId]);

  useEffect(() => {
    if (!selectedId || !selectedTrialId) {
      setTrialMetrics([]);
      return;
    }
    let cancelled = false;
    void tuningApi.getTrialMetrics(selectedId, selectedTrialId).then(
      (r) => {
        if (!cancelled) setTrialMetrics(r.metrics ?? []);
      },
      () => {
        if (!cancelled) setTrialMetrics([]);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [selectedId, selectedTrialId]);

  // Ray holds one cluster per process and tears it down with the sweep that started it, so the
  // dashboard is read fresh whenever this panel comes up rather than cached across selections.
  useEffect(() => {
    setRayUrl(null);
    setRayError(null);
    if (!selectedId) return;
    let cancelled = false;
    void tuningApi.getRayDashboard().then(
      (r) => {
        if (cancelled) return;
        setRayUrl(r.url);
        setRayError(r.url ? null : "Ray's dashboard isn't running right now.");
      },
      (e) => {
        if (!cancelled) setRayError(messageOf(e));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [selectedId, rayAttempt]);

  // One TensorBoard over the whole sweep directory: the launch is idempotent and answers with the
  // url of whatever is already serving it, so there is nothing to wait for afterwards.
  useEffect(() => {
    setSweepTb({ url: null, error: null });
    if (!selectedId) return;
    let cancelled = false;
    void tuningApi.launchSweepTensorboard(selectedId).then(
      (launched) => {
        if (!cancelled) setSweepTb(launchOutcome(launched));
      },
      (e) => {
        if (!cancelled) setSweepTb({ url: null, error: messageOf(e) });
      },
    );
    return () => {
      cancelled = true;
    };
  }, [selectedId, sweepTbAttempt]);

  // Every trial's TensorBoard costs a port out of a bounded range, so moving off a trial stops
  // the one it was showing.
  useEffect(() => {
    setTrialTb({ url: null, error: null });
    if (!selectedId || !selectedTrialId) return;
    const sweepId = selectedId;
    const trialId = selectedTrialId;
    let cancelled = false;
    void tuningApi.launchTrialTensorboard(sweepId, trialId).then(
      (launched) => {
        if (!cancelled) setTrialTb(launchOutcome(launched));
      },
      (e) => {
        if (!cancelled) setTrialTb({ url: null, error: messageOf(e) });
      },
    );
    return () => {
      cancelled = true;
      void tuningApi.stopTrialTensorboard(sweepId, trialId).catch(() => {
        /* the trial's TensorBoard is already gone */
      });
    };
  }, [selectedId, selectedTrialId, trialTbAttempt]);

  const metricColumns = useMemo(() => {
    const keys: string[] = [];
    trialMetrics.forEach((row) => {
      Object.keys(row).forEach((k) => {
        if (!keys.includes(k)) keys.push(k);
      });
    });
    return keys;
  }, [trialMetrics]);

  const selectedTrial = trials.find((t) => t.trial_id === selectedTrialId) ?? null;

  function toggleSweep(sweepId: string) {
    setSelectedTrialId(null);
    setSelectedId((current) => (current === sweepId ? null : sweepId));
  }

  async function onCancelSweep(sweepId: string) {
    try {
      await tuningApi.cancel(sweepId);
      void refresh();
    } catch (e) {
      useStore.getState().pushToast(`Cancel failed: ${messageOf(e)}`);
    }
  }

  async function onRelaunchSweep(sweepId: string) {
    try {
      const result = await tuningApi.relaunch(sweepId);
      void refresh();
      if (typeof result.sweep_id === "string") setSelectedId(result.sweep_id);
    } catch (e) {
      useStore.getState().pushToast(`Relaunch failed: ${messageOf(e)}`);
    }
  }

  function sendToAgent() {
    useStore.getState().sendToAgentTerminal(request);
    setPickerOpen(false);
  }

  // Cancel while running, gone once "stop requested" (shown as status text) except for an
  // external sweep; "Run again" once terminal and relaunchable; nothing otherwise.
  function sweepAction(s: Sweep) {
    const terminal = TERMINAL_STATUSES.has(s.status);
    if (!terminal) {
      if (s.cancel_requested && !s.external) return null;
      return (
        <button
          type="button"
          className="tcip-btn text-[10px] shrink-0 mt-2"
          onClick={(e) => {
            e.stopPropagation();
            void onCancelSweep(s.sweep_id);
          }}
        >
          Cancel
        </button>
      );
    }
    if (s.relaunchable) {
      return (
        <button
          type="button"
          className="tcip-btn text-[10px] shrink-0 mt-2"
          onClick={(e) => {
            e.stopPropagation();
            void onRelaunchSweep(s.sweep_id);
          }}
        >
          Run again
        </button>
      );
    }
    return null;
  }

  return (
    <RunMonitorLayout
      title="Sweeps"
      onRefresh={() => void refresh()}
      headerRight={
        <button
          type="button"
          aria-expanded={pickerOpen}
          className={pickerOpen ? "tcip-btn text-[11px]" : "tcip-btn-primary text-[11px]"}
          onClick={() => setPickerOpen((open) => !open)}
        >
          Start a sweep
        </button>
      }
      detailHeader={
        selectedTrial ? (
          <>
            <span className="tcip-heading">Trial</span>
            <span className="font-mono text-[12px] text-tcip-fg">{selectedTrial.trial_id}</span>
            <span className="text-[11px] text-tcip-muted">of {detail?.sweep_id}</span>
          </>
        ) : detail ? (
          <>
            <span className="tcip-heading">Sweep</span>
            <span className="font-mono text-[12px] text-tcip-fg">{detail.sweep_id}</span>
            <span className="text-[11px] text-tcip-muted">({detail.status})</span>
          </>
        ) : (
          <span className="tcip-heading">Select a sweep</span>
        )
      }
      detail={
        selectedTrial ? (
          <div className="flex flex-col gap-4">
            <div className="h-[52vh] min-h-[320px] shrink-0">
              <EmbeddedTool
                title="Trial TensorBoard"
                url={trialTb.url}
                loading={!trialTb.url && !trialTb.error}
                error={trialTb.error}
                onRetry={() => setTrialTbAttempt((n) => n + 1)}
              />
            </div>
            <div>
              <div className="tcip-heading mb-1">Parameters</div>
              {Object.keys(selectedTrial.params).length === 0 ? (
                <div className="text-[11px] text-tcip-muted">
                  This trial has not written a resolved config yet.
                </div>
              ) : (
                <pre className="text-[11px] font-mono p-3 tcip-panel overflow-auto">
                  {JSON.stringify(selectedTrial.params, null, 2)}
                </pre>
              )}
              {selectedTrial.unconsumed_params.length > 0 && (
                <div className="mt-1 text-[11px] text-tcip-fp">
                  Swept but not read by the training config:{" "}
                  {selectedTrial.unconsumed_params.join(", ")}
                </div>
              )}
            </div>
            <div>
              <div className="tcip-heading mb-1">Metrics</div>
              {trialMetrics.length === 0 ? (
                <div className="text-[11px] text-tcip-muted">
                  No metrics rows from this trial yet.
                </div>
              ) : (
                <div className="tcip-panel overflow-auto">
                  <table className="w-full text-[11px] font-mono">
                    <thead>
                      <tr className="text-tcip-muted">
                        {metricColumns.map((k) => (
                          <th key={k} className="px-2 py-1 text-left font-normal">
                            {k}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {trialMetrics.map((row, i) => (
                        <tr key={i} className="border-t border-tcip-border">
                          {metricColumns.map((k) => (
                            <td key={k} className="px-2 py-1 tabular-nums">
                              {cellText(row[k])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </div>
        ) : detail ? (
          <div className="flex flex-col gap-3">
            <div className="h-[46vh] min-h-[300px] shrink-0">
              <EmbeddedTool
                title="Ray dashboard"
                url={rayUrl}
                loading={!rayUrl && !rayError}
                error={rayError}
                onRetry={() => setRayAttempt((n) => n + 1)}
              />
            </div>
            <div className="h-[38vh] min-h-[260px] shrink-0">
              <EmbeddedTool
                title="Sweep TensorBoard"
                url={sweepTb.url}
                loading={!sweepTb.url && !sweepTb.error}
                error={sweepTb.error}
                onRetry={() => setSweepTbAttempt((n) => n + 1)}
              />
            </div>
            {detail.status === "cancelled" ? (
              <div className="text-[11px] text-tcip-muted">
                Cancelled: {detail.error ?? "no reason recorded"}
              </div>
            ) : (
              <>
                {detail.error ? (
                  <div className="text-[11px] text-tcip-fp">{detail.error}</div>
                ) : null}
                {hasContent(detail.result) ? (
                  <pre className="max-h-[24vh] text-[11px] font-mono p-3 tcip-panel overflow-auto">
                    {JSON.stringify(detail.result, null, 2)}
                  </pre>
                ) : (
                  <div className="text-[11px] text-tcip-muted">
                    The best config appears here once the sweep finishes. Pick one of its trials to
                    follow that trial while it runs.
                  </div>
                )}
              </>
            )}
          </div>
        ) : (
          <div className="text-[11px] text-tcip-muted">
            Select a sweep to see its trials and its result.
          </div>
        )
      }
    >
      {pickerOpen && (
        <div className="mb-3 pb-3 border-b border-tcip-border">
          <LaunchPicker
            composerLabel="Describe a new one to the agent"
            request={request}
            onRequestChange={setRequest}
            onSend={sendToAgent}
          />
        </div>
      )}

      {sweeps.length === 0 ? (
        <RunMonitorEmpty>No sweeps yet. Use "Start a sweep" above.</RunMonitorEmpty>
      ) : (
        <ul className="space-y-1">
          {sweeps.map((s) => {
            const expanded = selectedId === s.sweep_id;
            const running = !TERMINAL_STATUSES.has(s.status);
            const searchLine = [
              s.n_trials != null ? `${s.n_trials} trials` : null,
              s.search_alg,
              s.scheduler,
              s.param_space_keys && s.param_space_keys.length > 0
                ? s.param_space_keys.join(", ")
                : null,
            ]
              .filter((part): part is string => !!part)
              .join(" · ");
            return (
              <li key={s.sweep_id}>
                <div
                  className={`flex items-start gap-1 p-2 rounded border transition-colors ${
                    expanded && !selectedTrialId
                      ? "border-tcip-accent bg-tcip-accent/10"
                      : "border-tcip-border hover:border-tcip-border-hover hover:bg-tcip-hover"
                  }`}
                >
                  <button
                    type="button"
                    aria-expanded={expanded}
                    aria-pressed={expanded}
                    className="flex-1 flex items-start gap-2 text-left"
                    onClick={() => toggleSweep(s.sweep_id)}
                  >
                    <span className="mt-[3px] text-tcip-muted">
                      <DisclosureChevron open={expanded} />
                    </span>
                    <span className="flex-1">
                      <span className="block font-mono text-[11px]">{s.sweep_id}</span>
                      <span className="block text-[10px] text-tcip-muted">
                        {s.status}
                        {running && s.cancel_requested ? " · stop requested" : ""}
                      </span>
                      {s.error && <span className="block text-[10px] text-tcip-fp">{s.error}</span>}
                      {!s.relaunchable && s.reason && (
                        <span className="block text-[10px] text-tcip-muted">{s.reason}</span>
                      )}
                    </span>
                  </button>
                  {sweepAction(s)}
                </div>
                {expanded && (
                  <div className="mt-1 ml-5">
                    {searchLine && (
                      <div className="text-[10px] text-tcip-muted mb-1">{searchLine}</div>
                    )}
                    <ul className="space-y-1">
                      {trials.length === 0 ? (
                        <li className="text-[10px] text-tcip-muted">No trials on disk yet.</li>
                      ) : (
                        trials.map((t) => (
                          <li key={t.trial_id}>
                            <button
                              type="button"
                              aria-pressed={selectedTrialId === t.trial_id}
                              className={`w-full p-2 rounded border text-left transition-colors ${
                                selectedTrialId === t.trial_id
                                  ? "border-tcip-accent bg-tcip-accent/10"
                                  : "border-tcip-border hover:border-tcip-border-hover hover:bg-tcip-hover"
                              }`}
                              onClick={() => setSelectedTrialId(t.trial_id)}
                            >
                              <span className="block font-mono text-[11px]">{t.trial_id}</span>
                              <span className="block text-[10px] text-tcip-muted">
                                {t.has_metrics ? "metrics" : "no metrics yet"}
                                {Object.keys(t.params).length > 0
                                  ? ` · ${Object.entries(t.params)
                                      .map(([k, v]) => `${k}=${cellText(v)}`)
                                      .join(", ")}`
                                  : ""}
                              </span>
                            </button>
                          </li>
                        ))
                      )}
                    </ul>
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </RunMonitorLayout>
  );
}
