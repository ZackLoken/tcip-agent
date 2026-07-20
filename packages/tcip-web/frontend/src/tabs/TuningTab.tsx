import { useEffect, useState } from "react";

import { tuningApi, type Sweep, type SweepDetail } from "@/api/tuning";
import {
  buildSearchSpace,
  DEFAULT_HPO_PARAMS,
  parseNumList,
  SCHEDULERS,
  SEARCH_ALGORITHMS,
  type HpoParam,
} from "@/tabs/hpoSpace";

const DEFAULT_BASE = `{
  "model_source": {
    "builder": "my_models:build_net",
    "builder_kwargs": {"num_classes": 1, "in_chans": 3},
    "task": "detection"
  },
  "data": {"images_dir": "", "labels_dir": "", "task": "detection"},
  "training": {"batch_size": 4, "num_workers": 0, "stages": [{"epochs": 3}]}
}`;

const TERMINAL = new Set(["completed", "failed", "interrupted"]);

export function TuningTab() {
  const [base, setBase] = useState(DEFAULT_BASE);
  const [params, setParams] = useState<HpoParam[]>(DEFAULT_HPO_PARAMS);
  const [nTrials, setNTrials] = useState(5);
  const [searchAlg, setSearchAlg] = useState<string>("random");
  const [scheduler, setScheduler] = useState<string>("asha");
  const [outputDir, setOutputDir] = useState("");

  const [sweeps, setSweeps] = useState<Sweep[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<SweepDetail | null>(null);
  const [launchMsg, setLaunchMsg] = useState<{ text: string; ok: boolean } | null>(null);

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

  // Load the selected sweep's detail, then poll while it's still running. Keyed on the id
  // so RE-selecting the same sweep keeps its already-loaded detail (the old code blanked
  // the result on click and never re-fetched a same-id sweep).
  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const r = await tuningApi.getSweep(selectedId);
        if (cancelled) return;
        setDetail(r);
        if (!TERMINAL.has(r.status)) timer = setTimeout(poll, 3000);
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

  function updateParam(key: string, patch: Record<string, unknown>) {
    setParams((ps) => ps.map((p) => (p.key === key ? ({ ...p, ...patch } as HpoParam) : p)));
  }

  async function launch() {
    setLaunchMsg(null);
    const space = buildSearchSpace(params);
    if (Object.keys(space).length === 0) {
      setLaunchMsg({ text: "Enable at least one parameter to sweep.", ok: false });
      return;
    }
    try {
      const resp = await tuningApi.launch({
        base_config: JSON.parse(base),
        param_space: space,
        n_trials: nTrials,
        output_dir: outputDir,
        search_alg: searchAlg,
        scheduler,
      });
      if (resp.sweep_id) {
        setLaunchMsg({ text: `Launched ${resp.sweep_id}`, ok: true });
        setSelectedId(resp.sweep_id);
        void refresh();
      } else {
        setLaunchMsg({ text: `Error: ${JSON.stringify(resp)}`, ok: false });
      }
    } catch (e) {
      setLaunchMsg({ text: String(e), ok: false });
    }
  }

  return (
    <div className="flex-1 grid grid-cols-[440px_1fr] overflow-hidden">
      <div className="border-r border-tcip-border p-4 overflow-auto">
        <div className="tcip-heading mb-3">HPO config</div>

        <label className="tcip-label mb-1">Base training config (JSON)</label>
        <textarea
          className="tcip-input w-full h-40 font-mono text-[11px] leading-4 resize-none mb-3"
          value={base}
          onChange={(e) => setBase(e.target.value)}
          spellCheck={false}
        />

        {/* Structured search space (fed to Ray Tune) */}
        <label className="tcip-label mb-1">Search space — each trial trains for real</label>
        <div className="tcip-panel p-2 mb-3">
          {params.map((p) => (
            <div key={p.key} className="py-1 border-b border-tcip-border last:border-0">
              <label className="flex items-center gap-2 text-[11px]">
                <input
                  type="checkbox"
                  checked={p.enabled}
                  onChange={(e) => updateParam(p.key, { enabled: e.target.checked })}
                />
                <span className="font-medium">{p.label}</span>
                <span className="text-[10px] text-tcip-muted font-mono">{p.key}</span>
              </label>
              {p.enabled && (
                <div className="mt-1 ml-5 text-[10px]">
                  {p.kind === "loguniform" && (
                    <div className="flex items-center gap-2">
                      <label className="flex items-center gap-1">
                        low
                        <input
                          className="tcip-input w-24 text-[11px]"
                          type="number"
                          step="any"
                          value={Number.isFinite(p.low) ? p.low : ""}
                          onChange={(e) => updateParam(p.key, { low: parseFloat(e.target.value) })}
                        />
                      </label>
                      <label className="flex items-center gap-1">
                        high
                        <input
                          className="tcip-input w-24 text-[11px]"
                          type="number"
                          step="any"
                          value={Number.isFinite(p.high) ? p.high : ""}
                          onChange={(e) => updateParam(p.key, { high: parseFloat(e.target.value) })}
                        />
                      </label>
                      <span className="text-tcip-muted">log-uniform</span>
                    </div>
                  )}
                  {p.kind === "numlist" && (
                    <input
                      className="tcip-input w-full text-[11px]"
                      value={p.values.join(", ")}
                      onChange={(e) => updateParam(p.key, { values: parseNumList(e.target.value) })}
                      placeholder="comma-separated, e.g. 2, 4, 8"
                      spellCheck={false}
                    />
                  )}
                  {p.kind === "choices" && (
                    <div className="flex flex-wrap gap-1">
                      {p.options.map((opt) => {
                        const on = p.selected.includes(opt);
                        return (
                          <button
                            key={opt}
                            type="button"
                            aria-pressed={on}
                            className={`h-6 rounded border px-2 text-[10px] transition-colors ${
                              on
                                ? "border-tcip-accent bg-tcip-accent text-white"
                                : "border-tcip-border bg-tcip-bg text-tcip-muted hover:border-tcip-border-hover hover:text-tcip-fg"
                            }`}
                            onClick={() =>
                              updateParam(p.key, {
                                selected: on
                                  ? p.selected.filter((o) => o !== opt)
                                  : [...p.selected, opt],
                              })
                            }
                          >
                            {opt}
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-2 mb-3 text-[11px] text-tcip-muted">
          <label className="flex flex-col gap-1">
            Trials
            <input
              className="tcip-input w-full"
              type="number"
              min={1}
              max={200}
              value={nTrials}
              onChange={(e) => setNTrials(parseInt(e.target.value, 10) || 1)}
            />
          </label>
          <label className="flex flex-col gap-1">
            Search algorithm
            <select
              className="tcip-select w-full"
              value={searchAlg}
              onChange={(e) => setSearchAlg(e.target.value)}
            >
              {SEARCH_ALGORITHMS.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-col gap-1">
            Scheduler
            <select
              className="tcip-select w-full"
              value={scheduler}
              onChange={(e) => setScheduler(e.target.value)}
            >
              {SCHEDULERS.map((s) => (
                <option key={s} value={s}>
                  {s}
                </option>
              ))}
            </select>
          </label>
        </div>

        <label className="tcip-label mb-1">Output directory</label>
        <input
          className="tcip-input w-full mb-3"
          value={outputDir}
          onChange={(e) => setOutputDir(e.target.value)}
          placeholder=".../Valley_Farm/.tcip/hpo"
        />

        <button className="tcip-btn-primary w-full" onClick={launch}>
          ▶&nbsp;&nbsp;Launch HPO
        </button>
        {launchMsg && (
          <div className={`mt-2 text-[11px] ${launchMsg.ok ? "text-tcip-tp" : "text-tcip-fp"}`}>
            {launchMsg.text}
          </div>
        )}
        <div className="mt-1 text-[10px] text-tcip-muted">
          Note: a running sweep can’t be cancelled — trials run to completion. Keep the trial count
          modest.
        </div>

        <div className="mt-5">
          <div className="tcip-heading mb-2">Sweeps</div>
          {sweeps.length === 0 ? (
            <div className="text-[11px] text-tcip-muted">No sweeps yet.</div>
          ) : (
            <ul className="space-y-1">
              {sweeps.map((s) => (
                <li
                  key={s.sweep_id}
                  className={`p-2 rounded border cursor-pointer transition-colors ${
                    selectedId === s.sweep_id
                      ? "border-tcip-accent bg-tcip-accent/10"
                      : "border-tcip-border hover:border-tcip-border-hover hover:bg-tcip-hover"
                  }`}
                  onClick={() => setSelectedId(s.sweep_id)}
                >
                  <div className="font-mono text-[11px]">{s.sweep_id}</div>
                  <div className="text-[10px] text-tcip-muted">
                    {s.status}
                    {s.error ? ` — ${s.error}` : ""}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="p-4 overflow-auto">
        <div className="mb-3 flex items-center gap-2">
          {detail ? (
            <>
              <span className="tcip-heading">Sweep</span>
              <span className="font-mono text-[12px] text-tcip-fg">{detail.sweep_id}</span>
              <span className="text-[11px] text-tcip-muted">({detail.status})</span>
            </>
          ) : (
            <span className="tcip-heading">Select a sweep</span>
          )}
        </div>
        {detail?.result ? (
          <pre className="text-[11px] font-mono p-3 tcip-panel overflow-auto max-h-[80vh]">
            {JSON.stringify(detail.result, null, 2)}
          </pre>
        ) : (
          <div className="text-[11px] text-tcip-muted">
            Launch a sweep; the full trials + best config appear here when it finishes.
            Parallel-coords visualisation planned for a post-Phase-1 iteration.
          </div>
        )}
      </div>
    </div>
  );
}
