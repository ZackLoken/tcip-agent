import { useEffect, useState } from "react";

interface Sweep {
  sweep_id: string;
  status: string;
  error: string | null;
  has_result: boolean;
}

const DEFAULT_BASE = `{
  "model_spec": {
    "backbone": {"name": "resnet50", "pretrained": true},
    "neck": {"name": "fpn"},
    "heads": [{"name": "detection_head", "task": "detection", "num_classes": 1}],
    "loss": {"name": "focal_loss"}
  },
  "data": {"images_dir": "", "labels_dir": "", "task": "detection"},
  "training": {"batch_size": 4, "num_workers": 0, "stages": [{"lr": 1e-3, "epochs": 3}]}
}`;

const DEFAULT_SPACE = `{
  "optimizer.backbone_lr": [1e-5, 5e-5, 1e-4, 5e-4],
  "optimizer.head_lr": [1e-4, 5e-4, 1e-3],
  "training.batch_size": [2, 4, 8]
}`;

export function TuningTab() {
  const [base, setBase] = useState(DEFAULT_BASE);
  const [space, setSpace] = useState(DEFAULT_SPACE);
  const [nTrials, setNTrials] = useState(5);
  const [direction, setDirection] = useState<"maximize" | "minimize">("maximize");
  const [useOptuna, setUseOptuna] = useState(false);
  const [outputDir, setOutputDir] = useState("");

  const [sweeps, setSweeps] = useState<Sweep[]>([]);
  const [active, setActive] = useState<{
    sweep_id: string;
    status: string;
    result: unknown;
  } | null>(null);
  const [launchMsg, setLaunchMsg] = useState<string | null>(null);

  async function refresh() {
    try {
      const r = await fetch("/api/tuning/sweeps").then((r) => r.json());
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

  useEffect(() => {
    if (!active?.sweep_id) return;
    const t = setInterval(async () => {
      try {
        const r = await fetch(`/api/tuning/sweeps/${active.sweep_id}`).then((r) => r.json());
        setActive({ sweep_id: r.sweep_id, status: r.status, result: r.result });
        if (r.status === "completed" || r.status === "failed") return clearInterval(t);
      } catch {
        /* ignore */
      }
    }, 3000);
    return () => clearInterval(t);
  }, [active?.sweep_id]);

  async function launch() {
    setLaunchMsg(null);
    try {
      const body = {
        base_config: JSON.parse(base),
        param_space: JSON.parse(space),
        n_trials: nTrials,
        output_dir: outputDir,
        use_optuna: useOptuna,
        direction,
      };
      const resp = await fetch("/api/tuning/launch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).then((r) => r.json());
      if (resp.sweep_id) {
        setLaunchMsg(`Launched ${resp.sweep_id}`);
        setActive({ sweep_id: resp.sweep_id, status: "running", result: null });
        void refresh();
      } else {
        setLaunchMsg(`Error: ${JSON.stringify(resp)}`);
      }
    } catch (e) {
      setLaunchMsg(String(e));
    }
  }

  return (
    <div className="flex-1 grid grid-cols-[440px_1fr] overflow-hidden">
      <div className="border-r border-tcip-border p-4 overflow-auto">
        <div className="font-semibold text-[13px] mb-3">HPO config</div>

        <label className="block text-[11px] text-tcip-muted mb-1">
          Base training config (JSON)
        </label>
        <textarea
          className="tcip-input w-full h-40 font-mono text-[11px] leading-4 resize-none mb-3"
          value={base}
          onChange={(e) => setBase(e.target.value)}
          spellCheck={false}
        />

        <label className="block text-[11px] text-tcip-muted mb-1">Param space (JSON)</label>
        <textarea
          className="tcip-input w-full h-32 font-mono text-[11px] leading-4 resize-none mb-3"
          value={space}
          onChange={(e) => setSpace(e.target.value)}
          spellCheck={false}
        />

        <div className="grid grid-cols-2 gap-2 mb-3 text-[11px]">
          <label>
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
          <label>
            Direction
            <select
              className="tcip-select w-full"
              value={direction}
              onChange={(e) => setDirection(e.target.value as "maximize" | "minimize")}
            >
              <option value="maximize">Maximize</option>
              <option value="minimize">Minimize</option>
            </select>
          </label>
          <label className="col-span-2 flex items-center gap-2">
            <input
              type="checkbox"
              checked={useOptuna}
              onChange={(e) => setUseOptuna(e.target.checked)}
            />
            Use Optuna (TPE + ASHA)
          </label>
        </div>

        <label className="block text-[11px] text-tcip-muted mb-1">Output directory</label>
        <input
          className="tcip-input w-full mb-3"
          value={outputDir}
          onChange={(e) => setOutputDir(e.target.value)}
          placeholder=".../Valley_Farm/.tcip/hpo"
        />

        <button className="tcip-btn-primary w-full" onClick={launch}>
          ▶ Launch HPO
        </button>
        {launchMsg && <div className="mt-2 text-[11px] text-tcip-muted">{launchMsg}</div>}

        <div className="mt-4">
          <div className="font-semibold text-[13px] mb-2">Sweeps</div>
          {sweeps.length === 0 ? (
            <div className="text-[11px] text-tcip-muted">No sweeps yet.</div>
          ) : (
            <ul className="space-y-1">
              {sweeps.map((s) => (
                <li
                  key={s.sweep_id}
                  className={`p-2 rounded border cursor-pointer ${
                    active?.sweep_id === s.sweep_id
                      ? "border-tcip-accent bg-tcip-accent/10"
                      : "border-tcip-border hover:border-tcip-muted"
                  }`}
                  onClick={() =>
                    setActive({ sweep_id: s.sweep_id, status: s.status, result: null })
                  }
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
        <div className="font-semibold text-[13px] mb-2">
          {active ? `Sweep ${active.sweep_id} (${active.status})` : "Select a sweep"}
        </div>
        {active?.result ? (
          <pre className="text-[11px] font-mono p-3 tcip-panel overflow-auto max-h-[80vh]">
            {JSON.stringify(active.result, null, 2)}
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
