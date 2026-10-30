/** The launch of a review queue over the selected capture. */

import { useEffect, useState } from "react";

import { api } from "@/api/client";
import { messageOf } from "@/api/http";
import type { Listed, ScorerCatalog } from "@/api/types.generated";
import { ModelPicker } from "@/components/ModelPicker";
import { jsonValue } from "@/lib/jsonValue";
import { useStore } from "@/store";

/**
 * The form launching a review queue over the selected capture: a registered model
 * (`ModelPicker`), a scorer the backend's catalog lists with each value its factory takes, the
 * budget and the reason. Nothing is filled in until the person states it; Rank stays disabled
 * until the model, the scorer, the budget and the reason are. Each value typed is sent as the
 * JSON value it reads as (`jsonValue`), an empty one not at all; a refused launch shows the
 * backend's refusal as it wrote it. The launched job is held in the store (`setReviewQueue`) as
 * the image stepper's chosen ordering, and `onLaunched` follows.
 */
export function ReviewQueueLaunch({ onLaunched }: { onLaunched: () => void }) {
  const [catalog, setCatalog] = useState<ScorerCatalog | null>(null);
  const [model, setModel] = useState<Listed | null>(null);
  const [scorerName, setScorerName] = useState("");
  const [values, setValues] = useState<Record<string, string>>({});
  const [budget, setBudget] = useState("");
  const [rationale, setRationale] = useState("");
  const [launching, setLaunching] = useState(false);
  const [refusal, setRefusal] = useState<string | null>(null);

  useEffect(() => {
    api.annotate
      .scorers()
      .then(setCatalog, (e: unknown) =>
        useStore.getState().pushToast(`Could not read the scorers: ${messageOf(e)}`),
      );
  }, []);

  const scorer = catalog?.scorers.find((s) => s.name === scorerName);
  // The values schema names each value under `properties` and the ones a block must state under
  // `required`.
  const parameters = Object.keys((scorer?.values_schema.properties ?? {}) as object);
  const required = (scorer?.values_schema.required ?? []) as string[];
  const ready =
    model !== null && scorer !== undefined && budget.trim() !== "" && rationale.trim() !== "";

  async function launch() {
    if (model === null || scorer === undefined) return;
    setLaunching(true);
    setRefusal(null);
    try {
      const job = await api.annotate.launchQueue({
        checkpoint_path: model.entry.checkpoint_path,
        scorer: {
          name: scorer.name,
          ...Object.fromEntries(
            Object.entries(values)
              .filter(([, text]) => text.trim() !== "")
              .map(([name, text]) => [name, jsonValue(text.trim())]),
          ),
        },
        budget: Number(budget),
        rationale: rationale.trim(),
      });
      useStore.getState().setReviewQueue(job);
      useStore.getState().setImageFilter("queue");
      onLaunched();
    } catch (e) {
      setRefusal(messageOf(e));
    } finally {
      setLaunching(false);
    }
  }

  return (
    <div className="flex w-72 flex-col p-3" role="group" aria-label="Review queue">
      <ModelPicker id="queue-model" value={model} onChange={setModel} />
      <label className="tcip-label mb-1" htmlFor="queue-scorer">
        Scorer
      </label>
      <select
        id="queue-scorer"
        className="tcip-select w-full mb-3"
        value={scorerName}
        onChange={(e) => {
          setScorerName(e.target.value);
          setValues({});
        }}
      >
        <option value="">Choose a scorer</option>
        {catalog?.scorers.map(({ name }) => (
          <option key={name} value={name}>
            {name}
          </option>
        ))}
      </select>
      {parameters.map((name) => (
        <div key={name}>
          <label className="tcip-label mb-1" htmlFor={`queue-${name}`}>
            {required.includes(name) ? name : `${name}, optional`}
          </label>
          <input
            id={`queue-${name}`}
            className="tcip-input w-full mb-3 font-mono"
            value={values[name] ?? ""}
            onChange={(e) => setValues((prev) => ({ ...prev, [name]: e.target.value }))}
          />
        </div>
      ))}
      <label className="tcip-label mb-1" htmlFor="queue-budget">
        Budget
      </label>
      <input
        id="queue-budget"
        className="tcip-input w-full mb-3 font-mono"
        type="number"
        min={1}
        step={1}
        value={budget}
        onChange={(e) => setBudget(e.target.value)}
      />
      <label className="tcip-label mb-1" htmlFor="queue-rationale">
        Reason
      </label>
      <input
        id="queue-rationale"
        className="tcip-input w-full mb-3"
        value={rationale}
        onChange={(e) => setRationale(e.target.value)}
      />
      <button
        className="tcip-btn-primary w-full"
        onClick={() => void launch()}
        disabled={!ready || launching}
      >
        Rank
      </button>
      {refusal !== null && (
        <p className="tcip-problem mt-3" role="alert">
          {refusal}
        </p>
      )}
    </div>
  );
}
