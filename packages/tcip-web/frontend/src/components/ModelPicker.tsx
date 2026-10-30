/** The open project's registered models, one chosen by its checkpoint. */

import { useCallback, useEffect, useState } from "react";

import { messageOf } from "@/api/http";
import { resultsApi } from "@/api/inference";
import type { Listed, RegisteredModels } from "@/api/types.generated";
import { useStore } from "@/store";
import { selectProjectRoot } from "@/store/slices/gui";

/**
 * A labeled select over the open project's registered models, read once per project with a
 * retry on a failed read. `value` is the chosen model, `onChange` hands on the one the person
 * picks, null for none; `id` names the select for its label.
 */
export function ModelPicker({
  id,
  value,
  onChange,
}: {
  id: string;
  value: Listed | null;
  onChange: (model: Listed | null) => void;
}) {
  const projectRoot = useStore(selectProjectRoot);
  const [registered, setRegistered] = useState<RegisteredModels | null>(null);
  const [error, setError] = useState<string | null>(null);
  const models = registered?.models ?? [];

  const refresh = useCallback(() => {
    if (!projectRoot) return;
    void resultsApi
      .registeredModels()
      .then((r) => {
        setRegistered(r);
        setError(null);
      })
      .catch((e) => {
        setRegistered(null);
        setError(`Could not load registered models: ${messageOf(e)}`);
      });
  }, [projectRoot]);

  useEffect(refresh, [refresh]);

  return (
    <>
      <label className="tcip-label mb-1" htmlFor={id}>
        Model
      </label>
      {error && (
        <div className="tcip-problem mb-1">
          {error}{" "}
          <button className="tcip-btn ml-1" onClick={refresh}>
            Retry
          </button>
        </div>
      )}
      {models.length > 0 ? (
        <select
          id={id}
          className="tcip-select w-full mb-3"
          value={value?.entry.checkpoint_path ?? ""}
          onChange={(e) =>
            onChange(models.find((m) => m.entry.checkpoint_path === e.target.value) ?? null)
          }
        >
          <option value="">Choose a model</option>
          {models.map(({ entry }) => (
            <option key={entry.checkpoint_path} value={entry.checkpoint_path}>
              {entry.name}
            </option>
          ))}
        </select>
      ) : (
        !error && projectRoot && <p className="text-tcip-muted mb-3">No model is registered.</p>
      )}
    </>
  );
}
