/** The plant mapping names the open project holds. */

import { useCallback, useEffect, useState } from "react";

import { messageOf } from "@/api/http";
import { resultsApi } from "@/api/inference";
import type { MappingNames } from "@/api/types.generated";

/**
 * List the open project's plant mapping names when `projectRoot` is held, and again on a change
 * of it or a call of the returned `refresh`.
 *
 * Returns the names of the last listing that answered, or null with `error` holding the sentence
 * of the one that failed, never both; a listing that answers clears `error`. Nothing is listed
 * while `projectRoot` is null.
 */
export function useMappingNames(projectRoot: string | null): {
  mappings: MappingNames | null;
  error: string | null;
  refresh: () => void;
} {
  const [mappings, setMappings] = useState<MappingNames | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    if (!projectRoot) return;
    void resultsApi
      .listPlantMappings()
      .then((res) => {
        setMappings(res);
        setError(null);
      })
      .catch((e: unknown) => {
        setMappings(null);
        setError(`Could not load the plant mappings: ${messageOf(e)}`);
      });
  }, [projectRoot]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return { mappings, error, refresh };
}
