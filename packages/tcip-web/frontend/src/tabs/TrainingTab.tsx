/**
 * Training tab — Slice 2. Stub until then: shows a placeholder with the
 * list of runs from the backend (read-only).
 */

import { useEffect, useState } from "react";

async function safeFetch<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    if (!r.ok) return null;
    return (await r.json()) as T;
  } catch {
    return null;
  }
}

export function TrainingTab() {
  const [runs, setRuns] = useState<{ run_id: string; status: string }[]>([]);

  useEffect(() => {
    void safeFetch<{ run_id: string; status: string }[]>("/api/training").then((r) => {
      if (r) setRuns(r);
    });
  }, []);

  return (
    <div className="flex-1 p-6">
      <div className="text-xl font-semibold mb-2">Training</div>
      <div className="text-[12px] text-tcip-muted mb-4">
        Wire-up lands in Slice 2: config editor, launch, live loss curves, compare runs.
      </div>
      <div className="tcip-panel p-4 max-w-3xl">
        <div className="text-[12px] text-tcip-muted mb-2">Existing runs</div>
        {runs.length === 0 ? (
          <div className="text-[11px] text-tcip-muted">No runs yet.</div>
        ) : (
          <ul className="text-[12px] space-y-1">
            {runs.map((r) => (
              <li key={r.run_id} className="flex justify-between">
                <span className="font-mono">{r.run_id}</span>
                <span className="text-tcip-muted">{r.status}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
