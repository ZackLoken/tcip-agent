/** Meta-loop API helpers — Claude's friction reports and retrospectives. */

import { getJson } from "@/api/http";

export interface FrictionReport {
  file: string;
  timestamp: string | null;
  category: string;
  detail: string;
  context: Record<string, unknown>;
}

export interface Retrospective {
  project_id: string;
  modified: string;
  content: string;
}

export const metaApi = {
  reports: (project_root: string) =>
    getJson<{ reports: FrictionReport[]; count: number; total_available: number }>(
      `/api/meta/reports?project_root=${encodeURIComponent(project_root)}`,
    ),

  retrospectives: (project_root: string) =>
    getJson<{ retrospectives: Retrospective[]; count: number; total_available: number }>(
      `/api/meta/retrospectives?project_root=${encodeURIComponent(project_root)}`,
    ),
};
