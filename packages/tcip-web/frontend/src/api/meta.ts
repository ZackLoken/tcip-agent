/** Meta-loop API helpers: the agent's friction reports and retrospectives. */

import { getJson } from "@/api/http";
import { ROUTES } from "@/api/routes";

export interface FrictionReport {
  file: string;
  timestamp: string | null;
  category: string;
  detail: string;
  context: Record<string, unknown>;
}

/** `timestamp` is the latest section header the document itself states, empty when it states none. */
export interface Retrospective {
  project_id: string;
  timestamp: string;
  content: string;
}

/** Both read the project the backend has open. */
export const metaApi = {
  reports: () =>
    getJson<{ reports: FrictionReport[]; count: number; total_available: number }>(
      ROUTES.getMetaReports,
    ),

  retrospectives: () =>
    getJson<{ retrospectives: Retrospective[]; count: number; total_available: number }>(
      ROUTES.getMetaRetrospectives,
    ),
};
