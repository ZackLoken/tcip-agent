/** Meta-loop API helpers: the agent's friction reports and retrospectives. */

import { getJson } from "@/api/http";
import { ROUTES } from "@/api/routes";
import type { MemoryPage_FrictionReport_, MemoryPage_Retrospective_ } from "@/api/types.generated";

/** Both read the project the backend has open. */
export const metaApi = {
  reports: () => getJson<MemoryPage_FrictionReport_>(ROUTES.getMetaReports),

  retrospectives: () => getJson<MemoryPage_Retrospective_>(ROUTES.getMetaRetrospectives),
};
