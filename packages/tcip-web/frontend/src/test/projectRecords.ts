import type { api } from "@/api/client";
import type { CaptureInvariances, InvariancesCorrection } from "@/api/types.generated";
import fixtures from "@/test/projectFixtures.json";

/**
 * What `tools/generate_project_fixture.py` served from the platform's own doors: the `listing`
 * `GET /api/projects` answers for two created projects, the `correction` the invariances route
 * answers for one correction of the second, and the route's `refusal` of an out-of-range block.
 */
export const PROJECT_FIXTURES = fixtures as {
  listing: Awaited<ReturnType<typeof api.projects.list>>;
  correction: InvariancesCorrection;
  refusal: { detail: unknown[] };
};

/** A sample breeder's answer a readable project record carries: every capture transform leaves
 * the trait's measurement unchanged. */
export const INVARIANCES: CaptureInvariances = {
  horizontal_flip: true,
  vertical_flip: true,
  rotation: 180,
  color: true,
  intensity: true,
  scale: true,
};
