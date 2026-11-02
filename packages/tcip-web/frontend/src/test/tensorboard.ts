import type { TensorboardLaunch, TensorboardServing } from "@/api/types.generated";
import produced from "@/test/tensorboardServings.json";

/** What `tools/generate_tensorboard_fixture.py` received from the TensorBoard route, by run id:
 *  one serving for each of two runs, each over its own board, and two failed launches, one that
 *  could take no port and one whose child exited during startup with what it wrote. */
const LAUNCHES = produced as unknown as Record<string, TensorboardLaunch>;

/** The produced servings, in the order the generator served them. */
export const TENSORBOARD_SERVINGS = Object.values(LAUNCHES).flatMap((launch) =>
  launch.kind === "serving" ? [launch] : [],
);

/** The produced failures, the portless launch's first and then the dying child's. */
export const TENSORBOARD_FAILURES = Object.values(LAUNCHES).flatMap((launch) =>
  launch.kind === "failed" ? [launch] : [],
);

/** The produced serving of the run `experimentId` names; an id the generator served no
 *  TensorBoard for throws. */
export function servingOf(experimentId: string): TensorboardServing {
  const launch = LAUNCHES[experimentId];
  if (launch?.kind !== "serving") {
    throw new Error(`the generator served no TensorBoard for ${experimentId}`);
  }
  return launch;
}
