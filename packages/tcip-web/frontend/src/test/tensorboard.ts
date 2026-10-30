import type { TensorboardServing } from "@/api/types.generated";
import produced from "@/test/tensorboardServings.json";

/** What `tools/generate_tensorboard_fixture.py` received from the TensorBoard route, by run id:
 *  one serving for each of two runs, each over its own board. */
export const TENSORBOARD_SERVINGS = produced as unknown as Record<string, TensorboardServing>;

/** The produced serving of the run `experimentId` names; an id the generator served no
 *  TensorBoard for throws. */
export function servingOf(experimentId: string): TensorboardServing {
  const serving = TENSORBOARD_SERVINGS[experimentId];
  if (!serving) throw new Error(`the generator served no TensorBoard for ${experimentId}`);
  return serving;
}
