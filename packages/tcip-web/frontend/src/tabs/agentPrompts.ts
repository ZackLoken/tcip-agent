/**
 * Plain-language requests the run tabs stage for the agent, editable before they're sent.
 *
 * Each one states what the GUI knows (the selection the breeder made) and leaves every CV/ML
 * decision to the agent: routing those decisions through it, rather than through a form asking
 * the breeder for the same values, is the point of the hand-off.
 */

function scopePhrase(datasetRoot: string | null, subject: string | null): string {
  if (datasetRoot && subject) return ` for the dataset at ${datasetRoot}, subject ${subject}`;
  if (datasetRoot) return ` for the dataset at ${datasetRoot}`;
  if (subject) return ` for subject ${subject}`;
  return "";
}

export function defaultTrainingRequest(datasetRoot: string | null, subject: string | null): string {
  return (
    `Configure and launch a training run${scopePhrase(datasetRoot, subject)}. Pick the model ` +
    "and training config (architecture, task, batch size, schedule) that suit this data, " +
    "check the config is sound before launching it, then launch the run. Check how large " +
    "this dataset's objects are relative to the full-frame resolution and decide from that " +
    "whether to train on tiles or on whole frames, then tell me which you chose and why. If " +
    "you train tiled on full-width strip-layout rasters read windowed, also consider the " +
    "sampler choice: shuffled tile access forces repeated strip decodes there. " +
    "Let me know once it's running so I can watch it here."
  );
}

export function defaultSweepRequest(datasetRoot: string | null): string {
  const where = datasetRoot ? ` for the dataset at ${datasetRoot}` : "";
  return (
    `Run a hyperparameter sweep${where}. Pick the training config and task, decide which ` +
    "hyperparameters are worth searching (e.g. learning rate, batch size, weight decay) and " +
    "reasonable ranges for them, choose a search algorithm and scheduler and a sensible " +
    "number of trials, then launch it. Let me know once it's running so I can watch it here."
  );
}
