/** Reading one thing again and again, one read at a time. */

/** A running loop: `readNow` reads at once, or once more as soon as the read in flight answers,
 *  whether or not a handler stopped the interval reads; `stop` ends it, and an answer arriving
 *  after is dropped. */
export interface PollLoop {
  readNow: () => void;
  stop: () => void;
}

/** What a loop reads and what it does with each outcome; each handler returns whether to read
 *  again after the interval. */
export interface PollReads<T> {
  read: () => Promise<T>;
  /** The wait between one read's answer and the next read. */
  intervalMs: number;
  /** Read at once, or only after the first interval. */
  readFirstNow: boolean;
  onAnswer: (answer: T) => boolean;
  onFailure: (error: unknown) => boolean;
}

/**
 * Start a loop over `reads.read`: each read waits for the last one's answer, then the interval,
 * so no two reads are ever outstanding. A handler returning false leaves the loop reading only
 * when `readNow` asks; `stop` ends it, a handler's own call included.
 */
export function pollLoop<T>({
  read,
  intervalMs,
  readFirstNow,
  onAnswer,
  onFailure,
}: PollReads<T>): PollLoop {
  let stopped = false;
  let inFlight = false;
  let askedInFlight = false;
  let timer: ReturnType<typeof setTimeout> | null = null;

  function run() {
    timer = null;
    inFlight = true;
    read().then(
      (answer) => {
        if (!stopped) settle(onAnswer(answer));
      },
      (error: unknown) => {
        if (!stopped) settle(onFailure(error));
      },
    );
  }

  function settle(again: boolean) {
    inFlight = false;
    if (stopped) return;
    if (askedInFlight) {
      askedInFlight = false;
      run();
    } else if (again) {
      timer = setTimeout(run, intervalMs);
    }
  }

  if (readFirstNow) run();
  else timer = setTimeout(run, intervalMs);

  return {
    readNow: () => {
      if (stopped) return;
      if (inFlight) {
        askedInFlight = true;
        return;
      }
      if (timer !== null) clearTimeout(timer);
      run();
    },
    stop: () => {
      stopped = true;
      if (timer !== null) clearTimeout(timer);
    },
  };
}
