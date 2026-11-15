const DISMISS_LABEL_WORDS = 6;

/** A dismiss button's own name, so two messages on screen at once seldom share one control
 *  name: the message's first few words, the whole message when it is already that short. */
export function dismissLabel(message: string): string {
  const words = message.trim().split(/\s+/);
  const lead = words.slice(0, DISMISS_LABEL_WORDS).join(" ");
  return `Dismiss: ${lead}${words.length > DISMISS_LABEL_WORDS ? "…" : ""}`;
}
