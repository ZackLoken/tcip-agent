/** A typed text read as the JSON value it states. */

/** The JSON value `typed` states: the text parsed as JSON when it parses, else the text itself. */
export function jsonValue(typed: string): unknown {
  try {
    return JSON.parse(typed) as unknown;
  } catch {
    return typed;
  }
}
