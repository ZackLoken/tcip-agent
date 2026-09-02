/**
 * Shared fetch helpers. Every API call goes through `asJson`, so a non-2xx response
 * surfaces as a thrown Error carrying the backend's `detail`, instead of being
 * silently parsed as if it were a success body (which yielded `undefined` fields and
 * crashed callers on the next render). Callers catch and route errors to a toast.
 *
 * A backend refusal carries `detail` as either a string or an object. `decodeRefusal` reads a
 * non-2xx body once and is the only place either shape is turned into an error, so a path that
 * reads its own body (a blob download) branches on the same object a JSON call would see rather
 * than stringifying it to `[object Object]`.
 */

/** A refusal whose `detail` is an object, kept parsed so a caller can branch on its own fields. */
export class StructuredRefusalError extends Error {
  readonly detail: Record<string, unknown>;
  readonly status: number;

  constructor(detail: Record<string, unknown>, status: number, message: string) {
    super(message);
    this.name = "StructuredRefusalError";
    this.detail = detail;
    this.status = status;
  }
}

export async function decodeRefusal(r: Response, fallback = ""): Promise<Error> {
  let detail: unknown;
  try {
    detail = ((await r.json()) as { detail?: unknown })?.detail;
  } catch {
    /* non-JSON error body */
  }
  const status = fallback || `${r.status} ${r.statusText}`;
  if (typeof detail === "object" && detail !== null) {
    const parsed = detail as Record<string, unknown>;
    const message = typeof parsed.message === "string" ? parsed.message : "";
    return new StructuredRefusalError(parsed, r.status, message || status);
  }
  // A string detail still carries a status a caller may branch on; a body with no detail at
  // all has nothing structured to carry and stays a plain Error.
  if (typeof detail === "string") {
    return new StructuredRefusalError({ message: detail }, r.status, detail);
  }
  return new Error(status);
}

export async function asJson<T>(r: Response): Promise<T> {
  if (!r.ok) {
    throw await decodeRefusal(r);
  }
  return (await r.json()) as T;
}

/** Absolute ws:// or wss:// URL for a backend socket path, matching the page's own scheme. */
export function wsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}${path}`;
}

const JSON_HEADERS = { "Content-Type": "application/json" };

export function getJson<T>(url: string): Promise<T> {
  return fetch(url).then((r) => asJson<T>(r));
}

export function postJson<T>(url: string, body: unknown): Promise<T> {
  return fetch(url, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
  }).then((r) => asJson<T>(r));
}
