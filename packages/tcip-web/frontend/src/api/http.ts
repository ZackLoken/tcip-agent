/**
 * Shared fetch helpers. A non-2xx response throws an Error carrying the backend's `detail`, a
 * string or, as a `StructuredRefusalError`, an object, decoded once by `decodeRefusal` whatever
 * the body is read as.
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

/** The stable marker a route answers with when a mutation it already committed could not be
 *  recorded to the audit log (`routes/audit_gap.py`'s own name for it). Carried in `detail.error`
 *  at whatever status the route answers. */
export const AUDIT_ENTRY_NOT_WRITTEN = "audit_entry_not_written";

/** True when `e` is a refusal carrying that marker, at any status. */
export function isAuditEntryNotWritten(e: unknown): e is StructuredRefusalError {
  return e instanceof StructuredRefusalError && e.detail.error === AUDIT_ENTRY_NOT_WRITTEN;
}

/** The `committed` body an audit-gap refusal carries: the response a healthy call would have
 *  returned, so a caller adopts it and reaches the state a 200 would have left it in. Null when
 *  `e` is not one of these refusals, or the route itself could not say what committed. */
export function committedOf<T>(e: unknown): T | null {
  if (!isAuditEntryNotWritten(e)) return null;
  const committed = e.detail.committed;
  return committed === null || committed === undefined ? null : (committed as T);
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

function postBody(url: string, body: unknown, signal?: AbortSignal): Promise<Response> {
  return fetch(url, { method: "POST", headers: JSON_HEADERS, body: JSON.stringify(body), signal });
}

/** POST a JSON body; `signal` abandons the request, which then rejects with an AbortError. */
export function postJson<T>(url: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return postBody(url, body, signal).then((r) => asJson<T>(r));
}

/** POST a JSON body and read the answer as a file, with the headers it came with; a refusal
 *  throws through `decodeRefusal`, as `asJson`'s does. */
export async function postForBlob(
  url: string,
  body: unknown,
): Promise<{ blob: Blob; headers: Headers }> {
  const r = await postBody(url, body);
  if (!r.ok) throw await decodeRefusal(r);
  return { blob: await r.blob(), headers: r.headers };
}
