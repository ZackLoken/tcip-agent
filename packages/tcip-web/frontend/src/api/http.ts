/**
 * Shared fetch helpers. A non-2xx response throws an Error carrying the backend's `detail`, a
 * string or, as a `StructuredRefusalError`, an object, decoded once by `decodeRefusal` whatever
 * the body is read as.
 */

import { AUDIT_ENTRY_NOT_WRITTEN, type AuditGapRefusal } from "@/api/types.generated";

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
    // A list is a request model's own refusal: one entry per field it refused, naming its place.
    const message = Array.isArray(detail)
      ? (detail as { loc: (string | number)[]; msg: string }[])
          .map((issue) => `${issue.loc.slice(1).join(".")}: ${issue.msg}`)
          .join("; ")
      : typeof parsed.message === "string"
        ? parsed.message
        : "";
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

/** The refusal `R` a thrown error carries, read by its detail's `kind` at any status, or null for
 *  every other failure. */
export function refusalOf<R extends { kind: string }>(e: unknown, kind: R["kind"]): R | null {
  return e instanceof StructuredRefusalError && e.detail.kind === kind
    ? (e.detail as unknown as R)
    : null;
}

/** The `committed` body an audit-gap refusal carries: the response a healthy call would have
 *  returned, so a caller adopts it and reaches the state a 200 would have left it in. Null when
 *  `e` is not one of these refusals, or the route itself could not say what committed. The one
 *  decode of a body the refusal carries uninterpreted. */
export function committedOf<T>(e: unknown): T | null {
  const gap = refusalOf<AuditGapRefusal>(e, AUDIT_ENTRY_NOT_WRITTEN);
  return (gap?.committed ?? null) as T | null;
}

/** The text a caught `e` reads as: an error's own message, anything else as it prints. */
export function messageOf(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
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
