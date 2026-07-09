/**
 * Shared fetch helpers. Every API call goes through `asJson`, so a non-2xx response
 * surfaces as a thrown Error carrying the backend's `detail` — instead of being
 * silently parsed as if it were a success body (which yielded `undefined` fields and
 * crashed callers on the next render). Callers catch and route errors to a toast.
 */

export async function asJson<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = "";
    try {
      detail = ((await r.json()) as { detail?: string })?.detail ?? "";
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail || `${r.status} ${r.statusText}`);
  }
  return (await r.json()) as T;
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
