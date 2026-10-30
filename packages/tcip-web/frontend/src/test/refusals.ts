import { vi } from "vitest";

import { decodeRefusal } from "@/api/http";

/** A refusal a fixture generator received (`tools/fixture_session.answered`): the status the route
 *  answered and the body it sent, whose detail is `D`. */
export interface Refusal<D> {
  status: number;
  body: { detail: D };
}

/** A route's answer as `fetch` resolves it: `status`, with `body` sent as JSON (no body when it is
 *  omitted) and `headers`. */
export function answeredWith(
  status: number,
  body?: unknown,
  headers: Record<string, string> = {},
): Response {
  return responded(status, body === undefined ? null : JSON.stringify(body), headers);
}

/** A route's answer as `fetch` resolves it: `status`, sending `content` as it is, with `headers`. */
function responded(status: number, content: string | null, headers: Record<string, string>) {
  return new Response(content, { status, headers });
}

/** Make every `fetch` resolve a fresh answer `answer` makes; return the mock. */
function fetching(answer: () => Response) {
  const fetch = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) =>
    Promise.resolve(answer()),
  );
  vi.stubGlobal("fetch", fetch);
  return fetch;
}

/** Make every `fetch` resolve a fresh `answeredWith(status, body, headers)`; return the mock. */
export function fetchAnswers(status: number, body?: unknown, headers?: Record<string, string>) {
  return fetching(() => answeredWith(status, body, headers));
}

/** Make every `fetch` resolve a 200 sending the file `text` with `headers`; return the mock. */
export function fetchFile(text: string, headers: Record<string, string>) {
  return fetching(() => responded(200, text, headers));
}

/** The error the client decodes from the produced `refusal`, as a route call rejects with it. */
export function thrown(refusal: Refusal<unknown>): Promise<Error> {
  return decodeRefusal(answeredWith(refusal.status, refusal.body));
}
