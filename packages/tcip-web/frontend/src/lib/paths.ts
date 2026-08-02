/** Normalize a user-pasted filesystem path. */

/**
 * Trim whitespace and strip surrounding quotes. Windows "Copy as path" wraps the path in
 * double quotes, and pasting from an address bar often adds stray spaces; either makes
 * the backend reject an otherwise-correct path, so clean it before use.
 */
export function cleanPath(raw: string): string {
  return raw
    .trim()
    .replace(/^["']+|["']+$/g, "")
    .trim();
}
