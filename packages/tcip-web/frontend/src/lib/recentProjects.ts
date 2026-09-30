/**
 * The ids of the last few projects the user opened, most recent first, for the status-bar
 * fast-track; each is named through the current project listing. Stored in localStorage.
 */

const KEY = "tcip.recent_projects";
const MAX = 5;

/** The stored ids, or none when nothing is stored. A stored value that is not a JSON list of
 *  ids throws, naming what it holds. */
export function loadRecentProjectIds(): string[] {
  const raw = localStorage.getItem(KEY);
  if (raw === null) return [];
  const parsed: unknown = JSON.parse(raw);
  if (!Array.isArray(parsed) || !parsed.every((id) => typeof id === "string" && id)) {
    throw new Error(`the recent-projects list in this browser does not hold project ids: ${raw}`);
  }
  return parsed as string[];
}

/** Record a project as most-recently opened (dedup by id, cap at MAX). */
export function recordRecentProject(id: string): void {
  const list = loadRecentProjectIds().filter((p) => p !== id);
  list.unshift(id);
  localStorage.setItem(KEY, JSON.stringify(list.slice(0, MAX)));
}

/** Drop a project from this browser's own recents after it is removed. */
export function forgetRecentProject(id: string): void {
  localStorage.setItem(KEY, JSON.stringify(loadRecentProjectIds().filter((p) => p !== id)));
}
