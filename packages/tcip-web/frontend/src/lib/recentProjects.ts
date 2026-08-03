/**
 * The last few projects the user opened, for the status-bar "project name" fast-track. Stored
 * in localStorage (UI convenience, not authoritative; the workspace list is). Most-recent first.
 */

export interface RecentProject {
  name: string;
  path: string;
}

const KEY = "tcip.recent_projects";
const MAX = 5;

export function loadRecentProjects(): RecentProject[] {
  try {
    const raw = localStorage.getItem(KEY);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((p) => p?.name && p?.path) : [];
  } catch {
    return [];
  }
}

/** Record a project as most-recently opened (dedup by name or path, cap at MAX).
 *
 * A moved project keeps its name and changes path, so matching on either replaces the entry that
 * describes the same project instead of leaving a stale one behind until the cap evicts it. */
export function recordRecentProject(name: string, path: string): void {
  if (!name || !path) return;
  try {
    const list = loadRecentProjects().filter((p) => p.path !== path && p.name !== name);
    list.unshift({ name, path });
    localStorage.setItem(KEY, JSON.stringify(list.slice(0, MAX)));
  } catch {
    /* private mode / disabled storage: the fast-track just won't remember */
  }
}
