/** Meta-loop API helpers — Claude's friction reports and retrospectives. */

export interface FrictionReport {
  file: string;
  timestamp: string | null;
  category: string;
  detail: string;
  context: Record<string, unknown>;
}

export interface Retrospective {
  project_id: string;
  modified: string;
  content: string;
}

export const metaApi = {
  reports: (project_root: string) =>
    fetch(`/api/meta/reports?project_root=${encodeURIComponent(project_root)}`).then(
      (r) =>
        r.json() as Promise<{
          reports: FrictionReport[];
          count: number;
          total_available: number;
        }>,
    ),

  retrospectives: (project_root: string) =>
    fetch(`/api/meta/retrospectives?project_root=${encodeURIComponent(project_root)}`).then(
      (r) =>
        r.json() as Promise<{
          retrospectives: Retrospective[];
          count: number;
          total_available: number;
        }>,
    ),
};
