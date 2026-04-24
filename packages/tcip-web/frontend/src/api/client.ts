/**
 * Typed REST client for the tcip-web backend.
 * All routes hit /api/* and return typed payloads.
 */

import type {
  Box,
  Detection,
  DatasetSelection,
  GuiState,
  ImageLabels,
  MatchesResponse,
  PolygonShape,
  PredictionReference,
} from "@/store/types";

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  return (await resp.json()) as T;
}

function q(params: Record<string, string | number | boolean | null | undefined>) {
  const u = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === null || v === undefined) continue;
    u.set(k, String(v));
  }
  return u.toString();
}

export const api = {
  state: (): Promise<GuiState> => call("/api/state"),

  dataset: {
    tree: (dataset_root: string) =>
      call<{
        dataset_root: string;
        dates_with_images: string[];
        annotation_types: string[];
        model_names: string[];
      }>(`/api/dataset/tree?${q({ dataset_root })}`),

    listImages: (dataset_root: string, date: string) =>
      call<{ images: string[]; count: number; dataset_root: string; date: string }>(
        `/api/dataset/images?${q({ dataset_root, date })}`,
      ),

    select: (body: {
      project_root: string;
      dataset_root: string;
      annotation_type?: string | null;
      date?: string | null;
      model_name?: string | null;
    }) =>
      call<{ status: string; selection: DatasetSelection }>("/api/dataset/select", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },

  images: {
    url: (path: string, max_width?: number) =>
      `/api/images?${q({ path, max_width })}`,

    dimensions: (path: string) =>
      call<{ path: string; width: number; height: number }>(
        `/api/images/dimensions?${q({ path })}`,
      ),
  },

  annotate: {
    load: (image_path: string, detect_path?: string | null, segment_path?: string | null) =>
      call<ImageLabels>(
        `/api/annotate/labels?${q({ image_path, detect_path, segment_path })}`,
      ),

    save: (body: {
      image_path: string;
      detect_path?: string | null;
      segment_path?: string | null;
      boxes: Box[];
      polygons: PolygonShape[];
    }) =>
      call<{ status: string; n_boxes: number; n_polygons: number }>("/api/annotate/labels", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    openImage: (body: {
      image_path: string;
      image_index?: number;
      scale?: number;
      offset_x?: number;
      offset_y?: number;
      mode?: string;
      pred_reference?: PredictionReference | null;
    }) =>
      call<{ status: string; image_path: string }>("/api/annotate/open", {
        method: "POST",
        body: JSON.stringify(body),
      }),
  },

  review: {
    matches: (body: {
      project_root: string;
      image_name: string;
      image_path: string;
      gt_detect_path?: string | null;
      gt_segment_path?: string | null;
      pred_detect_path?: string | null;
      pred_segment_path?: string | null;
      iou_threshold?: number;
      conf_threshold?: number;
      filter_type?: string;
      filter_class?: string | number;
      status_filter?: string;
    }) =>
      call<MatchesResponse>("/api/review/matches", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    action: (body: {
      project_root: string;
      image_name: string;
      image_path: string;
      gt_detect_path?: string | null;
      gt_segment_path?: string | null;
      pred_detect_path?: string | null;
      pred_segment_path?: string | null;
      det_type: string;
      class_id: number;
      conf?: number | null;
      iou?: number | null;
      gt_type?: string | null;
      gt_idx?: number | null;
      pred_type?: string | null;
      pred_idx?: number | null;
      bbox: [number, number, number, number];
      action: "accepted" | "rejected" | "edited";
    }) =>
      call<{ status: string }>("/api/review/action", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    backupLabels: (project_root: string, label_dirs: string[]) =>
      call<{ status: string; labels_backed_up: boolean }>("/api/review/backup_labels", {
        method: "POST",
        body: JSON.stringify({ project_root, label_dirs }),
      }),

    saveGt: (body: {
      project_root: string;
      image_name: string;
      image_path: string;
      detect_path?: string | null;
      segment_path?: string | null;
      boxes: Box[];
      polygons: PolygonShape[];
    }) =>
      call<{ status: string }>("/api/review/save_gt", {
        method: "POST",
        body: JSON.stringify(body),
      }),

    imageStatus: (project_root: string, image_name: string) =>
      call<{ status: string }>(
        `/api/review/image_status?${q({ project_root, image_name })}`,
      ),
  },
};

export type { Detection };
