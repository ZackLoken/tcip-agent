import { create } from "zustand";

import { subjectColor } from "@/api/classes";
import type { AppState } from "@/store/appState";
import { createAgentActivitySlice } from "@/store/slices/agentActivity";
import { createBandSelectionSlice } from "@/store/slices/bandSelection";
import { createBannerSlice } from "@/store/slices/banners";
import { createCanvasSlice } from "@/store/slices/canvas";
import { createGuiSlice } from "@/store/slices/gui";
import { createPendingTerminalMessageSlice } from "@/store/slices/pendingTerminalMessage";
import { createRegistryStatusSlice } from "@/store/slices/registryStatus";
import { createReviewSlice } from "@/store/slices/review";
import { createTerminalOpenSlice } from "@/store/slices/terminalOpen";
import { createToastSlice } from "@/store/slices/toasts";
import { createUserSlice } from "@/store/slices/user";

export type { AppState } from "@/store/appState";
export type { AgentActivity } from "@/store/slices/agentActivity";
export type { Banner } from "@/store/slices/banners";
export type { CanvasState } from "@/store/slices/canvas";
export type { Toast } from "@/store/slices/toasts";

/** One slice file per labelled group of store state; each slice's own file is the full read of
 *  that group's state and actions, and this create() call is the only place they compose.
 *  Cross-group reads/writes (e.g. setReviewDetectionIdx writing into gui.review) still work:
 *  every slice shares the same set/get, since they all belong to this one store. */
export const useStore = create<AppState>()((...a) => ({
  ...createGuiSlice(...a),
  ...createCanvasSlice(...a),
  ...createReviewSlice(...a),
  ...createBandSelectionSlice(...a),
  ...createRegistryStatusSlice(...a),
  ...createAgentActivitySlice(...a),
  ...createToastSlice(...a),
  ...createBannerSlice(...a),
  ...createTerminalOpenSlice(...a),
  ...createPendingTerminalMessageSlice(...a),
  ...createUserSlice(...a),
}));

// Re-export so callers derive a subject's colour from one source (GUI-local, name-hashed).
export { subjectColor };
