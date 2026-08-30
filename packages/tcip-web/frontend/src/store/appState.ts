import type { AgentActivitySlice } from "@/store/slices/agentActivity";
import type { BandSelectionSlice } from "@/store/slices/bandSelection";
import type { BannerSlice } from "@/store/slices/banners";
import type { CanvasSlice } from "@/store/slices/canvas";
import type { GuiSlice } from "@/store/slices/gui";
import type { PendingTerminalMessageSlice } from "@/store/slices/pendingTerminalMessage";
import type { RegistryStatusSlice } from "@/store/slices/registryStatus";
import type { ReviewSlice } from "@/store/slices/review";
import type { TerminalOpenSlice } from "@/store/slices/terminalOpen";
import type { ToastSlice } from "@/store/slices/toasts";
import type { UserSlice } from "@/store/slices/user";

/**
 * The whole store's shape, one slice interface per labelled group in the create() call that
 * composes them: each slice file stays a self-contained read of its own group, and this is the
 * one place they combine into the type every slice's StateCreator is typed against.
 */
export interface AppState
  extends
    GuiSlice,
    CanvasSlice,
    ReviewSlice,
    BandSelectionSlice,
    RegistryStatusSlice,
    AgentActivitySlice,
    ToastSlice,
    BannerSlice,
    TerminalOpenSlice,
    PendingTerminalMessageSlice,
    UserSlice {}
