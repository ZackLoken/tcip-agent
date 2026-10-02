import { Suspense, lazy, useEffect, useRef, type ReactNode } from "react";

import { subjectsApi } from "@/api/subjects";
import { ROUTES } from "@/api/routes";
import {
  PANEL_EVENT_ANNOTATE_FOCUS,
  PANEL_EVENT_CANVAS_STATE_REQUEST,
  TAB_NAMES,
} from "@/api/types.generated";
import { sessionsApi } from "@/api/sessions";
import { ProjectPicker } from "@/components/ProjectPicker";
import { TerminalRail } from "@/components/TerminalRail";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { HelpOverlay } from "@/components/HelpOverlay";
import { StatusBar } from "@/components/StatusBar";
import { TabBanner } from "@/components/TabBanner";
import { Toasts } from "@/components/Toasts";
import { TopBar, tabButtonId, tabPanelId } from "@/components/TopBar";
import { stateSocket } from "@/api/ws";
import { useActiveTabSync } from "@/hooks/useActiveTabSync";
import { applyAnnotateFocus, type AnnotateFocusData } from "@/lib/annotateFocus";
import { notifyCanvasStateRequest } from "@/lib/canvasSync";
import { attachCtrlWheelGuard } from "@/lib/ctrlWheelGuard";
import { useStore } from "@/store";
import { declaredClient } from "@/store/slices/agentActivity";
import { selectProjectRoot } from "@/store/slices/gui";
import type { TabName } from "@/store/types";
import { AnnotateTab } from "@/tabs/AnnotateTab";
import { MetaTab } from "@/tabs/MetaTab";

// Every tab has an agent panel of the same name (the backend's own panel set also carries
// "app", handled by its own subscription below).
const TAB_PANELS: readonly TabName[] = TAB_NAMES;

// Code-split the recharts-heavy tabs (recharts + its d3 deps are ~5MB unpacked and used only
// here) so the Annotate workflow (the primary use) paints without them. App mounts
// exactly one tab at a time, so deferring these chunks costs no UX.
const InferenceTab = lazy(() =>
  import("@/tabs/InferenceTab").then((m) => ({ default: m.InferenceTab })),
);
const ResultsTab = lazy(() => import("@/tabs/ResultsTab").then((m) => ({ default: m.ResultsTab })));
const SetupTab = lazy(() => import("@/tabs/SetupTab").then((m) => ({ default: m.SetupTab })));
const TrainingTab = lazy(() =>
  import("@/tabs/TrainingTab").then((m) => ({ default: m.TrainingTab })),
);
const TuningTab = lazy(() => import("@/tabs/TuningTab").then((m) => ({ default: m.TuningTab })));

function TabFallback() {
  return (
    <div className="flex-1 flex items-center justify-center bg-tcip-canvas text-xs text-tcip-muted">
      Loading…
    </div>
  );
}

function App() {
  const activeTab = useStore((s) => s.gui.active_tab);
  // A dataset with a date is selected: the canvas tabs this gates need an image directory, so an
  // open project with no dated images still shows the picker here.
  const datasetReady = useStore((s) => !!s.gui.dataset.dataset_root && !!s.gui.dataset.date);
  const projectRoot = useStore(selectProjectRoot);
  const datasetKey = useStore(
    (s) =>
      `${s.gui.dataset.dataset_root ?? ""}::${s.gui.dataset.subject ?? ""}::${s.gui.dataset.date ?? ""}`,
  );
  const imageList = useStore((s) => s.gui.dataset.image_list);
  const annotationsDir = useStore((s) => s.gui.dataset.annotations_dir);
  const subject = useStore((s) => s.gui.dataset.subject);
  const datasetRoot = useStore((s) => s.gui.dataset.dataset_root);
  const setRegistry = useStore((s) => s.setRegistry);
  const endedSessionForRoot = useRef<string | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    stateSocket.connect();
    return () => stateSocket.close();
  }, []);

  // Every active_tab writer converges here; the backend GUI state follows what is on screen.
  useActiveTabSync();

  // Browser ctrl+wheel zoom stays off inside the app; iframe-embedded tools (TensorBoard, Ray)
  // receive wheel events inside their own documents and keep it.
  useEffect(() => {
    if (!rootRef.current) return;
    return attachCtrlWheelGuard(rootRef.current);
  }, []);

  // Subscribe to agent panel pushes for every tab: a "banner" event becomes that tab's note
  // (rendered by TabBanner), and the Annotate panel's pushes also drive the agent-activity
  // indicator the StatusBar and the Annotate tab's refresh prompt read.
  useEffect(() => {
    const unsubscribes = TAB_PANELS.map((panel) =>
      stateSocket.subscribePanel(panel, (ev) => {
        if (ev.event_type === "banner") {
          const text = ev.data.text;
          if (typeof text === "string") useStore.getState().pushBanner(ev.panel, ev.event_id, text);
          return;
        }
        if (ev.panel === "annotate") {
          useStore
            .getState()
            .pushAgentActivity(ev.panel, ev.event_type, ev.data, declaredClient(ev));
        }
      }),
    );
    return () => unsubscribes.forEach((unsubscribe) => unsubscribe());
  }, []);

  // Agent → GUI steering on the "app" panel: banners, focus, and canvas refresh requests.
  useEffect(() => {
    const unsubscribe = stateSocket.subscribePanel("app", (ev) => {
      if (ev.event_type === "banner") {
        const text = ev.data.text;
        if (typeof text === "string") useStore.getState().pushBanner(ev.panel, ev.event_id, text);
        return;
      }

      // Agent → GUI "focus the Annotate tab" through local setters (see applyAnnotateFocus).
      if (ev.event_type === PANEL_EVENT_ANNOTATE_FOCUS) {
        void applyAnnotateFocus(ev.data as AnnotateFocusData).catch(() => {
          useStore
            .getState()
            .pushToast("Agent tried to focus the Annotate tab, but it couldn't be applied.");
        });
        return;
      }

      // Agent → GUI "push your canvas now": capture_live_canvas pings before rendering so it
      // sees the freshest state; the mounted tab answers with an immediate full push.
      if (ev.event_type === PANEL_EVENT_CANVAS_STATE_REQUEST) {
        notifyCanvasStateRequest();
        return;
      }
    });
    return unsubscribe;
  }, []);

  useEffect(() => {
    if (!projectRoot) return;
    // Best-effort telemetry: never surface a failure to the user.
    void sessionsApi.start(useStore.getState().user).catch(() => {});
    endedSessionForRoot.current = null;

    function endSession() {
      if (endedSessionForRoot.current === projectRoot) return;
      endedSessionForRoot.current = projectRoot;

      const payload = JSON.stringify({});
      try {
        if (navigator.sendBeacon) {
          const blob = new Blob([payload], { type: "application/json" });
          navigator.sendBeacon(ROUTES.postSessionsEnd, blob);
          return;
        }
      } catch {
        // Fall through to fetch keepalive.
      }

      try {
        void fetch(ROUTES.postSessionsEnd, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
          keepalive: true,
        });
      } catch {
        // Best-effort telemetry only.
      }
    }

    window.addEventListener("pagehide", endSession);
    window.addEventListener("beforeunload", endSession);
    return () => {
      window.removeEventListener("pagehide", endSession);
      window.removeEventListener("beforeunload", endSession);
      endSession();
    };
  }, [projectRoot]);

  // A refresh/close with unsaved canvas edits gets the leave-page prompt (React never unmounts).
  useEffect(() => {
    function guardUnload(e: BeforeUnloadEvent) {
      if (useStore.getState().canvas.dirty) e.preventDefault();
    }
    window.addEventListener("beforeunload", guardUnload);
    return () => window.removeEventListener("beforeunload", guardUnload);
  }, []);

  // Hydrate the subject registry whenever the dataset selection changes.
  useEffect(() => {
    if (!projectRoot || !datasetRoot || imageList.length === 0) return;
    void (async () => {
      try {
        const reg = await subjectsApi.load(datasetRoot, annotationsDir);
        setRegistry(reg.subjects, reg.version);
        if (reg.unreadable.length) {
          useStore
            .getState()
            .pushToast(
              `${reg.unreadable.length} label file(s) could not be read: ${reg.unreadable.join(", ")}`,
            );
        }
        // Default the active authoring subject to the selection's subject when it exists in the
        // registry, else the first declared subject: a shape can't be authored with none set.
        const names = Object.keys(reg.subjects);
        const active = useStore.getState().gui.active_subject;
        if (!active || !names.includes(active)) {
          useStore
            .getState()
            .setActiveSubject(subject && names.includes(subject) ? subject : (names[0] ?? null));
        }
      } catch (err) {
        console.warn("registry hydrate failed", err);
        useStore.getState().pushToast("Could not load the registry for this project.");
      }
    })();
  }, [projectRoot, datasetKey, imageList, subject, datasetRoot, annotationsDir, setRegistry]);

  // Only Annotate / Results need an imagery dataset+date and show the picker until one is set;
  // Setup needs an open project. Keyed by TabName, so an added tab with no entry fails the typecheck.
  const tabPanels: Record<TabName, ReactNode> = {
    setup: projectRoot ? <SetupTab /> : <ProjectPicker />,
    annotate: datasetReady ? <AnnotateTab /> : <ProjectPicker />,
    results: datasetReady ? <ResultsTab /> : <ProjectPicker />,
    training: <TrainingTab />,
    tuning: <TuningTab />,
    inference: <InferenceTab />,
    meta: <MetaTab />,
  };

  return (
    <div ref={rootRef} className="h-full flex flex-col bg-tcip-bg text-tcip-fg">
      <TopBar />
      {/* The agent rail docks to the right; the tabs are its canvas, driven through the
          MCP panel channel. */}
      <div className="flex-1 flex min-h-0">
        <div className="flex-1 flex flex-col min-w-0 min-h-0">
          <TabBanner />
          <div
            id={tabPanelId(activeTab)}
            role="tabpanel"
            aria-labelledby={tabButtonId(activeTab)}
            className="flex-1 flex flex-col min-w-0 min-h-0"
          >
            <ErrorBoundary resetKey={activeTab}>
              <Suspense fallback={<TabFallback />}>{tabPanels[activeTab]}</Suspense>
            </ErrorBoundary>
          </div>
        </div>
        <TerminalRail />
      </div>
      <StatusBar />
      <HelpOverlay activeTab={activeTab} />
      <Toasts />
    </div>
  );
}

export default App;
