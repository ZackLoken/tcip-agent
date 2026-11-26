import { Suspense, lazy, useEffect, type ReactNode } from "react";

import { subjectsApi } from "@/api/subjects";
import {
  PANEL_EVENT_ANNOTATE_FOCUS,
  PANEL_EVENT_BANNER,
  PANEL_EVENT_CANVAS_STATE_REQUEST,
  TAB_NAMES,
  type PanelEnvelope,
} from "@/api/types.generated";
import { stateSocket } from "@/api/ws";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { HelpOverlay } from "@/components/HelpOverlay";
import { ProjectPicker } from "@/components/ProjectPicker";
import { StatusBar } from "@/components/StatusBar";
import { TabBanner } from "@/components/TabBanner";
import { TerminalRail } from "@/components/TerminalRail";
import { tabButtonId, tabPanelId } from "@/components/TopBar";
import { useActiveTabSync } from "@/hooks/useActiveTabSync";
import { useJobListings } from "@/hooks/useJobListings";
import { applyAnnotateFocus } from "@/lib/annotateFocus";
import { notifyCanvasStateRequest } from "@/lib/canvasSync";
import { adoptLoad } from "@/lib/registrySave";
import { endRecordedSession } from "@/lib/sessionLifecycle";
import { TAB_LABELS } from "@/lib/tabLabels";
import { takingOnce } from "@/lib/takenEvents";
import { useStore } from "@/store";
import { UNDECLARED_ACTOR } from "@/store/slices/agentActivity";
import { selectProjectRoot } from "@/store/slices/gui";
import type { DatasetSelection, TabName } from "@/store/types";
import { AnnotateTab } from "@/tabs/AnnotateTab";
import { MetaTab } from "@/tabs/MetaTab";

// Every tab has an agent panel of the same name (the backend's own panel set also carries
// "app", handled by its own subscription below).
const TAB_PANELS: readonly TabName[] = TAB_NAMES;

// Code-split every tab but Annotate (ResultsTab carries recharts and its d3 deps, ~5MB unpacked)
// so Annotate paints without them; the shell mounts one tab at a time, so deferring costs no UX.
const InferenceTab = lazy(() =>
  import("@/tabs/InferenceTab").then((m) => ({ default: m.InferenceTab })),
);
const ResultsTab = lazy(() => import("@/tabs/ResultsTab").then((m) => ({ default: m.ResultsTab })));
const SetupTab = lazy(() => import("@/tabs/SetupTab").then((m) => ({ default: m.SetupTab })));
const TrainingTab = lazy(() =>
  import("@/tabs/TrainingTab").then((m) => ({ default: m.TrainingTab })),
);

/** Record a banner event as its tab's note; false for any other event. */
function takeBanner({ panel, event, event_id }: PanelEnvelope): boolean {
  if (event.event_type !== PANEL_EVENT_BANNER) return false;
  useStore.getState().pushBanner(panel, event_id, event.text);
  return true;
}

/** List the agent's activity on the Annotate panel an envelope tells, as a notice of the project
 *  it was posted for naming who acted, by the client name its sender declared, and what; a repeat
 *  counts the listed notice. */
function noticeActivity({ event, identity, project_id }: PanelEnvelope) {
  const actor = identity.agent_client_name ?? UNDECLARED_ACTOR;
  useStore.getState().raiseNotification({
    kind: "notice",
    level: "info",
    text: { sentence: `${actor}: ${event.event_type.replaceAll("_", " ")}` },
    context: { project: project_id, target: TAB_LABELS.annotate },
  });
}

/** The capture whose subject registry the shell hydrates, as one comparable value: its root, date
 *  and subject; null with no selection or one holding no images. */
function registryCaptureOf(dataset: DatasetSelection | null): string | null {
  if (dataset === null || dataset.images.length === 0) return null;
  return JSON.stringify([dataset.dataset_root, dataset.date, dataset.subject]);
}

function TabFallback() {
  return (
    <div className="flex-1 flex items-center justify-center bg-tcip-canvas text-xs text-tcip-muted">
      Loading…
    </div>
  );
}

/**
 * Everything that runs on behalf of a project: the state socket and its snapshot adoption, the
 * agent's panel events, the job listings, the registry hydration, the session end, the agent
 * rail, the status bar and the tabs.
 */
export function ProjectShell() {
  const activeTab = useStore((s) => s.gui.active_tab);
  // A capture is selected: the canvas tabs this gates need one, so an open project with no
  // dated images still shows the picker here.
  const datasetReady = useStore((s) => s.gui.dataset !== null);
  const projectRoot = useStore(selectProjectRoot);
  const registryCapture = useStore((s) => registryCaptureOf(s.gui.dataset));

  useEffect(() => {
    stateSocket.connect();
    return () => stateSocket.close();
  }, []);

  // Every change to the active tab is sent to the backend's GUI state, debounced.
  useActiveTabSync();

  useJobListings();

  // Every tab's agent panel, each event taken once: a banner becomes that tab's note, and any
  // other Annotate event reaches the activity the tab reads, a live one also listed as a notice.
  useEffect(() => {
    const unsubscribes = TAB_PANELS.map((panel) =>
      stateSocket.subscribePanel(
        panel,
        takingOnce((ev) => {
          if (takeBanner(ev) || ev.panel !== "annotate") return;
          useStore.getState().pushAgentActivity(ev);
          if (!ev.sent_at_connect) noticeActivity(ev);
        }),
      ),
    );
    return () => unsubscribes.forEach((unsubscribe) => unsubscribe());
  }, []);

  // Agent → GUI steering on the "app" panel, each event taken once: banners, focus, and canvas
  // refresh requests.
  useEffect(() => {
    const unsubscribe = stateSocket.subscribePanel(
      "app",
      takingOnce((ev) => {
        if (takeBanner(ev)) return;

        const { event } = ev;
        // Agent → GUI "focus the Annotate tab" through local setters (see applyAnnotateFocus).
        if (event.event_type === PANEL_EVENT_ANNOTATE_FOCUS) {
          void applyAnnotateFocus(event, () => true).catch(() => {
            useStore.getState().raiseNotification({
              kind: "failure",
              level: "error",
              text: {
                sentence: "Agent tried to focus the Annotate tab, but it couldn't be applied.",
              },
              context: { project: ev.project_id },
            });
          });
          return;
        }

        // Agent → GUI "push your canvas now": the mounted tab answers with an immediate full push.
        if (event.event_type === PANEL_EVENT_CANVAS_STATE_REQUEST) {
          notifyCanvasStateRequest();
          return;
        }
      }),
    );
    return unsubscribe;
  }, []);

  // An end for the recorded session is sent when the page is hidden or about to unload and when
  // the shell unmounts, once the contributions queued before it have answered.
  useEffect(() => {
    window.addEventListener("pagehide", endRecordedSession);
    window.addEventListener("beforeunload", endRecordedSession);
    return () => {
      window.removeEventListener("pagehide", endRecordedSession);
      window.removeEventListener("beforeunload", endRecordedSession);
      endRecordedSession();
    };
  }, []);

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
    const { gui, openProject, raiseNotification } = useStore.getState();
    const held = gui.dataset;
    if (!openProject || registryCapture === null || held === null) return;
    const { dataset_root: datasetRoot, date, subject } = held;
    const context = { project: openProject.id, target: date };
    void (async () => {
      try {
        const reg = await subjectsApi.load(datasetRoot, date);
        adoptLoad(reg);
        const declared = useStore.getState().registry.subjects;
        if (reg.unreadable.length) {
          raiseNotification({
            kind: "failure",
            level: "error",
            text: {
              sentence: `${reg.unreadable.length} label document(s) could not be read: ${reg.unreadable.join(", ")}`,
            },
            context,
          });
        }
        // Default the active authoring subject to the selection's subject when it exists in the
        // registry, else the first declared subject: a shape can't be authored with none set.
        const names = Object.keys(declared);
        const active = useStore.getState().gui.active_subject;
        if (!active || !names.includes(active)) {
          useStore
            .getState()
            .setActiveSubject(subject && names.includes(subject) ? subject : (names[0] ?? null));
        }
      } catch (err) {
        console.warn("registry hydrate failed", err);
        raiseNotification({
          kind: "failure",
          level: "error",
          text: { sentence: "Could not load the registry for this project." },
          context,
        });
      }
    })();
  }, [projectRoot, registryCapture]);

  // Only Annotate / Results need an imagery dataset+date and show the picker until one is set;
  // Setup needs an open project. Keyed by TabName, so an added tab with no entry fails the typecheck.
  const tabPanels: Record<TabName, ReactNode> = {
    setup: projectRoot ? <SetupTab /> : <ProjectPicker />,
    annotate: datasetReady ? <AnnotateTab /> : <ProjectPicker />,
    results: datasetReady ? <ResultsTab /> : <ProjectPicker />,
    training: <TrainingTab />,
    inference: <InferenceTab />,
    meta: <MetaTab />,
  };

  return (
    <div className="flex-1 flex flex-col min-h-0">
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
    </div>
  );
}
