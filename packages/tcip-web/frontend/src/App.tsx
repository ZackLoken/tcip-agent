import { Suspense, lazy, useEffect, useRef } from "react";

import { classesApi } from "@/api/classes";
import { sessionsApi } from "@/api/sessions";
import { ProjectPicker } from "@/components/ProjectPicker";
import { TerminalRail } from "@/components/TerminalRail";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { HelpOverlay } from "@/components/HelpOverlay";
import { StatusBar } from "@/components/StatusBar";
import { Toasts } from "@/components/Toasts";
import { TopBar } from "@/components/TopBar";
import { stateSocket } from "@/api/ws";
import { applyAnnotateFocus, type AnnotateFocusData } from "@/lib/annotateFocus";
import { notifyCanvasStateRequest } from "@/lib/canvasSync";
import { applyReviewFocus, type ReviewFocusData } from "@/lib/reviewFocus";
import { openProjectByName } from "@/lib/openProject";
import { useStore } from "@/store";
import { AnnotateTab } from "@/tabs/AnnotateTab";
import { MetaTab } from "@/tabs/MetaTab";
import { ReviewTab } from "@/tabs/ReviewTab";

// Code-split the recharts-heavy tabs (recharts + its d3 deps are ~5MB unpacked and used only
// here) so the Annotate/Review workflow (the primary use) paints without them. App mounts
// exactly one tab at a time, so deferring these chunks costs no UX.
const InferenceTab = lazy(() =>
  import("@/tabs/InferenceTab").then((m) => ({ default: m.InferenceTab })),
);
const ResultsTab = lazy(() => import("@/tabs/ResultsTab").then((m) => ({ default: m.ResultsTab })));
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
  const datasetReady = useStore((s) => !!s.gui.dataset.dataset_root && !!s.gui.dataset.date);
  const projectRoot = useStore((s) => s.gui.dataset.project_root);
  const datasetKey = useStore(
    (s) =>
      `${s.gui.dataset.dataset_root ?? ""}::${s.gui.dataset.subject ?? ""}::${s.gui.dataset.date ?? ""}`,
  );
  const imageList = useStore((s) => s.gui.dataset.image_list);
  const annotationsDir = useStore((s) => s.gui.dataset.annotations_dir);
  const subject = useStore((s) => s.gui.dataset.subject);
  const datasetRoot = useStore((s) => s.gui.dataset.dataset_root);
  const datasetDate = useStore((s) => s.gui.dataset.date);
  const setRegistry = useStore((s) => s.setRegistry);
  const setImageStatuses = useStore((s) => s.setImageStatuses);
  const endedSessionForRoot = useRef<string | null>(null);

  useEffect(() => {
    stateSocket.connect();
    return () => stateSocket.close();
  }, []);

  // Subscribe to agent panel pushes (e.g. the agent writing labels) and route
  // them into the store so the StatusBar can surface them and the Annotate tab
  // can offer to refresh. Previously push_panel_data had no consumer at all.
  useEffect(() => {
    const unsubscribe = stateSocket.subscribePanel("annotate", (ev) => {
      useStore.getState().pushAgentActivity(ev.panel, ev.event_type, ev.data);
    });
    return unsubscribe;
  }, []);

  // Agent → GUI "look here": when the agent sets the active project (e.g. after
  // ingesting a breeder's images), open it here so the GUI lands on what it built.
  useEffect(() => {
    const unsubscribe = stateSocket.subscribePanel("app", (ev) => {
      if (ev.event_type === "active_project_changed") {
        const name = (ev.data as { name?: string }).name;
        if (!name) return;
        void openProjectByName(name)
          .then((selection) => {
            if (!selection) return;
            if (!selection.date) {
              // No dated capture to land on: say so instead of appearing to do nothing.
              useStore
                .getState()
                .pushToast(`Opened ${name}, but it has no dated images yet.`, "info");
            }
          })
          .catch(() => {
            useStore
              .getState()
              .pushToast(`Agent opened a project but it couldn't be loaded: ${name}`);
          });
        return;
      }

      // Agent → GUI "focus the Annotate tab": land on a (subject, date) in the right mode on an
      // annotated frame (see applyAnnotateFocus, which uses local setters like Review→Edit so the
      // deliberate "mode/index stay local" behavior of mergeSnapshot is preserved).
      if (ev.event_type === "annotate_focus") {
        void applyAnnotateFocus(ev.data as AnnotateFocusData).catch(() => {
          useStore
            .getState()
            .pushToast("Agent tried to focus the Annotate tab, but it couldn't be applied.");
        });
        return;
      }

      // Agent → GUI "push your canvas now": capture_live_canvas pings before rendering so it
      // sees the freshest state; the mounted tab answers with an immediate full push.
      if (ev.event_type === "canvas_state_request") {
        notifyCanvasStateRequest();
        return;
      }

      // Agent → GUI "focus the Review tab": load a model's predictions on a frame/detection so
      // the human sees exactly what the agent flagged (a false positive, a missed detection).
      if (ev.event_type === "review_focus") {
        void applyReviewFocus(ev.data as ReviewFocusData).catch(() => {
          useStore
            .getState()
            .pushToast("Agent tried to focus the Review tab, but it couldn't be applied.");
        });
        return;
      }
    });
    return unsubscribe;
  }, []);

  // Session start / end.  Browser identifier is "web" since we don't have
  // OS user; the backend can rewrite this from a header later if needed.
  useEffect(() => {
    if (!projectRoot) return;
    const user = useStore.getState().user || "web";
    // Best-effort telemetry: never surface a failure to the user.
    void sessionsApi.start(projectRoot, user).catch(() => {});
    endedSessionForRoot.current = null;

    function endSession() {
      if (endedSessionForRoot.current === projectRoot) return;
      endedSessionForRoot.current = projectRoot;

      const payload = JSON.stringify({ project_root: projectRoot });
      try {
        if (navigator.sendBeacon) {
          const blob = new Blob([payload], { type: "application/json" });
          navigator.sendBeacon("/api/sessions/end", blob);
          return;
        }
      } catch {
        // Fall through to fetch keepalive.
      }

      try {
        void fetch("/api/sessions/end", {
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

  // A refresh/close with unsaved canvas edits gets the browser's leave-page prompt;
  // React never unmounts on unload, so AnnotateTab's flush-on-unmount can't cover this.
  useEffect(() => {
    function guardUnload(e: BeforeUnloadEvent) {
      if (useStore.getState().canvas.dirty) e.preventDefault();
    }
    window.addEventListener("beforeunload", guardUnload);
    return () => window.removeEventListener("beforeunload", guardUnload);
  }, []);

  // Hydrate the subject registry + per-image status whenever the dataset selection changes.
  useEffect(() => {
    if (!projectRoot || imageList.length === 0) return;
    void (async () => {
      try {
        const reg = await classesApi.load(projectRoot, datasetRoot, annotationsDir);
        setRegistry(reg.subjects);
        // Default the active authoring subject to the selection's subject when it exists in the
        // registry, else the first declared subject: a shape can't be authored with none set.
        const names = Object.keys(reg.subjects);
        const active = useStore.getState().gui.active_subject;
        if (!active || !names.includes(active)) {
          useStore
            .getState()
            .setActiveSubject(subject && names.includes(subject) ? subject : (names[0] ?? null));
        }

        // Scoped to the selected subject: a Complete recorded while annotating leaf says
        // nothing about bush, and reading a global map re-applied it to subjects the breeder
        // never looked at, then wrote the result back over their original confirmations.
        const saved = await classesApi.loadImageStatus(
          projectRoot,
          subject,
          datasetDate,
          datasetRoot,
          annotationsDir,
        );
        const savedMap = saved.statuses ?? {};
        // Reconcile every image against the label files, honoring confirmed reviews via
        // complete_override, so a wrongly-saved "negative" whose files have content heals
        // to partial instead of silently locking the canvas forever.
        const confirmed = imageList.filter(
          (name) => savedMap[name] === "complete" || savedMap[name] === "negative",
        );
        const derivedRes = await classesApi.deriveImageStatus({
          project_root: projectRoot,
          annotations_dir: annotationsDir,
          subject,
          image_list: imageList,
          complete_override: confirmed,
        });
        const reconciled = (derivedRes.statuses ?? {}) as Record<
          string,
          "complete" | "partial" | "negative" | "unannotated"
        >;
        const changed = Object.fromEntries(
          Object.entries(reconciled).filter(([name, st]) => savedMap[name] !== st),
        ) as Record<string, "complete" | "partial" | "negative" | "unannotated">;
        if (Object.keys(changed).length) {
          await classesApi.setImageStatusBulk(
            projectRoot,
            changed,
            subject,
            datasetDate,
            datasetRoot,
            annotationsDir,
          );
        }
        setImageStatuses(reconciled);
      } catch (err) {
        console.warn("registry / image-status hydrate failed", err);
        useStore
          .getState()
          .pushToast("Could not load the registry / image status for this project.");
      }
    })();
  }, [
    projectRoot,
    datasetKey,
    imageList,
    subject,
    datasetRoot,
    datasetDate,
    annotationsDir,
    setRegistry,
    setImageStatuses,
  ]);

  return (
    <div className="h-full flex flex-col bg-tcip-bg text-tcip-fg">
      <TopBar />
      {/* Only Annotate / Review / Results need an imagery dataset+date; the rest
          (Training / Tuning / Inference / Meta) are reachable without one; being
          forced to pick a dataset just to read agent reports or watch a run was a
          usability trap. Dataset-dependent tabs show the picker until one is set.
          The agent rail (the real Claude Code in a PTY) docks to the right; the tabs
          are its canvas: it drives them through the MCP panel channel. */}
      <div className="flex-1 flex min-h-0">
        <div className="flex-1 flex flex-col min-w-0 min-h-0">
          <ErrorBoundary resetKey={activeTab}>
            <Suspense fallback={<TabFallback />}>
              {activeTab === "annotate" && (datasetReady ? <AnnotateTab /> : <ProjectPicker />)}
              {activeTab === "review" && (datasetReady ? <ReviewTab /> : <ProjectPicker />)}
              {activeTab === "results" && (datasetReady ? <ResultsTab /> : <ProjectPicker />)}
              {activeTab === "training" && <TrainingTab />}
              {activeTab === "tuning" && <TuningTab />}
              {activeTab === "inference" && <InferenceTab />}
              {activeTab === "meta" && <MetaTab />}
            </Suspense>
          </ErrorBoundary>
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
