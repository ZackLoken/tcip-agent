import { useCallback, useEffect, useRef, useState } from "react";

import { Dock } from "@/components/Dock";
import {
  NOTIFICATIONS_HEADING_ID,
  NotificationsButton,
  NotificationsLiveRegion,
} from "@/components/Notifications";
import { ProjectPicker } from "@/components/ProjectPicker";
import { TopBar } from "@/components/TopBar";
import { attachCtrlWheelGuard } from "@/lib/ctrlWheelGuard";
import { defaultDockModel, notificationsOpen, toggleNotifications } from "@/lib/dock";
import { ProjectShell } from "@/ProjectShell";
import { useStore } from "@/store";
import { selectAnnotatorNamed } from "@/store/slices/user";

/** The one admission point: the project shell (state socket, panel events, tab sync, session end,
 *  rail, status bar and tabs) mounts only while the person's name is committed; until then the
 *  work area holds the project picker alone. */
function Work() {
  const annotatorNamed = useStore(selectAnnotatorNamed);
  return (
    <div className="flex h-full flex-col">
      {annotatorNamed ? <ProjectShell /> : <ProjectPicker />}
    </div>
  );
}

/** The window: the top bar, the docking layout whose work area is the shell or the picker, and
 *  the notifications host (the top bar's button, the area's pane and the live region), which
 *  every screen shares. */
function App() {
  const [model] = useState(defaultDockModel);
  const [open, setOpen] = useState(false);
  const button = useRef<HTMLButtonElement>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  // A job told again keeps the count of listed entries and takes the newest id.
  const newest = useStore((s) => s.notifications.at(-1)?.id);
  const markSeen = useStore((s) => s.markNotificationsSeen);

  useEffect(() => {
    const follow = () => {
      const shown = notificationsOpen(model);
      setOpen(shown);
      // Closing the area from its own strip removes the control that held focus.
      if (!shown) {
        requestAnimationFrame(() => {
          if (document.activeElement === document.body) button.current?.focus();
        });
      }
    };
    model.addChangeListener(follow);
    return () => model.removeChangeListener(follow);
  }, [model]);

  useEffect(() => {
    if (open) markSeen();
  }, [open, newest, markSeen]);

  // Browser ctrl+wheel zoom stays off inside the app; an iframe-embedded tool (TensorBoard)
  // receives wheel events inside its own document and keeps it.
  useEffect(() => {
    if (!rootRef.current) return;
    return attachCtrlWheelGuard(rootRef.current);
  }, []);

  const toggle = useCallback(() => {
    toggleNotifications(model);
    if (notificationsOpen(model)) {
      requestAnimationFrame(() => document.getElementById(NOTIFICATIONS_HEADING_ID)?.focus());
    }
  }, [model]);

  return (
    <div ref={rootRef} className="h-full flex flex-col bg-tcip-bg text-tcip-fg">
      <TopBar notifications={<NotificationsButton ref={button} open={open} onToggle={toggle} />} />
      <Dock model={model} work={<Work />} />
      <NotificationsLiveRegion />
    </div>
  );
}

export default App;
