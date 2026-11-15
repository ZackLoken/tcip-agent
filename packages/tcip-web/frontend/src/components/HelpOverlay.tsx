import { useEffect, useState } from "react";

import { ModalDialog } from "@/components/ModalDialog";
import { isFocusableControl } from "@/hooks/useKeyboardShortcuts";
import { ANNOTATE_KEYS } from "@/lib/annotateKeys";
import { keydownInModal } from "@/lib/modalKeydown";
import type { TabName } from "@/store/types";

interface Shortcut {
  key: string;
  desc: string;
}

const GLOBAL: Shortcut[] = [
  { key: "?", desc: "Open this help overlay" },
  { key: "Esc", desc: "Close dialogs / cancel in-progress drawing" },
];

const ANNOTATE: Shortcut[] = Object.values(ANNOTATE_KEYS).map((k) => ({
  key: k.label,
  desc: k.desc,
}));

const MOUSE: Shortcut[] = [
  { key: "Scroll / two-finger", desc: "Pan (any direction)" },
  { key: "Ctrl + Scroll / pinch", desc: "Zoom at cursor" },
  { key: "Shift + Scroll", desc: "Pan horizontally" },
  { key: "Space + drag", desc: "Pan (hold space, drag with the left button)" },
  { key: "Middle-click + drag", desc: "Pan" },
  { key: "Click", desc: "Place a point (point mode); press and drag a placed point to move it" },
  { key: "Double-click", desc: "Close current polygon (polygon mode)" },
  {
    key: "Right-click",
    desc: "Cancel in-progress polygon, or delete vertex / shape under cursor (current mode only)",
  },
];

function Section({ title, items }: { title: string; items: Shortcut[] }) {
  return (
    <div>
      <div className="tcip-heading mb-1">{title}</div>
      <table className="w-full text-[13px]">
        <tbody>
          {items.map((s) => (
            <tr key={s.key}>
              <td className="font-mono text-tcip-warn pr-3 py-0.5 w-36">{s.key}</td>
              <td className="text-tcip-fg py-0.5">{s.desc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface HelpOverlayProps {
  activeTab: TabName;
}

export function HelpOverlay({ activeTab }: HelpOverlayProps) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (isFocusableControl(e.target) || keydownInModal(e)) return;
      if (e.key === "?" || (e.key === "/" && e.shiftKey)) {
        e.preventDefault();
        setOpen(true);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  if (!open) return null;
  return (
    <ModalDialog
      title="Shortcuts"
      onClose={() => setOpen(false)}
      closeOnOutsideClick
      className="w-[520px]"
      headerClose="Close"
    >
      <div className="flex flex-col gap-4">
        {activeTab === "annotate" && <Section title="Annotate" items={ANNOTATE} />}
        <Section title="Mouse" items={MOUSE} />
        <Section title="Global" items={GLOBAL} />
      </div>
    </ModalDialog>
  );
}
