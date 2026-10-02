import { useEffect, useState } from "react";

import { isFocusableControl } from "@/hooks/useKeyboardShortcuts";
import { ANNOTATE_KEYS } from "@/lib/annotateKeys";
import type { TabName } from "@/store/types";

interface Shortcut {
  key: string;
  desc: string;
}

const GLOBAL: Shortcut[] = [
  { key: "?", desc: "Toggle this help overlay" },
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
      <table className="w-full text-[11px]">
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
      if (isFocusableControl(e.target)) return;
      if (e.key === "?" || (e.key === "/" && e.shiftKey)) {
        e.preventDefault();
        setOpen((p) => !p);
      } else if (e.key === "Escape" && open) {
        setOpen(false);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  if (!open) return null;
  return (
    <div
      className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
      onClick={() => setOpen(false)}
    >
      <div
        className="tcip-panel p-5 w-[520px] max-w-[90vw] max-h-[80vh] overflow-auto flex flex-col gap-4"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between">
          <div className="text-[13px] font-semibold">Keyboard & mouse reference</div>
          <button className="tcip-btn text-[11px]" onClick={() => setOpen(false)}>
            Close
          </button>
        </div>
        {activeTab === "annotate" && <Section title="Annotate" items={ANNOTATE} />}
        <Section title="Mouse" items={MOUSE} />
        <Section title="Global" items={GLOBAL} />
      </div>
    </div>
  );
}
