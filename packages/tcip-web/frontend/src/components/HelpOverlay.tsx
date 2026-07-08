import { useEffect, useState } from "react";

import type { TabName } from "@/store/types";

interface Shortcut {
  key: string;
  desc: string;
}

const GLOBAL: Shortcut[] = [
  { key: "?", desc: "Toggle this help overlay" },
  { key: "Esc", desc: "Close dialogs / cancel in-progress drawing" },
];

const ANNOTATE: Shortcut[] = [
  { key: "m", desc: "Toggle Box / Polygon mode" },
  { key: "Ctrl+Z", desc: "Undo" },
  { key: "Ctrl+Shift+Z", desc: "Redo" },
  { key: "Ctrl+S", desc: "Save labels" },
  { key: "Enter", desc: "Close current polygon" },
  { key: "Delete", desc: "Delete selected polygon" },
  { key: "←  →", desc: "Prev / Next image" },
];

const REVIEW: Shortcut[] = [
  { key: "a", desc: "Accept (TP confirm / FP add-to-GT / FN keep-GT)" },
  { key: "e", desc: "Edit in Annotate tab (carries zoom + pred-reference)" },
  { key: "r", desc: "Reject (delete GT or dismiss pred)" },
  { key: "←  →", desc: "Prev / Next detection" },
  { key: "↑  ↓", desc: "Prev / Next image" },
];

const MOUSE: Shortcut[] = [
  { key: "Ctrl + Scroll", desc: "Zoom at cursor" },
  { key: "Middle-click + drag", desc: "Pan" },
  { key: "Scroll", desc: "Pan vertically" },
  { key: "Shift + Scroll", desc: "Pan horizontally" },
];

function Section({ title, items }: { title: string; items: Shortcut[] }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-tcip-muted mb-1">{title}</div>
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
          <div className="text-[14px] font-semibold">Keyboard & mouse reference</div>
          <button className="tcip-btn text-[11px]" onClick={() => setOpen(false)}>
            Close
          </button>
        </div>
        {activeTab === "annotate" && <Section title="Annotate" items={ANNOTATE} />}
        {activeTab === "review" && <Section title="Review" items={REVIEW} />}
        <Section title="Mouse" items={MOUSE} />
        <Section title="Global" items={GLOBAL} />
      </div>
    </div>
  );
}
