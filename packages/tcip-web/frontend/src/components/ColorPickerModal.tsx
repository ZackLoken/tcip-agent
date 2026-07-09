/**
 * Dark color picker modeled after yolo-annotator's _show_dark_color_picker.
 * SI palette + basic palette + hex input, resolving to a hex string.
 */

import { useEffect, useRef, useState } from "react";

const SI_PALETTE: [string, string][] = [
  ["SI Green", "#507754"],
  ["Water Blue", "#83A0BA"],
  ["Mulberry", "#996967"],
  ["Lake Blue", "#367A8A"],
  ["Stem Green", "#7E8F60"],
  ["Persimmon", "#E6976B"],
  ["Elderberry", "#2A194E"],
  ["Wood", "#C7B299"],
  ["Sage", "#889E6E"],
  ["Leaf Green", "#6F9382"],
];

const BASIC_PALETTE: [string, string][] = [
  ["Red", "#FF0000"],
  ["Orange", "#FF8C00"],
  ["Yellow", "#FFD700"],
  ["Lime", "#32CD32"],
  ["Cyan", "#00CED1"],
  ["Blue", "#4169E1"],
  ["Purple", "#8A2BE2"],
  ["Pink", "#FF69B4"],
  ["White", "#FFFFFF"],
  ["Gray", "#808080"],
];

interface Props {
  title: string;
  initialColor: string;
  onSubmit: (color: string) => void;
  onCancel: () => void;
}

const HEX_RE = /^#[0-9a-fA-F]{6}$/;

export function ColorPickerModal({ title, initialColor, onSubmit, onCancel }: Props) {
  const [color, setColor] = useState(initialColor);
  const [hexDraft, setHexDraft] = useState(initialColor);
  const hexRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    hexRef.current?.focus();
    hexRef.current?.select();
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") {
        e.preventDefault();
        onCancel();
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  function pick(c: string) {
    setColor(c);
    setHexDraft(c);
  }

  function commitHex() {
    if (HEX_RE.test(hexDraft)) setColor(hexDraft.toUpperCase());
    else setHexDraft(color);
  }

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 flex items-center justify-center"
      onClick={onCancel}
    >
      <div className="tcip-panel rounded-lg p-5 w-[400px]" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-3">
          <div className="text-[13px] font-semibold">{title}</div>
          <button className="tcip-btn text-[11px]" onClick={onCancel}>
            Cancel
          </button>
        </div>

        <div className="flex items-center gap-2 mb-3">
          <span className="text-[11px] text-tcip-muted">Selected:</span>
          <div
            className="w-7 h-7 rounded border border-tcip-border"
            style={{ background: color }}
          />
          <input
            ref={hexRef}
            className="tcip-input w-24 font-mono"
            value={hexDraft}
            onChange={(e) => setHexDraft(e.target.value)}
            onBlur={commitHex}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                commitHex();
                e.preventDefault();
              }
            }}
          />
          <button
            className="tcip-btn text-[11px]"
            onClick={() => {
              const el = document.createElement("input");
              el.type = "color";
              el.value = HEX_RE.test(color) ? color : "#ffffff";
              el.addEventListener("input", () => pick(el.value.toUpperCase()), { once: false });
              el.addEventListener("change", () => pick(el.value.toUpperCase()), { once: true });
              el.click();
            }}
          >
            System…
          </button>
        </div>

        <Section title="SI Palette" palette={SI_PALETTE} onPick={pick} selected={color} />
        <Section title="Basic Colors" palette={BASIC_PALETTE} onPick={pick} selected={color} />

        <div className="flex gap-2 mt-4">
          <button className="tcip-btn-primary flex-1" onClick={() => onSubmit(color)}>
            OK
          </button>
          <button className="tcip-btn flex-1" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

function Section({
  title,
  palette,
  onPick,
  selected,
}: {
  title: string;
  palette: [string, string][];
  onPick: (c: string) => void;
  selected: string;
}) {
  return (
    <div className="mb-3">
      <div className="tcip-heading mb-1">{title}</div>
      <div className="grid grid-cols-5 gap-2">
        {palette.map(([name, hex]) => (
          <button
            key={hex}
            title={`${name}  ${hex}`}
            onClick={() => onPick(hex)}
            className={`h-8 rounded border-2 transition-transform hover:scale-105 ${
              selected.toUpperCase() === hex.toUpperCase() ? "border-white" : "border-tcip-border"
            }`}
            style={{ background: hex }}
          />
        ))}
      </div>
    </div>
  );
}
