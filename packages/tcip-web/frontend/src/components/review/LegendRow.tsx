/** A legend row whose color swatch is a button: click it to retune that symbology color. */
export function LegendRow({
  color,
  dashed,
  label,
  onEdit,
}: {
  color: string;
  dashed?: boolean;
  label: string;
  onEdit: () => void;
}) {
  return (
    <li className="flex items-center gap-2.5">
      <button
        type="button"
        onClick={onEdit}
        title="Click to change this color"
        aria-label={`Change ${label} color`}
        className="inline-block w-6 shrink-0 rounded-sm hover:opacity-70"
        style={{ borderTop: `2.5px ${dashed ? "dashed" : "solid"} ${color}` }}
      />
      <span className="text-tcip-fg">{label}</span>
    </li>
  );
}
