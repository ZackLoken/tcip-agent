/** A legend row whose colour swatch is a button: click it to retune that symbology colour. */
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
        title="Click to change this colour"
        aria-label={`Change ${label} colour`}
        className="inline-block w-6 shrink-0 rounded-sm hover:opacity-70"
        style={{ borderTop: `2.5px ${dashed ? "dashed" : "solid"} ${color}` }}
      />
      <span className="text-tcip-fg">{label}</span>
    </li>
  );
}
