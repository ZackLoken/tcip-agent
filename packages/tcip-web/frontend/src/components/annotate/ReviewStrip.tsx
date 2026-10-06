import { Stepper } from "@/components/Stepper";
import { ANNOTATE_KEYS } from "@/lib/annotateKeys";
import {
  itemName,
  STATUS_SCOPES,
  type ItemFilters,
  type ItemShape,
  type ReviewItem,
  type StatusScope,
} from "@/lib/reviewItems";
import { STATUS_WORDS } from "@/lib/symbology";
import { MODES } from "@/lib/toolMode";

const SHAPES: (ItemShape | "all")[] = ["all", ...MODES];

/** The status strip under the toolbar: the proposals toggle, the confidence floor, the item type
 *  filter and the counts the filters leave; the accept, edit and reject controls; and the item
 *  stepper with the status scope it steps through. */
export function ReviewStrip({
  bucket,
  proposalsShown,
  reviewing,
  onProposalsShown,
  operatingPoint,
  filters,
  onFilters,
  scope,
  onScope,
  counts,
  focused,
  position,
  total,
  onStep,
  onJump,
  onAccept,
  onEdit,
  onReject,
}: {
  /** The prediction bucket under review; null when the dataset selection names none. */
  bucket: string | null;
  /** The toggle's own state: whether the person wants the bucket's proposals shown. */
  proposalsShown: boolean;
  /** Whether the bucket's proposals are on the canvas now: shown, and served for this image. */
  reviewing: boolean;
  onProposalsShown: (next: boolean) => void;
  /** The bucket's validated operating point, or the reason it has none; null until served. */
  operatingPoint: { conf: number | null; reason: string } | null;
  filters: ItemFilters;
  onFilters: (next: ItemFilters) => void;
  scope: StatusScope;
  onScope: (next: StatusScope) => void;
  counts: { items: number; undecided: number };
  focused: ReviewItem | null;
  position: number;
  total: number;
  onStep: (delta: number) => void;
  onJump: (oneBased: number) => void;
  onAccept: () => void;
  onEdit: () => void;
  onReject: () => void;
}) {
  const canDecide = reviewing && counts.undecided > 0;
  const K = ANNOTATE_KEYS;
  return (
    <div className="flex h-9 shrink-0 items-center gap-3 border-b border-tcip-border bg-tcip-panel px-3 text-[12px]">
      <label
        className="flex items-center gap-1.5"
        title={
          bucket
            ? `Show the bucket's proposals; a mark made while hidden records it (${K.hideProposals.label})`
            : "Choose a prediction bucket in Setup to review its proposals"
        }
      >
        <input
          type="checkbox"
          checked={!!bucket && proposalsShown}
          disabled={!bucket}
          onChange={(e) => onProposalsShown(e.target.checked)}
        />
        Proposals
      </label>

      <label
        className="flex items-center gap-1.5 text-tcip-muted"
        title={
          operatingPoint === null
            ? "The confidence floor starts at the bucket's operating point once its proposals load"
            : operatingPoint.conf === null
              ? `No validated operating point: ${operatingPoint.reason}`
              : `The bucket's validated operating point is ${operatingPoint.conf.toFixed(2)}`
        }
      >
        Confidence
        <input
          type="number"
          aria-label="Confidence floor"
          className="tcip-input h-7 w-16 text-[11px]"
          min={0}
          max={1}
          step={0.01}
          disabled={!reviewing}
          value={filters.confidence ?? ""}
          onChange={(e) =>
            onFilters({
              ...filters,
              confidence: e.target.value === "" ? null : Number(e.target.value),
            })
          }
        />
      </label>

      <label className="flex items-center gap-1.5 text-tcip-muted">
        Type
        <select
          aria-label="Item type"
          className="tcip-select h-7 text-[11px]"
          value={filters.shape}
          onChange={(e) => onFilters({ ...filters, shape: e.target.value as ItemShape | "all" })}
        >
          {SHAPES.map((shape) => (
            <option key={shape} value={shape}>
              {shape}
            </option>
          ))}
        </select>
      </label>

      <span className="font-mono text-[11px] text-tcip-muted" aria-live="polite">
        {counts.items} {counts.items === 1 ? "item" : "items"}
        {reviewing ? `, ${counts.undecided} ${STATUS_WORDS.undecided.toLowerCase()}` : ""}
      </span>

      <div className="flex-1" />

      <div className="flex items-center gap-1.5" role="group" aria-label="Decide">
        <button
          type="button"
          className="tcip-btn h-7 text-[11px]"
          disabled={!canDecide}
          onClick={onAccept}
          title={`${K.accept.desc} (${K.accept.label})`}
        >
          Accept
        </button>
        <button
          type="button"
          className="tcip-btn h-7 text-[11px]"
          disabled={!focused}
          onClick={onEdit}
          title={`${K.edit.desc} (${K.edit.label})`}
        >
          Edit
        </button>
        <button
          type="button"
          className="tcip-btn h-7 text-[11px]"
          disabled={!canDecide}
          onClick={onReject}
          title={`${K.reject.desc} (${K.reject.label})`}
        >
          Reject
        </button>
      </div>

      <Stepper
        label="Item"
        noun="item"
        name={itemName(focused)}
        position={position}
        total={total}
        canPrev={total > 0}
        canNext={total > 0}
        onStep={onStep}
        onJump={onJump}
      />
      <select
        aria-label="Step through"
        className="tcip-select h-7 text-[11px]"
        value={scope}
        disabled={!reviewing}
        onChange={(e) => onScope(e.target.value as StatusScope)}
      >
        {STATUS_SCOPES.map((s) => (
          <option key={s} value={s}>
            {s === "all" ? "every item" : STATUS_WORDS[s].toLowerCase()}
          </option>
        ))}
      </select>
    </div>
  );
}
