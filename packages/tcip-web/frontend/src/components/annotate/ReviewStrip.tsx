import { FlagPanel } from "@/components/annotate/FlagPanel";
import { Stepper } from "@/components/Stepper";
import { ANNOTATE_KEYS } from "@/lib/annotateKeys";
import {
  itemName,
  STEP_SCOPE_WORDS,
  STEP_SCOPES,
  type ItemFilters,
  type ReviewItem,
  type StepScope,
} from "@/lib/reviewItems";
import { MATCH_HINTS, MATCH_TYPES, MATCH_WORDS, type MatchType } from "@/lib/symbology";
import type { Flag } from "@/store/types";

/** The status strip under the toolbar: the proposals toggle, the confidence floor and the match
 *  filter; the accept, edit, reject and flag controls; and the item stepper with the scope it
 *  steps through, each scope naming how many items it holds. */
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
  canAccept,
  canReject,
  scopeCounts,
  flagCount,
  focused,
  position,
  total,
  onStep,
  onJump,
  onAccept,
  onEdit,
  onReject,
  flags,
  flagsOpen,
  onFlagsOpen,
  onFlag,
  onResolve,
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
  scope: StepScope;
  onScope: (next: StepScope) => void;
  /** Whether an accept, or a reject, has an item to act on. */
  canAccept: boolean;
  canReject: boolean;
  /** How many items each scope steps through, of those the filters leave. */
  scopeCounts: Record<StepScope, number>;
  /** The image's open flags. */
  flagCount: number;
  focused: ReviewItem | null;
  position: number;
  total: number;
  onStep: (delta: number) => void;
  onJump: (oneBased: number) => void;
  onAccept: () => void;
  onEdit: () => void;
  onReject: () => void;
  /** The open flags on the focused item, or on the image as a whole when nothing is focused. */
  flags: Flag[];
  flagsOpen: boolean;
  onFlagsOpen: (open: boolean) => void;
  onFlag: (text: string) => Promise<boolean>;
  onResolve: (id: string, reply: string) => void;
}) {
  const K = ANNOTATE_KEYS;
  return (
    <div className="flex h-9 shrink-0 items-center gap-3 border-b border-tcip-border bg-tcip-panel px-3 text-[13px]">
      <label
        className="flex items-center gap-1.5"
        title={
          bucket
            ? `Show proposals (${K.hideProposals.label})`
            : "Choose predictions on the project card"
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

      {reviewing && (
        <>
          <label
            className="flex items-center gap-1.5 text-tcip-muted"
            title={
              operatingPoint === null || operatingPoint.conf === null
                ? "No validated operating point"
                : `Validated operating point ${operatingPoint.conf.toFixed(2)}`
            }
          >
            Confidence
            <input
              type="number"
              aria-label="Confidence floor"
              className="tcip-input h-7 w-16"
              min={0}
              max={1}
              step={0.01}
              value={filters.confidence ?? ""}
              onChange={(e) =>
                onFilters({
                  ...filters,
                  confidence: e.target.value === "" ? null : Number(e.target.value),
                })
              }
            />
          </label>

          <label
            className="flex items-center gap-1.5 text-tcip-muted"
            title={filters.match === "all" ? undefined : MATCH_HINTS[filters.match]}
          >
            Show
            <select
              aria-label="Show"
              className="tcip-select h-7"
              value={filters.match}
              onChange={(e) =>
                onFilters({ ...filters, match: e.target.value as MatchType | "all" })
              }
            >
              <option value="all">all</option>
              {MATCH_TYPES.map((match) => (
                <option key={match} value={match} title={MATCH_HINTS[match]}>
                  {MATCH_WORDS[match]}
                </option>
              ))}
            </select>
          </label>
        </>
      )}

      <div className="flex-1" />

      <div className="relative flex items-center gap-1.5" role="group" aria-label="Decide">
        <button
          type="button"
          className="tcip-btn h-7 text-[13px]"
          disabled={!canAccept}
          onClick={onAccept}
          title={`${K.accept.desc} (${K.accept.label})`}
        >
          Accept
        </button>
        <button
          type="button"
          className="tcip-btn h-7 text-[13px]"
          disabled={focused?.kind !== "proposal"}
          onClick={onEdit}
          title={`${K.edit.desc} (${K.edit.label})`}
        >
          Edit
        </button>
        <button
          type="button"
          className="tcip-btn h-7 text-[13px]"
          disabled={!canReject}
          onClick={onReject}
          title={`${K.reject.desc} (${K.reject.label})`}
        >
          Reject
        </button>
        <button
          type="button"
          className="tcip-btn h-7 text-[13px]"
          aria-expanded={flagsOpen}
          onClick={() => onFlagsOpen(!flagsOpen)}
          title={`${K.flag.desc} (${K.flag.label})`}
        >
          Flag{flagCount > 0 ? ` (${flagCount})` : ""}
        </button>
        {flagsOpen && (
          <FlagPanel
            target={itemName(focused) ?? "this image"}
            flags={flags}
            onRaise={onFlag}
            onResolve={onResolve}
            onClose={() => onFlagsOpen(false)}
          />
        )}
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
      <label className="flex items-center gap-1.5 text-tcip-muted">
        Step
        <select
          aria-label="Step through"
          className="tcip-select h-7 text-[13px]"
          value={scope}
          onChange={(e) => onScope(e.target.value as StepScope)}
        >
          {STEP_SCOPES.map((s) => (
            <option key={s} value={s}>
              {`${STEP_SCOPE_WORDS[s]} (${scopeCounts[s]})`}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
