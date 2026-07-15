/**
 * Season rail — the app's signature. Phenology is seasonal time: a breeding project's
 * captures are a series of dates across a growing season. This plots those capture dates
 * as ticks along a dormant → bud → canopy → late-summer → fruit gradient, so a project's
 * temporal shape is legible at a glance (and unmistakably a phenology instrument, not a
 * generic dashboard). Display-only and behavior-neutral.
 */

const ISO = /^(\d{4})-(\d{2})-(\d{2})$/;

// Dormant → bud → canopy → late-summer → fruit (mirrors the tcip-season-* tokens).
const SEASON = ["#5B6B6A", "#7FA96A", "#507754", "#C9A24B", "#E6976B"];
const MONTH_ABBR = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

function parts(iso: string): [number, number, number] {
  const m = ISO.exec(iso)!;
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

/** Absolute timestamp, or null if the ISO-shaped string isn't a real calendar date
 *  (e.g. "2026-13-40" rolls over — reject it rather than mislabel it). */
function timeOf(iso: string): number | null {
  const [y, mo, d] = parts(iso);
  const date = new Date(y, mo - 1, d);
  if (date.getFullYear() !== y || date.getMonth() !== mo - 1 || date.getDate() !== d) {
    return null;
  }
  return date.getTime();
}

function seasonColor(month: number): string {
  if (month === 12 || month <= 2) return SEASON[0];
  if (month <= 4) return SEASON[1];
  if (month <= 7) return SEASON[2];
  if (month <= 9) return SEASON[3];
  return SEASON[4];
}

function label(iso: string): string {
  const [, mo, d] = parts(iso);
  return `${MONTH_ABBR[mo - 1]} ${d}`;
}

interface SeasonRailProps {
  dates: string[];
  active?: string | null;
  className?: string;
  /** Print each tick's date below the rail (staggered two rows to limit overlap). */
  showLabels?: boolean;
}

export function SeasonRail({ dates, active, className, showLabels }: SeasonRailProps) {
  // Keep only real calendar dates; everything else (undated bucket, invalid folder
  // names) is counted as undated so nothing renders a bogus tick.
  const iso = dates.filter((d) => ISO.test(d) && timeOf(d) !== null).sort();
  const undated = dates.length - iso.length;
  if (iso.length === 0 && undated === 0) return null;

  // Position by absolute timestamp (not day-of-year) so a season that crosses a year
  // boundary — the primary winter-catkin case — orders correctly.
  const times = iso.map((d) => timeOf(d) as number);
  const min = times.length ? Math.min(...times) : 0;
  const max = times.length ? Math.max(...times) : 0;
  const span = max - min || 1;
  const posOf = (d: string) =>
    iso.length <= 1 ? 50 : (((timeOf(d) as number) - min) / span) * 100;

  const summary =
    iso.length > 0
      ? `${iso.length} capture ${iso.length === 1 ? "date" : "dates"} from ${label(iso[0])} to ${label(iso[iso.length - 1])}` +
        (undated ? `, plus ${undated} undated` : "")
      : `${undated} undated capture${undated === 1 ? "" : "s"}`;

  // When labelled, pin the rail near the top of a taller strip and drop the date captions
  // below it; otherwise keep the compact vertically-centred rail.
  const railTop = showLabels ? "7px" : "50%";
  return (
    <div
      className={`flex ${showLabels ? "items-start" : "items-center"} gap-2 ${className ?? ""}`}
      role="img"
      aria-label={summary}
    >
      {iso.length > 0 && (
        <div
          className={`relative flex-1 ${showLabels ? "h-9" : "h-4"}`}
          data-testid="season-rail-axis"
        >
          {/* Seasonal baseline: dormant → fruit */}
          <div
            className="absolute left-0 right-0 -translate-y-1/2 h-[3px] rounded-full opacity-70"
            style={{ top: railTop, background: `linear-gradient(90deg, ${SEASON.join(", ")})` }}
          />
          {iso.map((d) => {
            const isActive = d === active;
            const [, mo] = parts(d);
            return (
              <span
                key={d}
                data-testid="season-tick"
                title={label(d)}
                className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full ${
                  isActive ? "h-3 w-3 ring-2 ring-tcip-fg/70" : "h-2 w-2"
                }`}
                style={{ left: `${posOf(d)}%`, top: railTop, background: seasonColor(mo) }}
              />
            );
          })}
          {showLabels &&
            iso.map((d, i) => (
              <span
                key={`lbl-${d}`}
                className={`absolute -translate-x-1/2 text-[9px] leading-none tabular-nums whitespace-nowrap ${
                  d === active ? "font-medium text-tcip-fg" : "text-tcip-muted"
                }`}
                style={{ left: `${posOf(d)}%`, top: i % 2 === 0 ? "15px" : "26px" }}
              >
                {label(d)}
              </span>
            ))}
        </div>
      )}
      {undated > 0 && (
        <span className="text-[10px] text-tcip-muted whitespace-nowrap">
          <span className="inline-block h-2 w-2 rounded-full bg-tcip-muted/60 align-middle mr-1" />
          {undated} undated
        </span>
      )}
    </div>
  );
}
