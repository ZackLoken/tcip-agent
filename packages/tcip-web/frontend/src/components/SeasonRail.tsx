/**
 * Season rail: the app's signature. Phenology is seasonal time: a breeding project's
 * captures are a series of dates across a growing season. This plots those capture dates
 * as ticks along a dormant → bud → canopy → late-summer → fruit gradient, so a project's
 * temporal shape is legible at a glance (and unmistakably a phenology instrument, not a
 * generic dashboard). Display-only and behavior-neutral.
 */

import type { Capture } from "@/api/types.generated";

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

/** A dated capture: its name and the calendar day its date names, read in UTC. */
interface Tick {
  name: string;
  day: Date;
}

function seasonColor(month: number): string {
  if (month === 12 || month <= 2) return SEASON[0];
  if (month <= 4) return SEASON[1];
  if (month <= 7) return SEASON[2];
  if (month <= 9) return SEASON[3];
  return SEASON[4];
}

function label({ day }: Tick): string {
  return `${MONTH_ABBR[day.getUTCMonth()]} ${day.getUTCDate()}`;
}

interface SeasonRailProps {
  captures: Capture[];
  active?: string | null;
  className?: string;
  /** Print each tick's date below the rail (staggered two rows to limit overlap). */
  showLabels?: boolean;
}

export function SeasonRail({ captures, active, className, showLabels }: SeasonRailProps) {
  // A capture no date names (the undated bucket among them) is counted as undated, never a tick;
  // the dated ones keep the order the tree lists them in.
  const iso: Tick[] = captures.flatMap((c) =>
    c.date === null ? [] : [{ name: c.name, day: new Date(c.date) }],
  );
  const undated = captures.length - iso.length;
  if (iso.length === 0 && undated === 0) return null;

  // Position by absolute timestamp (not day-of-year) so a season that crosses a year
  // boundary orders correctly.
  const times = iso.map((t) => t.day.getTime());
  const min = times.length ? Math.min(...times) : 0;
  const max = times.length ? Math.max(...times) : 0;
  const span = max - min || 1;
  const posOf = (t: Tick) => (iso.length <= 1 ? 50 : ((t.day.getTime() - min) / span) * 100);

  const summary =
    iso.length > 0
      ? `${iso.length} capture ${iso.length === 1 ? "date" : "dates"} from ${label(iso[0])} to ${label(iso[iso.length - 1])}` +
        (undated ? `, plus ${undated} undated` : "")
      : `${undated} undated capture${undated === 1 ? "" : "s"}`;

  // When labeled, pin the rail near the top of a taller strip and drop the date captions
  // below it; otherwise keep the compact vertically-centered rail.
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
          {iso.map((t) => {
            const isActive = t.name === active;
            return (
              <span
                key={t.name}
                data-testid="season-tick"
                title={label(t)}
                className={`absolute -translate-x-1/2 -translate-y-1/2 rounded-full ${
                  isActive ? "h-3 w-3 ring-2 ring-tcip-fg/70" : "h-2 w-2"
                }`}
                style={{
                  left: `${posOf(t)}%`,
                  top: railTop,
                  background: seasonColor(t.day.getUTCMonth() + 1),
                }}
              />
            );
          })}
          {showLabels &&
            iso.map((t, i) => (
              <span
                key={`lbl-${t.name}`}
                className={`absolute -translate-x-1/2 text-[9px] leading-none tabular-nums whitespace-nowrap ${
                  t.name === active ? "font-medium text-tcip-fg" : "text-tcip-muted"
                }`}
                style={{ left: `${posOf(t)}%`, top: i % 2 === 0 ? "15px" : "26px" }}
              >
                {label(t)}
              </span>
            ))}
        </div>
      )}
      {undated > 0 && (
        <span className="text-[12px] text-tcip-muted whitespace-nowrap">
          <span className="inline-block h-2 w-2 rounded-full bg-tcip-muted/60 align-middle mr-1" />
          {undated} undated
        </span>
      )}
    </div>
  );
}
