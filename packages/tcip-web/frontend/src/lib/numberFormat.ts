/** Significant figures a recorded number shows: the owner's ask that long values truncate alike. */
const SIGNIFICANT = 4;

/** A recorded number as the Training tab shows it: a whole number as it is, a value from 0.001
 *  up to 10,000 at four significant figures with no trailing zeros, any other in scientific
 *  notation at four significant figures. */
export function formatNumber(value: number): string {
  if (Number.isInteger(value) && Math.abs(value) < 1e4) return String(value);
  const magnitude = Math.abs(value);
  if (magnitude >= 1e-3 && magnitude < 1e4) return String(Number(value.toPrecision(SIGNIFICANT)));
  return value.toExponential(SIGNIFICANT - 1);
}
