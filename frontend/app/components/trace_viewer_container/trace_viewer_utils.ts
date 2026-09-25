
/** Parses a numeric trace event argument, ignoring non-finite values. */
export function parseNumericArg(value: unknown): number | undefined {
  if (typeof value !== 'number' && typeof value !== 'string') return undefined;
  if (typeof value === 'string' && value.trim().length === 0) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

/** Formats a value with an SI prefix, e.g. `formatSi(1.2e12, 'FLOP/s')`. */
export function formatSi(value: number, unit: string, fractionDigits = 2) {
  const prefixes = ['', 'K', 'M', 'G', 'T', 'P', 'E'];
  let scaled = Math.abs(value);
  let index = 0;
  while (scaled >= 1000 && index < prefixes.length - 1) {
    scaled /= 1000;
    index++;
  }
  const sign = value < 0 ? '-' : '';
  return `${sign}${scaled.toFixed(fractionDigits)} ${prefixes[index]}${unit}`;
}
