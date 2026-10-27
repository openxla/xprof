
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

/**
 * Formats a timestamp or duration in microseconds with adaptable units (s, ms, us, ns).
 *
 * Preserves source digits by shifting the decimal point per unit (s/ms/us/ns -> up to 12/9/6/3
 * fraction digits) and trimming trailing zeros. Rolls up boundary values so "1000 ms",
 * "1000 us", or "1000 ns" never appear. Non-finite values (NaN, Infinity, -Infinity) return '-'.
 */
export function formatTimeUs(valueUs: number): string {
  if (!Number.isFinite(valueUs)) {
    return '-';
  }
  const sign = valueUs < 0 ? '-' : '';
  const absUs = Math.abs(valueUs);

  let unit = 'ns';
  let scaled = absUs * 1e3;
  let maxDecimals = 3;

  if (absUs >= 1e6) {
    unit = 's';
    scaled = absUs / 1e6;
    maxDecimals = 12;
  } else if (absUs >= 1e3) {
    unit = 'ms';
    scaled = absUs / 1e3;
    maxDecimals = 9;
  } else if (absUs >= 1) {
    unit = 'us';
    scaled = absUs;
    maxDecimals = 6;
  }

  let formatted = scaled.toFixed(maxDecimals).replace(/\.?0+$/, '');
  if (formatted === '1000') {
    if (unit === 'ns') {
      unit = 'us';
      formatted = '1';
    } else if (unit === 'us') {
      unit = 'ms';
      formatted = '1';
    } else if (unit === 'ms') {
      unit = 's';
      formatted = '1';
    }
  }

  return `${sign}${formatted} ${unit}`;
}

