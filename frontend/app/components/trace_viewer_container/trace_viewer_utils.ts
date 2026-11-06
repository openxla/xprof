
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

/** Result of splitting a microsecond timestamp or duration into an adaptable unit and value. */
export interface SplitTime {
  value: number;
  unit: 's' | 'ms' | 'us' | 'ns';
}

/**
 * Splits a timestamp or duration in microseconds into an adaptable unit (s, ms, us, ns)
 * and numeric value rounded to at most `maxFractionDigits` (default 2).
 *
 * Rolls up boundary values so "1000 ms", "1000 us", or "1000 ns" roll up to 1 s, 1 ms, or 1 us.
 */
export function splitTimeUs(valueUs: number, maxFractionDigits = 2): SplitTime {
  if (!Number.isFinite(valueUs)) {
    return {value: valueUs, unit: 'us'};
  }
  const absUs = Math.abs(valueUs);

  let unit: 's' | 'ms' | 'us' | 'ns' = 'ns';
  let scaled = absUs * 1e3;

  if (absUs >= 1e6) {
    unit = 's';
    scaled = absUs / 1e6;
  } else if (absUs >= 1e3) {
    unit = 'ms';
    scaled = absUs / 1e3;
  } else if (absUs >= 1) {
    unit = 'us';
    scaled = absUs;
  }

  let formatted = scaled.toFixed(maxFractionDigits);
  if (formatted.includes('.')) {
    formatted = formatted.replace(/\.?0+$/, '');
  }
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

  const roundedValue = Number(formatted) * (valueUs < 0 ? -1 : 1);
  return {value: roundedValue, unit};
}

/**
 * Formats a timestamp or duration in microseconds with adaptable units (s, ms, us, ns).
 *
 * Non-finite values (NaN, Infinity, -Infinity) return '-'.
 */
export function formatTimeUs(valueUs: number, maxFractionDigits = 2): string {
  if (!Number.isFinite(valueUs)) {
    return '-';
  }
  const {value, unit} = splitTimeUs(valueUs, maxFractionDigits);
  return `${value} ${unit}`;
}

