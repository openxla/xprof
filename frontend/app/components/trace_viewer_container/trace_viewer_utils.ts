
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
 * Parses a source string or frame and returns the first valid
 * `filename:linenumber` found, or undefined if no valid frame exists.
 */
export function parseSourceFileAndLine(
  sourceString: string | undefined,
): string | undefined {
  if (!sourceString) return undefined;
  const lines = sourceString.trim().split('\n');
  if (lines.length === 0) return undefined;
  const framePattern =
    /^\s*([^:]+?)(?:\:(-?\d+))?(?:\:(\d+))?(?:[:-](\d+))?(?:\:-?\d+)*\s*$/;
  for (const line of lines) {
    const match = line.match(framePattern);
    if (match) {
      const fileName = match[1].trim();
      let lineNumber = Number(match[2]);
      if (!isNaN(lineNumber) && lineNumber > 0) {
        lineNumber = Math.floor(lineNumber);
        return `${fileName}:${lineNumber}`;
      }
    }
    const subMatch = line.match(/(?:^|[\s(\[])([^\s()\[\]:]+:\d+(?::\d+)?)/);
    if (subMatch) {
      const innerMatch = subMatch[1].match(framePattern);
      if (innerMatch) {
        const fileName = innerMatch[1].trim();
        let lineNumber = Number(innerMatch[2]);
        if (!isNaN(lineNumber) && lineNumber > 0) {
          lineNumber = Math.floor(lineNumber);
          return `${fileName}:${lineNumber}`;
        }
      }
    }
  }
  return undefined;
}
