import {Pipe, PipeTransform} from '@angular/core';

interface TimeUnitConfig {
  readonly minMagnitude: number;
  readonly scale: number;
  readonly suffix: string;
}

const TIME_UNITS: readonly TimeUnitConfig[] = [
  {minMagnitude: 1e6, scale: 1e6, suffix: 's'},
  {minMagnitude: 1e3, scale: 1e3, suffix: 'ms'},
  {minMagnitude: 1, scale: 1, suffix: 'μs'},
  {minMagnitude: 1e-3, scale: 1e-3, suffix: 'ns'},
  {minMagnitude: 0, scale: 1e-6, suffix: 'ps'},
];

/**
 * Transforms a raw number representing timestamps or durations into a
 * human readable format (s, ms, μs, ns, ps).
 */
@Pipe({
  name: 'timeFormat',
  standalone: true,
})
export class TimeFormatPipe implements PipeTransform {
  transform(value: number | undefined | null): string {
    if (value == null || Number.isNaN(value)) {
      return '0.00s';
    }

    const absVal = Math.abs(value);
    const unit =
      TIME_UNITS.find((u) => absVal >= u.minMagnitude) ??
      TIME_UNITS[TIME_UNITS.length - 1];

    return `${(value / unit.scale).toFixed(2)}${unit.suffix}`;
  }
}
