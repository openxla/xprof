/**
 * @fileoverview Utilities for diffing and aligning data tables.
 */

/** Represents the presence of a row across active and baseline sessions. */
export type DiffStatus = 'added' | 'removed' | 'common';

/**
 * Generic row comparison wrapper.
 */
export interface ComparisonRow<T> {
  active: T | null;
  baseline: T | null;
  delta: number | null;
  status: DiffStatus;
}

/**
 * Generic aligner utility that tools can extend.
 */
export function alignTables<T>(
  active: T[] | null | undefined,
  baseline: T[] | null | undefined,
  keySelector: (item: T) => string,
  deltaCalculator?: (active: T, baseline: T) => number,
): Map<string, ComparisonRow<T>> {
  const result = new Map<string, ComparisonRow<T>>();

  if (active) {
    for (const item of active) {
      const key = keySelector(item);
      result.set(key, {
        active: item,
        baseline: null,
        delta: null,
        status: 'added',
      });
    }
  }

  if (baseline) {
    for (const item of baseline) {
      const key = keySelector(item);
      const existing = result.get(key);

      if (existing) {
        existing.baseline = item;
        if (existing.active) {
          existing.status = 'common';
          existing.delta = deltaCalculator
            ? deltaCalculator(existing.active, item)
            : 0;
        } else {
          existing.status = 'removed';
          existing.delta = null;
        }
      } else {
        result.set(key, {
          active: null,
          baseline: item,
          delta: null,
          status: 'removed',
        });
      }
    }
  }

  return result;
}
