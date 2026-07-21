/**
 * @fileoverview Utilities for diffing and aligning data tables.
 */

/**
 * Generic row comparison wrapper.
 */
export interface ComparisonRow<T> {
  active: T | null;
  baseline: T | null;
  deltaPercent: number;
  isAdded: boolean;
  isRemoved: boolean;
}

/**
 * Generic aligner utility that tools can extend.
 */
export function alignTables<T>(
  active: T[],
  baseline: T[],
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
        deltaPercent: 0,
        isAdded: true,
        isRemoved: false,
      });
    }
  }

  if (baseline) {
    for (const item of baseline) {
      const key = keySelector(item);
      const existing = result.get(key);
      if (existing) {
        existing.baseline = item;
        existing.isAdded = false;
        if (deltaCalculator && existing.active) {
          existing.deltaPercent = deltaCalculator(existing.active, item);
        }
      } else {
        result.set(key, {
          active: null,
          baseline: item,
          deltaPercent: 0,
          isAdded: false,
          isRemoved: true,
        });
      }
    }
  }

  return result;
}
