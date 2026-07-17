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
 * Aligns items from active and baseline sessions into a unified map of
 * comparison rows indexed by a unique key.
 *
 * @template T The type of row items being compared and aligned.
 * @param active Array of items from the active session, or null/undefined.
 * @param baseline Array of items from the baseline session, or
 *     null/undefined.
 * @param keySelector A function that extracts a unique string key from an item
 *     to align rows across sessions.
 * @param deltaCalculator An optional function that calculates a numeric delta
 *     between aligned active and baseline items.
 * @return A map where the key is the string returned by `keySelector` and the
 *     value is a `ComparisonRow<T>` containing active/baseline items and status.
 */
export function alignTables<T>(
  active: T[] | null | undefined,
  baseline: T[] | null | undefined,
  keySelector: (item: T) => string,
  deltaCalculator?: (active: T, baseline: T) => number,
  baselineKeySelector?: (item: T) => string,
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
    const selector = baselineKeySelector || keySelector;
    for (const item of baseline) {
      const key = selector(item);
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
