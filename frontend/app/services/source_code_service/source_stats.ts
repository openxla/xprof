/**
 * @fileoverview Stats for a source code.
 */

/** Different metrics for individual lines. */
export declare interface Metric {
  occurrences: number;
  selfTimePs: number;
  timePs: number;
  flops: number;
}

/** Metric for a single line of a file. */
export declare interface LineMetric {
  lineNumber: number;
  metric: Metric;
}
