/**
 * @fileoverview Stats for a source code.
 */

/** Statistics pertaining to an individual line. */
export declare interface Metric {
  selfTimePs: number | undefined;
  flopsUtilization: number | undefined;
}

/** Metric for a single line of a file. */
export declare interface LineMetric {
  lineNumber: number;
  metric: Metric;
}
