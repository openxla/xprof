/** The OpProfileNode interface. */
export declare interface OpProfileNode {
  metrics?: Metrics;
}

/** The Metrics interface. */
declare interface Metrics {
  time?: number;
  flops?: number;
  bandwidthUtils?: number[];
  rawTime?: number;
  rawFlops?: number;
  rawBytesAccessedArray?: number[];
  memFractions?: number[];
}
