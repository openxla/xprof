/** Represents a single trace buffer drop. */
export declare interface TraceBufferDrop {
  dropType: string;
  host: string;
}

/** Represents the JSON response structure for trace buffer drop analysis. */
export declare interface TraceBufferDropAnalysisReport {
  message: string;
  traceBufferDrops: TraceBufferDrop[];
}
