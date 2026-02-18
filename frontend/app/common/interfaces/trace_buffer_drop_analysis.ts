/** Represents a single trace buffer drop. */
export declare interface TraceBufferDrop {
  drop_type: string;
  host: string;
}

/** Represents the JSON response structure for trace buffer drop analysis. */
export declare interface TraceBufferDropAnalysisReport {
  message: string;
  trace_buffer_drops: TraceBufferDrop[];
}
