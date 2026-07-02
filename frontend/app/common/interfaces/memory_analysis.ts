/**
 * @fileoverview TypeScript interfaces for peak heap memory breakdown analysis.
 */

export type MemorySpaceBreakdown = Record<string, number>;

/**
 * Detailed buffer structure from peak heap trace.
 * Uses index signature to comply with Bracket Notation JSON minification rules.
 */
export declare interface MemoryAnalysisBuffer {
  readonly [propertyName: string]: string | number | undefined;
  readonly name: string;
  readonly sizeMib: number;
  readonly category: string;
  readonly subCategory: string;
  readonly group: string;
  readonly dtype: string;
  readonly tfOpName: string;
  readonly shape: string;
  readonly jaxVariablePath: string;
  readonly paddingMib: number;
  readonly paddingPercentage: number;
  readonly memorySpace?: string;
}

/** Summary of memory consumption per category. */
export type CategorySummaries = Record<string, number>;

/**
 * Preprocessed peak memory analysis result payload.
 * Uses index signature to comply with Bracket Notation JSON minification rules.
 */
export declare interface MemoryAnalysisResult {
  readonly [propertyName: string]:
    | string
    | CategorySummaries
    | MemorySpaceBreakdown
    | MemoryAnalysisBuffer[]
    | undefined;
  readonly moduleName: string;
  readonly categorySummaries: CategorySummaries;
  readonly memorySpaceBreakdown?: MemorySpaceBreakdown;
  readonly buffers: MemoryAnalysisBuffer[];
}

/** Raw buffer structure from backend C++ JSON payload. */
export declare interface RawMemoryAnalysisBuffer {
  readonly [propertyName: string]: string | number | undefined;
  readonly label?: string;
  readonly size?: number;
  readonly unpaddedSize?: number;
  readonly category?: string;
  readonly subCategory?: string;
  readonly tfOp?: string;
  readonly shape?: string;
  readonly jaxPath?: string;
  readonly group?: string;
  readonly memorySpace?: string;
}

/** Raw summaries map from backend C++ JSON payload. */
export type RawCategorySummaries = Record<string, number>;

/** Raw C++ JSON payload structure returned from backend memory analysis tool. */
export declare interface RawMemoryAnalysisResult {
  readonly [propertyName: string]:
    | string
    | RawCategorySummaries
    | MemorySpaceBreakdown
    | RawMemoryAnalysisBuffer[]
    | undefined;
  readonly moduleName: string;
  readonly summary: RawCategorySummaries;
  readonly memorySpaceBreakdown?: MemorySpaceBreakdown;
  readonly peakHeapBuffers: RawMemoryAnalysisBuffer[];
}
