/**
 * @fileoverview TypeScript interfaces for peak heap memory breakdown analysis.
 */

/**
 * Detailed buffer structure from peak heap trace.
 * Uses index signature to comply with Bracket Notation JSON minification rules.
 */
export interface MemoryAnalysisBuffer {
  [key: string]: string | number | undefined;
  name: string;
  sizeMib: number;
  category: string;
  dtype: string;
  tfOpName: string;
  shape: string;
  jaxVariablePath: string;
}

/** Summary of memory consumption per category. */
export interface CategorySummaries {
  [categoryName: string]: number;
}

/**
 * Preprocessed peak memory analysis result payload.
 * Uses index signature to comply with Bracket Notation JSON minification rules.
 */
export interface MemoryAnalysisResult {
  [key: string]:
    | string
    | CategorySummaries
    | MemoryAnalysisBuffer[]
    | undefined;
  moduleName: string;
  categorySummaries: CategorySummaries;
  buffers: MemoryAnalysisBuffer[];
}
