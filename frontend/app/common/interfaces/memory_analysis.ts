/**
 * @fileoverview TypeScript interfaces for peak heap memory breakdown analysis.
 */

export interface MemorySpaceBreakdown {
  [spaceName: string]: number;
}

/**
 * Detailed buffer structure from peak heap trace.
 * Uses index signature to comply with Bracket Notation JSON minification rules.
 */
export interface MemoryAnalysisBuffer {
  [key: string]: string | number | undefined;
  name: string;
  sizeMib: number;
  category: string;
  subCategory: string;
  group: string;
  dtype: string;
  tfOpName: string;
  shape: string;
  jaxVariablePath: string;
  paddingMib: number;
  paddingPercentage: number;
  memorySpace?: string;
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
    | MemorySpaceBreakdown
    | MemoryAnalysisBuffer[]
    | undefined;
  moduleName: string;
  categorySummaries: CategorySummaries;
  memorySpaceBreakdown?: MemorySpaceBreakdown;
  buffers: MemoryAnalysisBuffer[];
}

/** Raw buffer structure from backend C++ JSON payload. */
export interface RawMemoryAnalysisBuffer {
  [key: string]: string | number | undefined;
  label?: string;
  size?: number;
  unpaddedSize?: number;
  category?: string;
  subCategory?: string;
  tfOp?: string;
  shape?: string;
  jaxPath?: string;
  group?: string;
  memorySpace?: string;
}

/** Raw summaries map from backend C++ JSON payload. */
export interface RawCategorySummaries {
  [categoryName: string]: number;
}

/** Raw C++ JSON payload structure returned from backend memory analysis tool. */
export interface RawMemoryAnalysisResult {
  [key: string]:
    | string
    | RawCategorySummaries
    | MemorySpaceBreakdown
    | RawMemoryAnalysisBuffer[]
    | undefined;
  moduleName: string;
  summary: RawCategorySummaries;
  memorySpaceBreakdown?: MemorySpaceBreakdown;
  peakHeapBuffers: RawMemoryAnalysisBuffer[];
}

/**
 * Hierarchical node representing a group of buffers or a single buffer leaf.
 */
export interface TreeNode {
  name: string;
  value: number; // Size in MiB (total or padding, depending on active metric)
  totalValue: number; // Stored total size in MiB
  paddingValue: number; // Stored padding size in MiB
  children: TreeNode[];
  buffer?: MemoryAnalysisBuffer; // Reference to original buffer if leaf
  path: string; // Full hierarchical path (e.g., 'transformer/layer_0' or 'Temporary/Activation')
  isLeaf: boolean;
  depth: number;
}

/** Supported color modes for visualizer rendering. */
export type ColorMode = 'category' | 'classic' | 'padding';
