/**
 * @fileoverview TypeScript view interfaces for peak heap memory breakdown analysis.
 */

import {MemoryAnalysisBuffer} from './memory_analysis';

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
