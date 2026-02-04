// DO NOT modify and only change in sync with the proto changes.

import {SourceInfo} from 'org_xprof/frontend/app/common/interfaces/source_info.jsonpb_decls';

/** */
export type MemBwType =
    'MEM_BW_TYPE_FIRST'|'MEM_BW_TYPE_ALL'|'MEM_BW_TYPE_HBM_RW'|
    'MEM_BW_TYPE_SRAM_RD'|'MEM_BW_TYPE_SRAM_WR'|'MEM_BW_TYPE_MAX';

/** */
export enum MemBwType$Names {
  MEM_BW_TYPE_FIRST = 'MEM_BW_TYPE_FIRST',
  MEM_BW_TYPE_ALL = 'MEM_BW_TYPE_ALL',
  MEM_BW_TYPE_HBM_RW = 'MEM_BW_TYPE_HBM_RW',
  MEM_BW_TYPE_SRAM_RD = 'MEM_BW_TYPE_SRAM_RD',
  MEM_BW_TYPE_SRAM_WR = 'MEM_BW_TYPE_SRAM_WR',
  MEM_BW_TYPE_MAX = 'MEM_BW_TYPE_MAX',
}

/** */
export enum NumericMemBwType {
  MEM_BW_TYPE_FIRST = 0,
  MEM_BW_TYPE_HBM_RW = 0,
  MEM_BW_TYPE_SRAM_RD = 1,
  MEM_BW_TYPE_SRAM_WR = 2,
  MEM_BW_TYPE_MAX = 2,
}

/** */
export type MemorySpace = 'MEMORY_SPACE_UNDEFINED'|'MEMORY_SPACE_HBM'|
    'MEMORY_SPACE_ON_CHIP'|'MEMORY_SPACE_ALL';

/** */
export enum MemorySpace$Names {
  MEMORY_SPACE_UNDEFINED = 'MEMORY_SPACE_UNDEFINED',
  MEMORY_SPACE_HBM = 'MEMORY_SPACE_HBM',
  MEMORY_SPACE_ON_CHIP = 'MEMORY_SPACE_ON_CHIP',
  MEMORY_SPACE_ALL = 'MEMORY_SPACE_ALL',
}

/** */
export enum NumericMemorySpace {
  MEMORY_SPACE_UNDEFINED = 0,
  MEMORY_SPACE_HBM = 1,
  MEMORY_SPACE_ON_CHIP = 2147483646,
  MEMORY_SPACE_ALL = 2147483647,
}

/** */
export type LayoutDimensionSemantics =
    'UNKNOWN_SEMANTICS'|'FEATURE'|'BATCH'|'SPATIAL';

/** */
export enum LayoutDimensionSemantics$Names {
  UNKNOWN_SEMANTICS = 'UNKNOWN_SEMANTICS',
  FEATURE = 'FEATURE',
  BATCH = 'BATCH',
  SPATIAL = 'SPATIAL',
}

/** */
export enum NumericLayoutDimensionSemantics {
  UNKNOWN_SEMANTICS = 0,
  FEATURE = 1,
  BATCH = 2,
  SPATIAL = 3,
}

/** */
export interface LayoutAnalysis {
  dimensions?: LayoutAnalysis.Dimension[];
}

/** */
export namespace LayoutAnalysis {
  export interface Dimension {
    size?: /* int32 */ number;
    alignment?: /* int32 */ number;
    semantics?: LayoutDimensionSemantics;
  }
}

/** */
export interface OpMetrics {
  hloModuleId?: /* uint64 */ string;
  name?: string;
  longName?: string;
  category?: string;
  provenance?: string;
  isEager?: boolean;
  occurrences?: /* uint32 */ number;
  timePs?: /* uint64 */ string;
  minTimePs?: /* uint64 */ string;
  selfTimePs?: /* uint64 */ string;
  flops?: /* uint64 */ string;
  bytesAccessed?: /* uint64 */ string;
  memoryAccessedBreakdown?: OpMetrics.MemoryAccessed[];
  dmaStallPs?: /* uint64 */ string;
  layout?: LayoutAnalysis;
  deduplicatedName?: string;
  children?: OpMetricsDb;
  numCores?: /* uint32 */ number;
  computationPrimitiveSize?: /* uint32 */ number;
  autotuned?: boolean;
  sourceInfo?: SourceInfo;
}

/** */
export namespace OpMetrics {
  export interface MemoryAccessed {
    operationType?: OpMetrics.MemoryAccessed.OperationType;
    memorySpace?: /* uint64 */ string;
    bytesAccessed?: /* uint64 */ string;
  }
  export namespace MemoryAccessed {
    export type OperationType = 'UNKNOWN'|'READ'|'WRITE';
    export enum OperationType$Names {
      UNKNOWN = 'UNKNOWN',
      READ = 'READ',
      WRITE = 'WRITE',
    }
    export enum NumericOperationType {
      UNKNOWN = 0,
      READ = 1,
      WRITE = 2,
    }
  }
}

/** */
export interface PrecisionStats {
  compute16bitPs?: /* uint64 */ string;
  compute32bitPs?: /* uint64 */ string;
}

/** */
export interface OpMetricsDb {
  metricsDb?: OpMetrics[];
  totalHostInfeedEnqDurationPs?: /* uint64 */ string;
  totalHostInfeedEnqStartTimestampPsDiff?: /* uint64 */ string;
  totalTimePs?: /* uint64 */ string;
  totalOpTimePs?: /* uint64 */ string;
  precisionStats?: PrecisionStats;
}
