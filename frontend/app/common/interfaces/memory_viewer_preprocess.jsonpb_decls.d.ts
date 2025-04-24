// This file must be kept in sync with the corresponding proto file.

/** See corresponding proto */
export interface HeapObject {
  numbered?: /* int32 */ number;
  named?: string;
  label?: string;
  logicalBufferId?: /* int32 */ number;
  logicalBufferSizeMib?: /* double */ number;
  unpaddedShapeMib?: /* double */ number;
  instructionName?: string;
  shapeString?: string;
  tfOpName?: string;
  groupName?: string;
  opCode?: string;
}

/** See corresponding proto */
export interface BufferSpan {
  start?: /* int32 */ number;
  limit?: /* int32 */ number;
}

/** See corresponding proto */
export interface LogicalBuffer {
  id?: /* int64 */ string;
  shape?: string;
  sizeMib?: /* double */ number;
  hloName?: string;
  shapeIndex?: /* int64 */ string[];
}

/** See corresponding proto */
export interface BufferAllocation {
  id?: /* int64 */ string;
  sizeMib?: /* double */ number;
  attributes?: string[];
  logicalBuffers?: LogicalBuffer[];
  commonShape?: string;
}

/** See corresponding proto */
export interface PreprocessResult {
  heapSizes?: /* double */ number[];
  unpaddedHeapSizes?: /* double */ number[];
  hloInstructionNames?: string[];
  maxHeap?: HeapObject[];
  maxHeapBySize?: HeapObject[];
  logicalBufferSpans?: {[key: /* int32 */ string]: BufferSpan};
  maxHeapToBySize?: /* int32 */ number[];
  bySizeToMaxHeap?: /* int32 */ number[];
  moduleName?: string;
  entryComputationName?: string;
  peakHeapMib?: /* double */ number;
  peakUnpaddedHeapMib?: /* double */ number;
  peakHeapSizePosition?: /* int32 */ number;
  entryComputationParametersMib?: /* double */ number;
  nonReusableMib?: /* double */ number;
  maybeLiveOutMib?: /* double */ number;
  totalBufferAllocationMib?: /* double */ number;
  indefiniteBufferAllocationMib?: /* double */ number;
  indefiniteLifetimes?: BufferAllocation[];
  allocationTimeline?: string;
}
