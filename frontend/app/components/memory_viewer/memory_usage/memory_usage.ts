import {MemoryViewerPreprocessResult} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';

interface MemoryUsageBytes {
  padded: number;
  unpadded: number;
}

/**
 * Provides calculation of memory usage from xla buffer assignment.
 * @final
 */
export class MemoryUsage {
  private nColor: number;

  totalBufferAllocationBytes: number;
  peakHeapSizeBytes: number;
  paddingOverhead: number;
  hloTempSizeWithoutFragmentationBytes: number;
  hloTempSizeWithFragmentationBytes: number;
  hloTempFragmentation: number;
  totalArgumentSizeBytes: number;
  peakLogicalBuffers: number[];
  peakHeapSizePosition: number;
  indefiniteMemoryUsageBytes: MemoryUsageBytes;
  heapSizes: number[];
  unpaddedHeapSizes: number[];
  hloInstructionNames: string[];
  maxHeap: HeapObject[];
  maxHeapBySize: HeapObject[];
  maxHeapByPaddingSize: HeapObject[];
  bySizeToMaxHeap: number[];
  maxHeapToBySize: number[];
  byPaddingSizeToMaxHeap: number[];
  maxHeapToByPaddingSize: number[];
  logicalBufferSpans: {[key: number]: number[]};

  smallBufferSize: number;

  diagnostics: {errors: string[], warnings: string[], info: string[]};
  // module name string in format of `<hlo_module_name>(<program_id>)`, should
  // be consistent with the hlo proto file assets name.
  moduleName: string;
  timelineUrl: string;

  // Only one of hloProto or preprocess is valid to construct MemoryUsage.
  constructor(
      preprocess: MemoryViewerPreprocessResult|null, memorySpaceColor: number,
      currentRun: string|null, currentHost: string|null,
      currentModule: string|null) {
    this.nColor = 0;

    this.totalBufferAllocationBytes = 0;
    this.peakHeapSizeBytes = 0;
    this.paddingOverhead = 0;
    this.hloTempSizeWithoutFragmentationBytes = 0;
    this.hloTempSizeWithFragmentationBytes = 0;
    this.hloTempFragmentation = 0;
    this.totalArgumentSizeBytes = 0;
    this.peakLogicalBuffers = [];
    this.peakHeapSizePosition = 0;
    this.indefiniteMemoryUsageBytes = {padded: 0, unpadded: 0};
    this.heapSizes = [];
    this.unpaddedHeapSizes = [];
    this.hloInstructionNames = [];
    this.maxHeap = [];
    this.maxHeapBySize = [];
    this.maxHeapByPaddingSize = [];
    this.bySizeToMaxHeap = [];
    this.maxHeapToBySize = [];
    this.byPaddingSizeToMaxHeap = [];
    this.maxHeapToByPaddingSize = [];
    this.logicalBufferSpans = {};
    this.smallBufferSize = 16 * 1024;
    this.diagnostics = {errors: [], warnings: [], info: []};
    this.moduleName = currentModule || '';
    this.timelineUrl = '';

    // Both input sources (HLOProto and preprocessed data) are invalid.
    if (!preprocess) {
      this.diagnostics.errors.push(
          'We failed to fetch a valid input. The input is empty or too large.');
      return;
    }

    if (preprocess) {
      // Initialize memory viewer from preprocessed data.
      this.initMemoryUsageFromPrecomputed(
          preprocess, currentRun, currentHost, currentModule);
    }
  }

  /**
   * Initializes memory usage from precomputed results.
   */
  private initMemoryUsageFromPrecomputed(
      preprocess: MemoryViewerPreprocessResult, currentRun: string|null,
      currentHost: string|null, currentModule: string|null) {
    // Copy the fields from preprocessed result.
    this.timelineUrl = preprocess.allocationTimeline || '';
    if (!this.timelineUrl.startsWith('/memory_viewer.json')) {
      // redirecting memory allocation timeline to this url on TensorBoard
      this.timelineUrl =
          `${window.parent.location.origin}/data/plugin/profile/data?run=${
              currentRun}&tag=memory_viewer&module_name=${
              currentModule}&view_memory_allocation_timeline=true`;
    }
    this.totalBufferAllocationBytes =
        (preprocess.totalBufferAllocationMib || 0) * 1024 * 1024;
    this.peakHeapSizeBytes = (preprocess.peakHeapMib || 0) * 1024 * 1024;
    this.paddingOverhead = this.peakHeapSizeBytes -
        (preprocess.peakUnpaddedHeapMib || 0) * 1024 * 1024;
    this.totalArgumentSizeBytes =
        (preprocess.entryComputationParametersMib || 0) * 1024 * 1024;
    this.hloTempSizeWithoutFragmentationBytes = this.peakHeapSizeBytes -
        (preprocess.indefiniteBufferAllocationMib || 0) * 1024 * 1024;
    this.hloTempSizeWithFragmentationBytes = this.totalBufferAllocationBytes -
        (preprocess.indefiniteBufferAllocationMib || 0) * 1024 * 1024;
    const fragmentationSizeBytes =
        this.totalBufferAllocationBytes - this.peakHeapSizeBytes;
    if (this.hloTempSizeWithFragmentationBytes) {
      this.hloTempFragmentation =
          fragmentationSizeBytes / this.hloTempSizeWithFragmentationBytes;
    }

    this.peakHeapSizePosition = (preprocess.peakHeapSizePosition || 0);
    this.heapSizes = preprocess.heapSizes || [];
    this.unpaddedHeapSizes = preprocess.unpaddedHeapSizes || [];
    this.hloInstructionNames = preprocess.hloInstructionNames || [];
    if (preprocess.logicalBufferSpans) {
      for (const [key, value] of Object.entries(
               preprocess.logicalBufferSpans)) {
        this.logicalBufferSpans[utils.toNumber(key)] =
            [value.start || 0, value.limit || 0];
      }
    }
    for (const heapObject of preprocess.maxHeap || []) {
      this.maxHeap.push({
        instructionName: heapObject.instructionName,
        shape: heapObject.shapeString,
        tfOpName: heapObject.tfOpName,
        sizeMiB: heapObject.logicalBufferSizeMib,
        unpaddedSizeMiB: heapObject.unpaddedShapeMib,
        color: this.nColor++,
        groupName: heapObject.groupName,
        opcode: heapObject.opCode,
        logicalBufferId: heapObject.logicalBufferId,
        sourceInfo: heapObject.sourceInfo,
      });
    }
    this.createMaxHeapIndex();
  }

  /**
   * Create index for this.maxHeap so it can be selected by size, unpadded size
   * and etc.
   */
  private createMaxHeapIndex() {
    const indexedMaxHeap = this.maxHeap.map((e, i) => {
      return {ind: i, val: e};
    });
    // Sort max heap objects by size and create index.
    indexedMaxHeap.sort((a, b) => {
      const sizeA = a && a.val && a.val.sizeMiB ? a.val.sizeMiB : 0;
      const sizeB = b && b.val && b.val.sizeMiB ? b.val.sizeMiB : 0;
      return sizeB - sizeA;
    });
    this.maxHeapBySize = indexedMaxHeap.map(e => e.val);
    this.bySizeToMaxHeap = indexedMaxHeap.map(e => e.ind);
    this.maxHeapToBySize.length = this.maxHeap.length;
    for (let i = 0; i < this.bySizeToMaxHeap.length; i++) {
      this.maxHeapToBySize[this.bySizeToMaxHeap[i]] = i;
    }
    // Sort max heap objects by padding size and create index.
    indexedMaxHeap.sort((a, b) => {
      const paddingA = a && a.val && a.val.sizeMiB && a.val.unpaddedSizeMiB ?
          a.val.sizeMiB - a.val.unpaddedSizeMiB :
          0;
      const paddingB = b && b.val && b.val.sizeMiB && b.val.unpaddedSizeMiB ?
          b.val.sizeMiB - b.val.unpaddedSizeMiB :
          0;
      return paddingB - paddingA;
    });
    this.maxHeapByPaddingSize = indexedMaxHeap.map(e => e.val);
    this.byPaddingSizeToMaxHeap = indexedMaxHeap.map(e => e.ind);
    this.maxHeapToByPaddingSize.length = this.maxHeap.length;
    for (let i = 0; i < this.byPaddingSizeToMaxHeap.length; i++) {
      this.maxHeapToByPaddingSize[this.byPaddingSizeToMaxHeap[i]] = i;
    }
  }
}
