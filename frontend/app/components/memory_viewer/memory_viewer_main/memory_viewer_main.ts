import {Component, Input, OnChanges, OnDestroy} from '@angular/core';
import {Store} from '@ngrx/store';
import {BufferAllocationInfo} from 'org_xprof/frontend/app/common/interfaces/buffer_allocation_info';
import {type MemoryViewerPreprocessResult} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {MemoryUsage} from 'org_xprof/frontend/app/components/memory_viewer/memory_usage/memory_usage';
import {setActiveHeapObjectAction} from 'org_xprof/frontend/app/store/actions';

interface BufferSpan {
  alloc: number;
  free: number;
}

/** A memory viewer component. */
@Component({
  standalone: false,
  selector: 'memory-viewer-main',
  templateUrl: './memory_viewer_main.ng.html',
  styleUrls: ['./memory_viewer_main.scss']
})
export class MemoryViewerMain implements OnDestroy, OnChanges {
  /** Preprocessed result for memory viewer */
  @Input()
  memoryViewerPreprocessResult: MemoryViewerPreprocessResult|null = null;

  /** XLA memory space color */
  @Input() memorySpaceColor = '0';

  /** Current run, host and hlo module name */
  @Input() currentRun = '';
  @Input() currentHost = '';
  @Input() currentModule = '';

  peakInfo?: BufferAllocationInfo;
  activeInfo?: BufferAllocationInfo;
  peakHeapSizeMiB = '';
  paddingOverhead = '';
  totalArgumentSizeBytes = '';
  hloTempSizeBytes = '';
  hloTempFragmentation = '';
  timelineUrl = '';
  usage?: MemoryUsage;
  heapSizes: number[] = [];
  maxHeap: HeapObject[] = [];
  maxHeapBySize: HeapObject[] = [];
  maxHeapByPaddingSize: HeapObject[] = [];
  selectedIndex: number = -1;
  selectedIndexBySize: number = -1;
  selectedIndexByPaddingSize: number = -1;
  unpaddedHeapSizes: number[] = [];
  hloInstructionNames: string[] = [];
  hasTrace = false;
  diagnostics: Diagnostics = {info: [], warnings: [], errors: []};

  constructor(private readonly store: Store<{}>) {}

  ngOnChanges() {
    this.update();
  }

  ngOnDestroy() {
    this.dispatchActiveHeapObject();
  }

  private dispatchActiveHeapObject(heapObject: HeapObject|null = null) {
    this.store.dispatch(
        setActiveHeapObjectAction({activeHeapObject: heapObject}));
    if (heapObject) {
      const span = this.getLogicalBufferSpan(heapObject.logicalBufferId);
      this.activeInfo = {
        size: heapObject.sizeMiB || 0,
        alloc: span.alloc,
        free: span.free,
        color: utils.getChartItemColorByIndex(heapObject.color || 0),
      };
    } else {
      this.activeInfo = undefined;
      this.selectedIndex = -1;
      this.selectedIndexBySize = -1;
      this.selectedIndexByPaddingSize = -1;
    }
  }

  private getLogicalBufferSpan(index?: number): BufferSpan {
    const bufferSpan: BufferSpan = {alloc: 0, free: 0};
    if (index && this.usage && this.usage.logicalBufferSpans &&
        this.heapSizes) {
      const span = this.usage.logicalBufferSpans[index];
      if (span) {
        bufferSpan.alloc = span[0];
        bufferSpan.free = span[1] < 0 ? this.heapSizes.length - 1 : span[1];
      } else {
        bufferSpan.free = this.heapSizes.length - 1;
      }
    }
    return bufferSpan;
  }

  setSelectedHeapObject(selectedIndex: number) {
    if (!this.usage) {
      return;
    }
    if (selectedIndex === -1) {
      this.dispatchActiveHeapObject();
    } else {
      this.dispatchActiveHeapObject(this.usage.maxHeap[selectedIndex]);
      this.selectedIndexBySize = this.usage.maxHeapToBySize[selectedIndex];
      this.selectedIndexByPaddingSize =
          this.usage.maxHeapToByPaddingSize[selectedIndex];
    }
  }

  setSelectedHeapObjectBySize(selectedIndexBySize: number) {
    if (!this.usage) {
      return;
    }
    if (selectedIndexBySize === -1) {
      this.dispatchActiveHeapObject();
    } else {
      this.dispatchActiveHeapObject(
          this.usage.maxHeapBySize[selectedIndexBySize]);
      this.selectedIndex = this.usage.bySizeToMaxHeap[selectedIndexBySize];
      this.selectedIndexByPaddingSize =
          this.usage.maxHeapToByPaddingSize[this.selectedIndex];
    }
  }

  setSelectedHeapObjectByPaddingSize(selectedIndexByPaddingSize: number) {
    if (!this.usage) {
      return;
    }
    if (selectedIndexByPaddingSize === -1) {
      this.dispatchActiveHeapObject();
    } else {
      this.dispatchActiveHeapObject(
          this.maxHeapByPaddingSize[selectedIndexByPaddingSize]);
      this.selectedIndex =
          this.usage.byPaddingSizeToMaxHeap[selectedIndexByPaddingSize];
      this.selectedIndexBySize = this.usage.maxHeapToBySize[this.selectedIndex];
    }
  }

  update() {
    this.usage = new MemoryUsage(
        this.memoryViewerPreprocessResult, Number(this.memorySpaceColor),
        this.currentRun, this.currentHost, this.currentModule);
    if (this.usage.diagnostics.errors.length > 0) {
      return;
    }

    this.timelineUrl = this.usage.timelineUrl;

    this.peakHeapSizeMiB =
        utils.bytesToMiB(this.usage.peakHeapSizeBytes).toFixed(2);
    this.paddingOverhead =
        utils.bytesToMiB(this.usage.paddingOverhead).toFixed(2);
    this.totalArgumentSizeBytes =
        utils.bytesToMiB(this.usage.totalArgumentSizeBytes).toFixed(2);
    this.hloTempSizeBytes =
        utils.bytesToMiB(this.usage.hloTempSizeBytes).toFixed(2);
    this.hloTempFragmentation =
        (this.usage.hloTempFragmentation * 100.0).toFixed(2);
    this.heapSizes = this.usage.heapSizes || [];
    this.unpaddedHeapSizes = this.usage.unpaddedHeapSizes || [];
    this.hloInstructionNames = this.usage.hloInstructionNames || [];
    this.peakInfo = {
      size: utils.bytesToMiB(this.usage.peakHeapSizeBytes),
      alloc: this.usage.peakHeapSizePosition + 1,
      free: this.usage.peakHeapSizePosition + 2,
    };
    this.maxHeap = this.usage.maxHeap || [];
    this.maxHeapBySize = this.usage.maxHeapBySize || [];
    this.maxHeapByPaddingSize = this.usage.maxHeapByPaddingSize || [];

    this.hasTrace = this.maxHeap.length > 0 || this.heapSizes.length > 0 ||
        this.maxHeapBySize.length > 0;
  }
}
