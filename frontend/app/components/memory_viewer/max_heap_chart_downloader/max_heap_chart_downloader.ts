import {CommonModule} from '@angular/common';
import {Component, Input} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {type MemoryViewerPreprocessResult} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import {MemoryUsage} from 'org_xprof/frontend/app/components/memory_viewer/memory_usage/memory_usage';

/** A component to download hlo module in proto, text or json formats. */
@Component({
  standalone: true,
  selector: 'max-heap-chart-downloader',
  templateUrl: 'max_heap_chart_downloader.ng.html',
  styleUrls: ['max_heap_chart_downloader.scss'],
  imports: [CommonModule, MatButtonModule],
  providers: [],
})
export class MaxHeapChartDownloader {
  /** Preprocessed result for memory viewer */
  @Input()
  memoryViewerPreprocessResult: MemoryViewerPreprocessResult | null = null;

  /** XLA memory space color */
  @Input() memorySpaceColor = '0';

  /** Heap objects to download. */
  heapObjects: HeapObject[] = [];
  /** Timespan (alloc and free) for logical buffer. */
  logicalBufferSpans: {[key: number]: number[]} = {};
  /** Heap size sequence. */
  heapSizes: number[] = [];

  private readonly data: string[][] = [];
  sessionId = '';

  private getLogicalBufferSpan(index?: number): [number, number] {
    const bufferSpan: [number, number] = [0, 0];
    if (index) {
      const span = this.logicalBufferSpans[index];
      if (span) {
        bufferSpan[0] = span[0];
        bufferSpan[1] = span[1] < 0 ? this.heapSizes.length - 1 : span[1];
      } else {
        bufferSpan[1] = this.heapSizes.length - 1;
      }
    }
    return bufferSpan;
  }

  async downloadMaxHeapChart() {
    const usage = new MemoryUsage(
      this.memoryViewerPreprocessResult,
      Number(this.memorySpaceColor),
      null,
      null,
      null,
    );
    if (usage.diagnostics.errors.length > 0) {
      console.error(usage.diagnostics.errors[0]);
      return;
    }

    this.heapObjects = usage.maxHeap || [];
    this.logicalBufferSpans = usage.logicalBufferSpans || {};
    this.heapSizes = usage.heapSizes || [];

    const moduleName = this.memoryViewerPreprocessResult?.moduleName || '';
    const fileName = moduleName + '.csv';
    this.data.push([
      'InstructionName',
      'UnpaddedSizeMiB',
      'SizeMiB',
      'AllocAt',
      'FreeAt',
      'TfOpName',
      'OpCode',
      'Shape',
      'AllocationType',
    ]);
    for (const heapObject of this.heapObjects) {
      const span = this.getLogicalBufferSpan(heapObject.logicalBufferId);
      this.data.push([
        heapObject.instructionName || 'UNKNOWN',
        heapObject.unpaddedSizeMiB
          ? heapObject.unpaddedSizeMiB.toString()
          : 'UNKNOWN',
        heapObject.sizeMiB ? heapObject.sizeMiB.toString() : 'UNKNOWN',
        span[0].toString(),
        span[1].toString(),
        heapObject.tfOpName || 'UNKNOWN',
        heapObject.opcode || 'UNKNOWN',
        heapObject.shape || 'UNKNOWN',
        heapObject.groupName || 'UNKNOWN',
      ]);
    }
  }
}
