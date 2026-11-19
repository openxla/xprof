import {AfterViewInit, ChangeDetectorRef, Component, Input, OnChanges, SimpleChanges, ChangeDetectionStrategy} from '@angular/core';
import {type MemoryProfileProto} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {humanReadableText} from 'org_xprof/frontend/app/common/utils/utils';

/** A memory profile summary view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'memory-profile-summary',
  templateUrl: './memory_profile_summary.ng.html',
  styleUrls: ['./memory_profile_summary.scss']
})
export class MemoryProfileSummary implements AfterViewInit, OnChanges {
  /** The memory profile summary data. */
  @Input() data: MemoryProfileProto|null = null;

  /** The selected memory ID to show memory profile for. */
  @Input() memoryId = '';

  constructor(private readonly changeDetector: ChangeDetectorRef) {}

  ngAfterViewInit() {
    this.memoryProfileSummary();
    this.changeDetector.detectChanges();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.memoryProfileSummary();
  }

  memoryProfileSummary() {
    this.resetSummary();
    if (!this.data || !this.data.memoryIds || !this.data.memoryIds.length ||
        !this.data.memoryProfilePerAllocator) {
      return;
    }

    const allocatorProfile =
      this.data.memoryProfilePerAllocator[this.memoryId];
    if (!allocatorProfile) {
      return;
    }

    const summary = allocatorProfile.profileSummary;
    if (!summary || !summary.peakStats) {
      return;
    }
    const peakStats = summary.peakStats;

    let numAllocations = 0;
    let numDeallocations = 0;
    if (
      summary.numAllocations === undefined ||
      summary.numDeallocations === undefined
    ) {
      // If version is set to 1, this means the backend is using the new
      // snapshot sampling algorithm, timeline data is stored in
      // sampledTimelineSnapshots.
      const snapshots =
        (this.data.version === 1
          ? allocatorProfile.sampledTimelineSnapshots
          : allocatorProfile.memoryProfileSnapshots) || [];
      for (const snapshot of snapshots) {
        const activity = snapshot?.activityMetadata?.memoryActivity;
        if (activity === 'ALLOCATION') {
          numAllocations++;
        } else if (activity === 'DEALLOCATION') {
          numDeallocations++;
        }
      }
    }

    this.numAllocations =
      summary.numAllocations !== undefined
        ? Number(summary.numAllocations)
        : numAllocations;
    this.numDeallocations =
      summary.numDeallocations !== undefined
        ? Number(summary.numDeallocations)
        : numDeallocations;
    this.memoryCapacity =
        humanReadableText(Number(summary.memoryCapacity) || 0);
    this.peakHeapUsageLifetime =
        humanReadableText(Number(summary.peakBytesUsageLifetime) || 0);
    this.timestampAtPeakMs =
        this.picoToMilli(summary.peakStatsTimePs).toFixed(1);
    this.peakMemUsageProfile =
        humanReadableText(Number(peakStats.peakBytesInUse) || 0);
    this.stackAtPeak =
        humanReadableText(Number(peakStats.stackReservedBytes) || 0);
    this.heapAtPeak =
        humanReadableText(Number(peakStats.heapAllocatedBytes) || 0);
    this.freeAtPeak = humanReadableText(Number(peakStats.freeMemoryBytes) || 0);
    this.fragmentationAtPeakPct =
        ((peakStats.fragmentation || 0) * 100).toFixed(2) + '%';
  }

  private resetSummary() {
    this.numAllocations = 0;
    this.numDeallocations = 0;
    this.memoryCapacity = '';
    this.peakHeapUsageLifetime = '';
    this.peakMemUsageProfile = '';
    this.timestampAtPeakMs = '';
    this.stackAtPeak = '';
    this.heapAtPeak = '';
    this.freeAtPeak = '';
    this.fragmentationAtPeakPct = '';
  }

  picoToMilli(timePs: string|undefined) {
    if (!timePs) return 0;
    return Number(timePs) / Math.pow(10, 9);
  }

  title = 'Memory Profile Summary';
  numAllocations = 0;
  numDeallocations = 0;
  memoryCapacity = '';
  peakHeapUsageLifetime = '';
  peakMemUsageProfile = '';
  timestampAtPeakMs = '';
  stackAtPeak = '';
  heapAtPeak = '';
  freeAtPeak = '';
  fragmentationAtPeakPct = '';
}
