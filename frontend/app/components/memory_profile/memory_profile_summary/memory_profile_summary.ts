import {ChangeDetectionStrategy, Component, effect, input} from '@angular/core';
import {MatCard, MatCardContent, MatCardTitle} from '@angular/material/card';
import {MatIcon} from '@angular/material/icon';
import {MatTooltip} from '@angular/material/tooltip';
import {type MemoryProfileProto} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {humanReadableText} from 'org_xprof/frontend/app/common/utils/utils';

/** A memory profile summary view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'memory-profile-summary',
  templateUrl: './memory_profile_summary.ng.html',
  styleUrls: ['./memory_profile_summary.scss'],
  imports: [MatCard, MatCardContent, MatCardTitle, MatIcon, MatTooltip],
})
export class MemoryProfileSummary {
  /** The memory profile summary data. */
  readonly data = input<MemoryProfileProto | null>(null);

  /** The selected memory ID to show memory profile for. */
  readonly memoryId = input('');

  constructor() {
    effect(() => {
      this.memoryProfileSummary();
    });
  }

  memoryProfileSummary() {
    const data = this.data();
    const memoryId = this.memoryId();
    if (
      !data ||
      !data.memoryIds ||
      !data.memoryIds.length ||
      !data.memoryProfilePerAllocator ||
      !data.memoryProfilePerAllocator[memoryId]
    ) {
      return;
    }

    const summary = data.memoryProfilePerAllocator[memoryId].profileSummary;
    let snapshots =
      data.memoryProfilePerAllocator[memoryId].memoryProfileSnapshots;
    // If version is set to 1, this means the backend is using the new snapshot
    // sampling algorithm, timeline data is stored in sampledTimelineSnapshots.
    if (data.version === 1) {
      snapshots =
        data.memoryProfilePerAllocator[memoryId].sampledTimelineSnapshots;
    }
    if (!summary || !snapshots) {
      return;
    }

    const peakStats = summary.peakStats;
    if (!peakStats) {
      return;
    }

    let numAllocations = 0;
    let numDeallocations = 0;
    for (let i = 0; i < snapshots.length; i++) {
      const snapshot = snapshots[i];
      if (
        !snapshot ||
        !snapshot.activityMetadata ||
        !snapshot.activityMetadata.memoryActivity
      ) {
        return;
      }
      if (snapshot.activityMetadata.memoryActivity === 'ALLOCATION') {
        numAllocations++;
      } else if (snapshot.activityMetadata.memoryActivity === 'DEALLOCATION') {
        numDeallocations++;
      }
    }

    this.numAllocations = numAllocations;
    this.numDeallocations = numDeallocations;
    this.memoryCapacity = humanReadableText(
      Number(summary.memoryCapacity) || 0,
    );
    this.peakHeapUsageLifetime = humanReadableText(
      Number(summary.peakBytesUsageLifetime) || 0,
    );
    this.timestampAtPeakMs = this.picoToMilli(summary.peakStatsTimePs).toFixed(
      1,
    );
    this.peakMemUsageProfile = humanReadableText(
      Number(peakStats.peakBytesInUse) || 0,
    );
    this.stackAtPeak = humanReadableText(
      Number(peakStats.stackReservedBytes) || 0,
    );
    this.heapAtPeak = humanReadableText(
      Number(peakStats.heapAllocatedBytes) || 0,
    );
    this.freeAtPeak = humanReadableText(Number(peakStats.freeMemoryBytes) || 0);
    this.fragmentationAtPeakPct =
      ((peakStats.fragmentation || 0) * 100).toFixed(2) + '%';
  }

  picoToMilli(timePs: string | undefined) {
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
