import {ChangeDetectionStrategy, ChangeDetectorRef, Component, inject, Input, OnChanges, OnInit, SimpleChanges, ViewChild} from '@angular/core';
import {MatPaginator} from '@angular/material/paginator';
import {MatSort} from '@angular/material/sort';
import {MatTableDataSource} from '@angular/material/table';
import {type MemoryProfileProto} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {bytesToGiBs} from 'org_xprof/frontend/app/common/utils/utils';

/** Set of uninitialized or null-like sentinel string tokens. */
export const NULL_TOKENS = new Set([
  '(null)',
  '<null>',
  'null',
  'nullptr',
  'invalid',
  'undefined',
  'none',
]);

/**
 * Sanitizes a table cell value against uninitialized sentinel tokens.
 *
 * @param val The cell string value to sanitize.
 * @param fallback The default fallback string if val is null, empty, or sentinel.
 * @return The trimmed value or fallback string.
 */
export function cleanCellToken(
  val: string | null | undefined,
  fallback: string,
): string {
  if (!val) return fallback;
  const trimmed = val.trim();
  if (!trimmed || NULL_TOKENS.has(trimmed.toLowerCase())) {
    return fallback;
  }
  return trimmed;
}

/** Row data for memory breakdown table. */
export interface MemoryBreakdownRow {
  opName: string;
  allocationSizeGiBs: number;
  requestedSizeGiBs: number;
  occurrences: number;
  regionType: string;
  dataType: string;
  shape: string;
}

/** A memory breakdown table view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'memory-breakdown-table',
  templateUrl: './memory_breakdown_table.ng.html',
  styleUrls: ['./memory_breakdown_table.scss']
})
export class MemoryBreakdownTable implements OnChanges, OnInit {
  /** The memory profile proto data. */
  @Input() memoryProfileData: MemoryProfileProto|null = null;

  /** The selected memory ID to show memory profile for. */
  @Input() memoryId: string = '';

  filterOperation: string = '';

  readonly displayedColumns: string[] = [
    'opName',
    'allocationSizeGiBs',
    'requestedSizeGiBs',
    'occurrences',
    'regionType',
    'dataType',
    'shape',
  ];

  readonly dataSource = new MatTableDataSource<MemoryBreakdownRow>([]);

  @ViewChild(MatSort) set sort(sort: MatSort | null) {
    if (sort) this.dataSource.sort = sort;
  }

  @ViewChild(MatPaginator) set paginator(paginator: MatPaginator | null) {
    if (paginator) this.dataSource.paginator = paginator;
  }

  private readonly cdr = inject(ChangeDetectorRef);

  ngOnInit() {
    this.setupDataSource();
    this.updateMatTableData();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.updateMatTableData();
  }

  private setupDataSource() {
    this.dataSource.sortingDataAccessor = (
      item: MemoryBreakdownRow,
      prop: string,
    ): string | number =>
      prop === 'allocationSizeGiBs' ? item.allocationSizeGiBs :
      prop === 'requestedSizeGiBs' ? item.requestedSizeGiBs :
      prop === 'occurrences' ? item.occurrences :
      (prop === 'opName' ? item.opName : prop === 'regionType' ? item.regionType :
       prop === 'dataType' ? item.dataType : item.shape).toLowerCase();
    this.dataSource.filterPredicate = (
      data: MemoryBreakdownRow,
      filter: string,
    ) => {
      const trimmed = filter.trim().toLowerCase();
      return trimmed ? data.opName.toLowerCase().includes(trimmed) : true;
    };
  }

  onFilterChanged(filter: string) {
    this.filterOperation = filter;
    this.applyFilter();
  }

  trackByRow(index: number, item: MemoryBreakdownRow): string {
    return `${item.opName}_${item.regionType}_${item.allocationSizeGiBs}_${index}`;
  }

  private applyFilter() {
    this.dataSource.filter = this.filterOperation;
    if (this.dataSource.paginator) {
      this.dataSource.paginator.firstPage();
    }
    this.cdr.markForCheck();
  }

  updateMatTableData() {
    this.dataSource.data = this.getRowsFromData();
    this.applyFilter();
    this.cdr.markForCheck();
  }

  getRowsFromData(): MemoryBreakdownRow[] {
    if (!this.memoryProfileData ||
        !this.memoryProfileData.memoryProfilePerAllocator ||
        !this.memoryProfileData.memoryProfilePerAllocator[this.memoryId]) {
      return [];
    }

    const snapshots =
        this.memoryProfileData.memoryProfilePerAllocator[this.memoryId]
            .memoryProfileSnapshots;
    const activeAllocations =
        this.memoryProfileData.memoryProfilePerAllocator[this.memoryId]
            .activeAllocations;
    const specialAllocations =
        this.memoryProfileData.memoryProfilePerAllocator[this.memoryId]
            .specialAllocations;
    if (!snapshots || !activeAllocations || !specialAllocations) {
      return [];
    }

    const rows: MemoryBreakdownRow[] = [];
    for (let i = 0; i < activeAllocations.length; i++) {
      const index: number = Number(activeAllocations[i].snapshotIndex);
      const specialIndex = Number(activeAllocations[i].specialIndex);
      // Use snapshot index or special index, whichever is positve.
      let metadata;
      if (index >= 0) {
        // It may be dropped depending on the max_num_snapshots query parameter
        // which is set to 1000 by default.
        if (!(index in snapshots)) continue;
        metadata = snapshots[index].activityMetadata;
      } else {
        metadata = specialAllocations[specialIndex];
      }
      if (!metadata) {
        continue;
      }
      rows.push({
        opName: cleanCellToken(metadata.tfOpName, 'System Reserved'),
        allocationSizeGiBs: bytesToGiBs(metadata.allocationBytes),
        requestedSizeGiBs: bytesToGiBs(metadata.requestedBytes),
        occurrences: Number(activeAllocations[i].numOccurrences) || 0,
        regionType: cleanCellToken(metadata.regionType, 'Unallocated'),
        dataType: cleanCellToken(metadata.dataType, 'N/A'),
        shape: cleanCellToken(metadata.tensorShape, 'N/A'),
      });
    }

    const hasLargeAllocations = rows.some((r) => r.requestedSizeGiBs >= 0.001);
    return hasLargeAllocations
      ? rows.filter((r) => r.requestedSizeGiBs >= 0.001)
      : rows;
  }
}
