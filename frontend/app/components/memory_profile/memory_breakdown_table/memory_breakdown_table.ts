import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  Input,
  OnChanges,
  OnInit,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {type MemoryProfileProto} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {bytesToGiBs} from 'org_xprof/frontend/app/common/utils/utils';

const DATA_TABLE_OPERATION_INDEX = 0;

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

/** A memory breakdown table view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'memory-breakdown-table',
  templateUrl: './memory_breakdown_table.ng.html',
  styleUrls: ['./memory_breakdown_table.scss'],
})
export class MemoryBreakdownTable implements OnChanges, OnInit {
  /** The memory profile proto data. */
  @Input() memoryProfileData: MemoryProfileProto | null = null;

  /** The selected memory ID to show memory profile for. */
  @Input() memoryId = '';

  @ViewChild('table', {static: false}) tableRef!: ElementRef;

  dataTable: google.visualization.DataTable | null = null;
  filterOperation = '';
  table: google.visualization.Table | null = null;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.dataTable = null;
    this.drawTable();
  }

  createDataTable() {
    if (
      !this.table ||
      !this.memoryProfileData ||
      !this.memoryProfileData.memoryProfilePerAllocator ||
      !!this.dataTable
    ) {
      return;
    }

    this.dataTable = new google.visualization.DataTable();
    this.dataTable.addColumn('string', 'Op Name');
    this.dataTable.addColumn('number', 'Allocation Size (GiBs)');
    this.dataTable.addColumn('number', 'Requested Size (GiBs)');
    this.dataTable.addColumn('number', 'Occurrences');
    this.dataTable.addColumn('string', 'Region type');
    this.dataTable.addColumn('string', 'Data type');
    this.dataTable.addColumn('string', 'Shape');

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
      return;
    }

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
      this.dataTable.addRow([
        cleanCellToken(metadata.tfOpName, 'System Reserved'),
        bytesToGiBs(metadata.allocationBytes),
        bytesToGiBs(metadata.requestedBytes),
        Number(activeAllocations[i].numOccurrences),
        cleanCellToken(metadata.regionType, 'Unallocated'),
        cleanCellToken(metadata.dataType, 'N/A'),
        cleanCellToken(metadata.tensorShape, 'N/A'),
      ]);
    }

    const decimalPtFormatter = new google.visualization.NumberFormat({
      fractionDigits: 3,
    });
    decimalPtFormatter.format(this.dataTable, 1); /* requested_size */
    decimalPtFormatter.format(this.dataTable, 2); /* allocation_size */
  }

  drawTable() {
    if (!this.table || !this.memoryProfileData) {
      return;
    }

    const dataView = this.getDataView();
    if (!dataView) {
      return;
    }

    const options = {
      allowHtml: true,
      alternatingRowStyle: false,
      showRowNumber: false,
      width: '100%',
      height: '600px',
      cssClassNames: {
        'headerCell': 'google-chart-table-header-cell',
        'tableCell': 'google-chart-table-table-cell',
      },
    };

    this.table.draw(dataView, options as google.visualization.TableOptions);
  }

  getDataView(): google.visualization.DataView | null {
    if (!this.dataTable) {
      this.createDataTable();
    }

    const dataTable = this.getFilteredDataTable();
    if (!dataTable) {
      return null;
    }

    const dataView = new google.visualization.DataView(dataTable);
    dataView.setRows(dataView.getFilteredRows([{column: 2, minValue: 0.001}]));
    return dataView;
  }

  getFilteredDataTable(): google.visualization.DataTable | null {
    if (!this.dataTable) {
      return null;
    }

    /* tslint:disable no-any */
    const filters = [];
    if (this.filterOperation.trim()) {
      const filter = this.filterOperation.trim().toLowerCase();
      filters.push({
        'column': DATA_TABLE_OPERATION_INDEX,
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
      } as any);
    }
    /* tslint:enable */

    if (filters.length > 0) {
      const dataView = new google.visualization.DataView(this.dataTable);
      dataView.setRows(this.dataTable.getFilteredRows(filters));
      return dataView.toDataTable();
    }

    return this.dataTable;
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
      return;
    }

    google.charts.safeLoad({'packages': ['table']});
    google.charts.setOnLoadCallback(() => {
      this.table = new google.visualization.Table(this.tableRef.nativeElement);
      this.drawTable();
    });
  }
}
