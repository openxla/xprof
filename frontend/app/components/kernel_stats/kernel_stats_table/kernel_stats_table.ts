import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  inject,
  Input,
  OnChanges,
  OnInit,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {type SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {loadGoogleChartPackages} from 'org_xprof/frontend/app/common/utils/utils';

declare interface KernelStatsColumn {
  rank: number;
  kernelName: number;
  registersPerThread: number;
  shmemBytes: number;
  blockDim: number;
  gridDim: number;
  occupancyPct: number;
  isKernelUsingTensorCore: number;
  isOpTensorCoreEligible: number;
  opName: number;
  occurrences: number;
  totalDurationUs: number;
  avgDurationUs: number;
  minDurationUs: number;
  maxDurationUs: number;
}

declare interface SortEvent {
  column: number;
  ascending: boolean;
}

/** A kernel stats table view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'kernel-stats-table',
  templateUrl: './kernel_stats_table.ng.html',
  styleUrls: ['./kernel_stats_table.scss'],
})
export class KernelStatsTable implements OnChanges, OnInit {
  /** The kernel stats data. */
  @Input() kernelStatsData: SimpleDataTable | null = null;

  @ViewChild('table', {static: false}) tableRef!: ElementRef;
  private readonly destroyRef = inject(DestroyRef);

  columns: KernelStatsColumn = createDefaultColumns();
  dataTable: google.visualization.DataTable | null = null;
  filterKernelName = '';
  filterOpName = '';
  sortAscending = true;
  sortColumn = 0;
  table: google.visualization.Table | null = null;

  loading = true;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.dataTable = null;
    this.drawTable();
  }

  createDataTable() {
    if (!this.table || !this.kernelStatsData || !!this.dataTable) {
      return;
    }

    const dataTable = new google.visualization.DataTable(this.kernelStatsData);
    this.dataTable = dataTable;
    this.enumerateColumns();
    if (dataTable.getNumberOfRows() > 0) {
      const setWidthStyle = (columnIndex: number, style: string) => {
        if (columnIndex >= 0) {
          dataTable.setProperty(0, columnIndex, 'style', style);
        }
      };
      setWidthStyle(this.columns.kernelName, 'width: 30%');
      setWidthStyle(this.columns.opName, 'width: 25%');
    }
  }

  drawTable() {
    if (!this.table || !this.kernelStatsData) {
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
      cssClassNames: {
        'headerCell': 'google-chart-table-header-cell',
        'tableCell': 'google-chart-table-table-cell',
      },
      sort: 'event',
      sortAscending: this.sortAscending,
      sortColumn: this.sortColumn,
      page: 'enable',
      pageSize: 100,
    };

    this.table.draw(dataView, options as google.visualization.TableOptions);
  }

  enumerateColumns() {
    this.columns = createDefaultColumns();
    if (!this.dataTable) {
      return;
    }

    for (let i = 0; i < this.dataTable.getNumberOfColumns(); i++) {
      switch (this.dataTable.getColumnId(i)) {
        case 'rank':
          this.columns.rank = i;
          break;
        case 'kernel_name':
          this.columns.kernelName = i;
          break;
        case 'registers_per_thread':
          this.columns.registersPerThread = i;
          break;
        case 'shmem_bytes':
          this.columns.shmemBytes = i;
          break;
        case 'block_dim':
          this.columns.blockDim = i;
          break;
        case 'grid_dim':
          this.columns.gridDim = i;
          break;
        case 'occupancy_pct':
          this.columns.occupancyPct = i;
          break;
        case 'is_kernel_using_tensor_core':
          this.columns.isKernelUsingTensorCore = i;
          break;
        case 'is_op_tensor_core_eligible':
          this.columns.isOpTensorCoreEligible = i;
          break;
        case 'op_name':
          this.columns.opName = i;
          break;
        case 'occurrences':
          this.columns.occurrences = i;
          break;
        case 'total_duration_us':
          this.columns.totalDurationUs = i;
          break;
        case 'avg_duration_us':
          this.columns.avgDurationUs = i;
          break;
        case 'min_duration_us':
          this.columns.minDurationUs = i;
          break;
        case 'max_duration_us':
          this.columns.maxDurationUs = i;
          break;
        default:
          break;
      }
    }
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
    const viewColumns = this.getViewColumns(dataTable.getNumberOfColumns());
    // google.visualization.Table 'sort' events emit DataView column indices,
    // which differ from DataTable column indices because the TensorCore columns
    // are swapped in the view.
    if (this.sortColumn >= 0 && this.sortColumn < viewColumns.length) {
      const sortedIndex = dataTable.getSortedRows({
        column: viewColumns[this.sortColumn],
        desc: !this.sortAscending,
      });
      dataView.setRows(sortedIndex);
    }
    dataView.setColumns(viewColumns);

    return dataView;
  }

  getFilteredDataTable(): google.visualization.DataTable | null {
    if (!this.dataTable || this.dataTable.getNumberOfColumns() <= 0) {
      return null;
    }

    const filters = [];
    if (this.filterKernelName.trim() && this.columns.kernelName >= 0) {
      const filter = this.filterKernelName.trim().toLowerCase();
      filters.push({
        'column': this.columns.kernelName,
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
        // tslint:disable-next-line:no-any
      } as any);
    }
    if (this.filterOpName.trim() && this.columns.opName >= 0) {
      const filter = this.filterOpName.trim().toLowerCase();
      filters.push({
        'column': this.columns.opName,
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
        // tslint:disable-next-line:no-any
      } as any);
    }

    if (filters.length > 0) {
      const dataView = new google.visualization.DataView(this.dataTable);
      dataView.setRows(this.dataTable.getFilteredRows(filters));
      return dataView.toDataTable();
    }

    return this.dataTable;
  }

  loadGoogleChart() {
    const cancelLoad = loadGoogleChartPackages(['table'], () => {
      this.table = new google.visualization.Table(this.tableRef.nativeElement);
      google.visualization.events.addListener(
        this.table,
        'sort',
        (event: SortEvent) => {
          this.sortColumn = event.column;
          this.sortAscending = event.ascending;
          this.drawTable();
        },
      );
      this.drawTable();
    });
    this.destroyRef.onDestroy(cancelLoad);
  }

  private getViewColumns(columnCount: number): number[] {
    return [
      this.columns.rank,
      this.columns.kernelName,
      this.columns.registersPerThread,
      this.columns.shmemBytes,
      this.columns.blockDim,
      this.columns.gridDim,
      this.columns.occupancyPct,
      this.columns.isKernelUsingTensorCore,
      this.columns.isOpTensorCoreEligible,
      this.columns.opName,
      this.columns.occurrences,
      this.columns.totalDurationUs,
      this.columns.avgDurationUs,
      this.columns.minDurationUs,
      this.columns.maxDurationUs,
    ].filter((col) => col >= 0 && col < columnCount);
  }
}

function createDefaultColumns(): KernelStatsColumn {
  return {
    rank: -1,
    kernelName: -1,
    registersPerThread: -1,
    shmemBytes: -1,
    blockDim: -1,
    gridDim: -1,
    occupancyPct: -1,
    isKernelUsingTensorCore: -1,
    isOpTensorCoreEligible: -1,
    opName: -1,
    occurrences: -1,
    totalDurationUs: -1,
    avgDurationUs: -1,
    minDurationUs: -1,
    maxDurationUs: -1,
  };
}
