import {EventEmitter} from '@angular/core';
import {ChartClass, ChartOptions} from 'org_xprof/frontend/app/common/interfaces/chart';
import {FrameworkOpStatsData, SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {computeDiffTable} from 'org_xprof/frontend/app/components/chart/table_utils';

declare interface SortEvent {
  column: number;
  ascending: boolean;
}

const DATA_TABLE_RANK_INDEX = 1;
const DATA_TABLE_DEVICE_PERCENT_INDEX = 10;
const DATA_TABLE_CUMULATIVE_DEVICE_PERCENT_INDEX = 11;
const DATA_TABLE_HOST_PERCENT_INDEX = 12;
const DATA_TABLE_CUMULATIVE_HOST_PERCENT_INDEX = 13;
const DATA_VIEW_CUMULATIVE_DEVICE_PERCENT_INDEX = 10;
const DATA_VIEW_CUMULATIVE_HOST_PERCENT_INDEX = 12;
const MAXIMUM_ROWS = 1000;
const MINIMUM_ROWS = 20;

/** A stats table data provider. */
export class StatsTableDataProvider extends DefaultDataProvider {
  diffTable?: google.visualization.DataTable;
  private rawDiffData: FrameworkOpStatsData | null = null;
  hasDiff = false;
  sortAscending = true;
  sortColumn = -1;
  options = {
    allowHtml: true,
    alternatingRowStyle: false,
    showRowNumber: false,
    width: '100%',
    height: '600px',
    cssClassNames: {
      'headerCell': 'google-chart-table-header-cell',
      'tableCell': 'google-chart-table-table-cell',
    },
    sort: 'event',
    sortAscending: true,
    sortColumn: -1,
  };

  private readonly totalOperationsChanged = new EventEmitter<string>();

  override setChart(chart: ChartClass) {
    this.chart = chart;
    google.visualization.events.addListener(
        this.chart, 'sort', (event: SortEvent) => {
          this.sortColumn = event.column;
          this.sortAscending = event.ascending;
          this.update.emit();
        });
  }

  override parseData(data: SimpleDataTable|Array<Array<(string | number)>>|null) {
    if (!data) {
      this.pendingData = null;
      this.dataTable = undefined;
      return;
    }
    if (!this.ensureChartPackagesLoaded(data)) return;
    const dataTable = new google.visualization.DataTable(data);
    this.diffTable = this.rawDiffData
        ? new google.visualization.DataTable(this.rawDiffData)
        : undefined;
    if (this.hasDiff && this.diffTable) {
      this.preProcessDiffTable(dataTable, this.diffTable);
    } else {
      this.preProcessDataTable(dataTable);
    }
  }

  setDiffData(diffData: FrameworkOpStatsData|null) {
    this.rawDiffData = diffData;
    if (diffData && this.isDataTableReady()) {
      this.diffTable = new google.visualization.DataTable(diffData);
    } else {
      this.diffTable = undefined;
    }
  }

  override setFilters(filters: google.visualization.DataTableCellFilter[]) {
    this.filters = filters;
    if (this.dataTable) {
      this.update.emit();
    }
  }

  override process(): google.visualization.DataTable|google.visualization.DataView|null {
    if (!this.dataTable) {
      return null;
    }

    let dataTable = this.dataTable;
    let dataView: google.visualization.DataView|null = null;

    if (this.filters && this.filters.length > 0) {
      dataView = new google.visualization.DataView(this.dataTable);
      dataView.setRows(this.dataTable.getFilteredRows(this.filters));
      dataTable = dataView.toDataTable();
    }

    if (!this.hasDiff) {
      dataView = this.formatDataTable(dataTable);
    } else {
      dataView = this.formatDiffTable(dataTable);
    }

    this.options.height =
        dataView.getNumberOfRows() < MINIMUM_ROWS ? '' : '600px';

    return dataView;
  }

  override getOptions(): ChartOptions|null {
    this.options.sortAscending = this.sortAscending;
    this.options.sortColumn = this.sortColumn;
    return this.options;
  }

  setTotalOperationsChangedEventListener(callback: Function) {
    this.totalOperationsChanged.subscribe(callback);
  }

  preProcessDataTable(dataTable: google.visualization.DataTable) {
    this.dataTable = dataTable.clone();
    this.updateTotalOperations(this.dataTable);

    const zeroDecimalPtFormatter =
        new google.visualization.NumberFormat({'fractionDigits': 0});
    zeroDecimalPtFormatter.format(this.dataTable, 5); /** total_time */
    zeroDecimalPtFormatter.format(this.dataTable, 6); /** avg_time */
    zeroDecimalPtFormatter.format(this.dataTable, 7); /** total_self_time */
    zeroDecimalPtFormatter.format(this.dataTable, 8); /** avg_self_time */

    const percentFormatter =
        new google.visualization.NumberFormat({pattern: '##.#%'});
    percentFormatter.format(
        this.dataTable, 9); /** device_total_self_time_percent */
    percentFormatter.format(
        this.dataTable, 11); /** host_total_self_time_percent */

    /**
     * Format tensorcore utilization column if it exists in dataTable.
     * This column does not exist in dataTable if the device is not GPU.
     */
    for (let i = 0; i < dataTable.getNumberOfColumns(); i++) {
      if (this.dataTable.getColumnId(i) === 'gpu_tensorcore_utilization') {
        percentFormatter.format(this.dataTable, i);
        break;
      }
    }

    // Insert column made the table filter column index drift by one
    // from 1,2,3 to 2,3,4
    this.dataTable.insertColumn(1, 'number', 'Rank');
  }

  preProcessDiffTable(
      dataTable: google.visualization.DataTable,
      diffTable: google.visualization.DataTable) {
    const sortColumn = [{column: 1, desc: false}];
    // The column hiding made the filter column index drift
    // from 1,2,3 to 0,1,2
    const hiddenColumn = [0, 4, 5, 6, 7, 10, 12];
    const formatDiffInfo = [
      {
        rangeMin: 0,
        rangeMax: 7,
        hasColor: false,
        isLargeBetter: false,
      },
      {
        rangeMin: 8,
        rangeMax: 12,
        hasColor: true,
        isLargeBetter: false,
      },
      {
        rangeMin: 13,
        hasColor: true,
        isLargeBetter: true,
      },
    ];
    const formatValueInfo = [
      {
        rangeMin: 0,
        rangeMax: 8,
        multiplier: 1,
        fixed: 0,
        suffix: '',
      },
      {
        rangeMin: 9,
        rangeMax: 12,
        multiplier: 100,
        fixed: 1,
        suffix: '%',
      },
      {
        rangeMin: 13,
        multiplier: 1,
        fixed: 1,
        suffix: '',
      },
    ];
    const dataView = computeDiffTable(
        /* oldTable= */ diffTable,
        /* newTable= */ dataTable,
        /* referenceCol= */ 3,
        /* comparisonCol= */ 8,
        /* addColumnType= */ 'number',
        /* addColumnLabel= */ 'Diff avg. self time',
        /* sortColumns= */ sortColumn,
        /* hiddenColumns= */ hiddenColumn,
        /* formatDiffInfo= */ formatDiffInfo,
        /* formatValueInfo= */ formatValueInfo);
    if (!dataView) {
      return;
    }

    this.dataTable = dataView.toDataTable();
    if (!this.dataTable) {
      return;
    }

    this.updateTotalOperations(this.dataTable);
  }

  formatDataTable(dataTable: google.visualization.DataTable):
      google.visualization.DataView {
    let sortColumn = this.sortColumn + 1;
    if (this.sortColumn === 0 ||
        this.sortColumn === DATA_VIEW_CUMULATIVE_DEVICE_PERCENT_INDEX ||
        this.sortColumn === DATA_VIEW_CUMULATIVE_HOST_PERCENT_INDEX) {
      sortColumn = this.sortColumn;
    }

    const sortedIndex = dataTable.getSortedRows({
      column: sortColumn,
      desc: !this.sortAscending,
    });
    let sumOfDevice = 0;
    let sumOfHost = 0;
    if (sortColumn === 0 && !this.sortAscending) {
      let n = sortedIndex.length;
      for (const v of sortedIndex) {
        dataTable.setCell(v, DATA_TABLE_RANK_INDEX, n--);
        sumOfDevice += dataTable.getValue(v, DATA_TABLE_DEVICE_PERCENT_INDEX);
        dataTable.setCell(
            v, DATA_TABLE_CUMULATIVE_DEVICE_PERCENT_INDEX, sumOfDevice);
        sumOfHost += dataTable.getValue(v, DATA_TABLE_HOST_PERCENT_INDEX);
        dataTable.setCell(
            v, DATA_TABLE_CUMULATIVE_HOST_PERCENT_INDEX, sumOfHost);
      }
    } else {
      for (const [index, v] of sortedIndex.entries()) {
        dataTable.setCell(v, DATA_TABLE_RANK_INDEX, index + 1);
        sumOfDevice += dataTable.getValue(v, DATA_TABLE_DEVICE_PERCENT_INDEX);
        if (sumOfDevice > 100) {
          sumOfDevice = 100;
        }
        dataTable.setCell(
            v, DATA_TABLE_CUMULATIVE_DEVICE_PERCENT_INDEX, sumOfDevice);
        sumOfHost += dataTable.getValue(v, DATA_TABLE_HOST_PERCENT_INDEX);
        if (sumOfHost > 100) {
          sumOfHost = 100;
        }
        dataTable.setCell(
            v, DATA_TABLE_CUMULATIVE_HOST_PERCENT_INDEX, sumOfHost);
      }
      if (!this.sortAscending &&
          (this.sortColumn === DATA_VIEW_CUMULATIVE_DEVICE_PERCENT_INDEX ||
           this.sortColumn === DATA_VIEW_CUMULATIVE_HOST_PERCENT_INDEX)) {
        let sumOfPercent =
            this.sortColumn === DATA_VIEW_CUMULATIVE_DEVICE_PERCENT_INDEX ?
            sumOfDevice :
            sumOfHost;
        const index =
            this.sortColumn === DATA_VIEW_CUMULATIVE_DEVICE_PERCENT_INDEX ?
            DATA_TABLE_CUMULATIVE_DEVICE_PERCENT_INDEX :
            DATA_TABLE_CUMULATIVE_HOST_PERCENT_INDEX;
        for (const v of sortedIndex) {
          dataTable.setCell(v, index, sumOfPercent);
          sumOfPercent -= dataTable.getValue(v, index - 1);
          if (sumOfPercent < 0) {
            sumOfPercent = 0;
          }
        }
      }
    }
    const percentFormatter =
        new google.visualization.NumberFormat({pattern: '##.#%'});
    percentFormatter.format(
        dataTable, DATA_TABLE_CUMULATIVE_DEVICE_PERCENT_INDEX);
    percentFormatter.format(
        dataTable, DATA_TABLE_CUMULATIVE_HOST_PERCENT_INDEX);

    const dataView = new google.visualization.DataView(dataTable);
    dataView.setRows(sortedIndex);

    dataView.hideColumns([0]);

    return dataView;
  }

  formatDiffTable(dataTable: google.visualization.DataTable):
      google.visualization.DataView {
    const dataView = new google.visualization.DataView(dataTable);
    if (this.sortColumn >= 0) {
      const sortedIndex = dataTable.getSortedRows({
        column: this.sortColumn,
        desc: !this.sortAscending,
      });
      dataView.setRows(sortedIndex);
    }
    return dataView;
  }

  updateTotalOperations(dataTable: google.visualization.DataTable) {
    const totalOperations = dataTable.getNumberOfRows();
    if (totalOperations > MAXIMUM_ROWS) {
      this.totalOperationsChanged.emit(String(totalOperations));
      dataTable.removeRows(MAXIMUM_ROWS, totalOperations - MAXIMUM_ROWS);
    } else {
      this.totalOperationsChanged.emit('');
    }
  }
}
