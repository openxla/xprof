import {OpExecutor} from 'org_xprof/frontend/app/common/constants/enums';
import {ChartOptions} from 'org_xprof/frontend/app/common/interfaces/chart';
import {
  FrameworkOpStatsData,
  SimpleDataTable,
} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {computeDiffTable} from 'org_xprof/frontend/app/components/chart/table_utils';

const MINIMUM_ROWS = 20;

/** A operations table data provider. */
export class OperationsTableDataProvider extends DefaultDataProvider {
  diffTable?: google.visualization.DataTable;
  private rawDiffData: FrameworkOpStatsData | null = null;
  hasDiff = false;
  opExecutor = OpExecutor.NONE;
  options = {
    allowHtml: true,
    alternatingRowStyle: false,
    showRowNumber: false,
    height: '600px',
    cssClassNames: {
      'headerCell': 'google-chart-table-header-cell',
      'tableCell': 'google-chart-table-table-cell',
    },
  };

  override parseData(
      data: SimpleDataTable | Array<Array<string | number>> | null,
  ) {
    if (!data) {
      this.pendingData = null;
      this.dataTable = undefined;
      return;
    }
    if (!this.ensureChartPackagesLoaded(data)) return;
    this.dataTable =
        new google.visualization.DataTable(data as SimpleDataTable);
    this.diffTable = this.rawDiffData
        ? new google.visualization.DataTable(this.rawDiffData)
        : undefined;
  }

  setDiffData(diffData: FrameworkOpStatsData|null) {
    this.rawDiffData = diffData;
    if (diffData && this.isDataTableReady()) {
      this.diffTable = new google.visualization.DataTable(diffData);
    } else {
      this.diffTable = undefined;
    }
  }

  override process(): google.visualization.DataTable|google.visualization.DataView|null {
    if (!this.dataTable ||
        (this.opExecutor !== OpExecutor.DEVICE &&
         this.opExecutor !== OpExecutor.HOST)) {
      return null;
    }

    let dataView: google.visualization.DataView|null = null;

    if (!this.hasDiff || !this.diffTable) {
      dataView = this.makeDataView(this.dataTable, this.opExecutor);
    } else {
      const oldDataView = this.makeDataView(this.diffTable, this.opExecutor);
      const newDataView = this.makeDataView(this.dataTable, this.opExecutor);
      if (!oldDataView || !newDataView) {
        return null;
      }
      const formatDiffInfo = [
        {
          rangeMin: 1,
          rangeMax: 1,
          hasColor: false,
          isLargeBetter: false,
        },
        {
          rangeMin: 0,
          rangeMax: 0,
          hasColor: true,
          isLargeBetter: false,
        },
        {
          rangeMin: 2,
          hasColor: true,
          isLargeBetter: false,
        },
      ];
      const formatValueInfo = [
        {
          rangeMin: 1,
          rangeMax: 1,
          multiplier: 1,
          fixed: 0,
          suffix: '',
        },
        {
          rangeMin: 2,
          rangeMax: 2,
          multiplier: 1,
          fixed: 3,
          suffix: '',
        },
        {
          rangeMin: 0,
          rangeMax: 0,
          multiplier: 100,
          fixed: 1,
          suffix: '%',
        },
        {
          rangeMin: 3,
          multiplier: 100,
          fixed: 1,
          suffix: '%',
        },
      ];
      dataView = computeDiffTable(
          /* oldTable= */ oldDataView.toDataTable(),
          /* newTable= */ newDataView.toDataTable(),
          /* referenceCol= */ 0,
          /* comparisonCol= */ 3,
          /* addColumnType= */ 'number',
          /* addColumnLabel= */ 'Diff total self time',
          /* sortColumns= */[],
          /* hiddenColumns= */[],
          /* formatDiffInfo= */ formatDiffInfo,
          /* formatValueInfo= */ formatValueInfo);
    }

    if (!dataView) {
      return null;
    }

    this.options.height =
        dataView.getNumberOfRows() < MINIMUM_ROWS ? '' : '600px';

    return dataView;
  }

  override getOptions(): ChartOptions|null {
    return this.options;
  }

  makeDataView(
      srcTable: google.visualization.DataTable, deviceType: OpExecutor) {
    return deviceType === OpExecutor.DEVICE ?
        this.makeDataViewForDevice(srcTable) :
        this.makeDataViewForHost(srcTable);
  }

  makeDataViewForDevice(srcTable: google.visualization.DataTable):
      google.visualization.DataView|null {
    if (!srcTable || this.opExecutor !== OpExecutor.DEVICE) {
      return null;
    }

    let dataView = new google.visualization.DataView(srcTable);
    dataView.setRows(dataView.getFilteredRows([{
      'column': 1,
      'value': 'Device',
    }]));

    /**
     * column 0: type
     * column 1: occurrences
     * column 2: self_time
     * column 3: self_time_on_device
     */
    dataView.setColumns([
      2,
      4,
      7,
      9,
    ]);

    /**
     * column 0: type
     * column 1: sum of occurrences
     * column 2: sum of self_time
     * column 3: sum of self_time_on_device
     */
    const dataTable = google.visualization.data.group(dataView, [0], [
      {
        'column': 1,
        'aggregation': google.visualization.data.sum,
        'type': 'number',
      },
      {
        'column': 2,
        'aggregation': google.visualization.data.sum,
        'type': 'number',
      },
      {
        'column': 3,
        'aggregation': google.visualization.data.sum,
        'type': 'number',
      },
    ]);
    dataTable.sort({
      'column': 2,
      'desc': true,
    });
    const decimalFormatter =
        new google.visualization.NumberFormat({'fractionDigits': 0});
    decimalFormatter.format(dataTable, 2);
    const percentFormatter =
        new google.visualization.NumberFormat({pattern: '##.#%'});
    percentFormatter.format(dataTable, 3);

    dataView = new google.visualization.DataView(dataTable);
    /**
     * column 0: type
     * column 1: sum of occurrences
     * column 2: sum of self_time
     * column 3: sum of self_time_on_device
     */
    dataView.setColumns([
      0,
      1,
      2,
      3,
    ]);

    return dataView;
  }

  makeDataViewForHost(srcTable: google.visualization.DataTable):
      google.visualization.DataView|null {
    if (!srcTable || this.opExecutor !== OpExecutor.HOST) {
      return null;
    }

    const dataView = new google.visualization.DataView(srcTable);
    dataView.setRows(dataView.getFilteredRows([{
      'column': 1,
      'value': 'Host',
    }]));

    /**
     * column 0: type
     * column 1: occurrences
     * column 2: self_time
     * column 3: self_time_on_host
     */
    dataView.setColumns([2, 4, 7, 11]);

    /**
     * column 0: type
     * column 1: sum of occurrences
     * column 2: sum of self_time
     * column 3: sum of self_time_on_host
     */
    const dataTable = google.visualization.data.group(dataView, [0], [
      {
        'column': 1,
        'aggregation': google.visualization.data.sum,
        'type': 'number',
      },
      {
        'column': 2,
        'aggregation': google.visualization.data.sum,
        'type': 'number',
      },
      {
        'column': 3,
        'aggregation': google.visualization.data.sum,
        'type': 'number',
      },
    ]);
    dataTable.sort({
      'column': 2,
      'desc': true,
    });
    const decimalFormatter =
        new google.visualization.NumberFormat({'fractionDigits': 0});
    decimalFormatter.format(dataTable, 2);
    const percentFormatter =
        new google.visualization.NumberFormat({pattern: '##.#%'});
    percentFormatter.format(dataTable, 3);

    return new google.visualization.DataView(dataTable);
  }
}
