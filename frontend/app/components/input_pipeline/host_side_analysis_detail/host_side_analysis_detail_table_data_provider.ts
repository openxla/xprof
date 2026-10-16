import {EventEmitter} from '@angular/core';
import {
  HostOpsColumn,
  SimpleDataTable,
} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';

/** A host side analysis detail table data provider. */
export class HostSideAnalysisDetailTableDataProvider extends DefaultDataProvider {
  private readonly hasHostOpsChanged = new EventEmitter<boolean>();

  override parseData(
    data: SimpleDataTable | Array<Array<string | number>> | null,
  ) {
    if (!data) return;
    const dataTable = new google.visualization.DataTable(data);
    if (dataTable.getNumberOfRows() < 1) {
      this.hasHostOpsChanged.emit(false);
      return;
    }
    this.hasHostOpsChanged.emit(true);

    const columns: HostOpsColumn = {
      opName: 0,
      count: 0,
      timeInMs: 0,
      timeInPercent: 0,
      selfTimeInMs: 0,
      selfTimeInPercent: 0,
      category: 0,
    };
    for (let i = 0; i < dataTable.getNumberOfColumns(); i++) {
      switch (dataTable.getColumnId(i)) {
        case 'opName':
          columns.opName = i;
          break;
        case 'count':
          columns.count = i;
          break;
        case 'timeInMs':
          columns.timeInMs = i;
          break;
        case 'timeInPercent':
          columns.timeInPercent = i;
          break;
        case 'selfTimeInMs':
          columns.selfTimeInMs = i;
          break;
        case 'selfTimeInPercent':
          columns.selfTimeInPercent = i;
          break;
        case 'category':
          columns.category = i;
          break;
        default:
          break;
      }
    }

    const percentFormatter = new google.visualization.NumberFormat({
      pattern: '##.#%',
    });
    percentFormatter.format(dataTable, columns.timeInPercent);
    percentFormatter.format(dataTable, columns.selfTimeInPercent);

    const zeroDecimalPtFormatter = new google.visualization.NumberFormat({
      'fractionDigits': 0,
    });
    zeroDecimalPtFormatter.format(dataTable, columns.timeInMs);
    zeroDecimalPtFormatter.format(dataTable, columns.selfTimeInMs);

    dataTable.setProperty(0, columns.opName, 'style', 'width: 40%');
    dataTable.setProperty(0, columns.count, 'style', 'width: 15%');
    dataTable.setProperty(0, columns.timeInMs, 'style', 'width: 10%');
    dataTable.setProperty(0, columns.timeInPercent, 'style', 'width: 5%');
    dataTable.setProperty(0, columns.selfTimeInMs, 'style', 'width: 10%');
    dataTable.setProperty(0, columns.selfTimeInPercent, 'style', 'width: 5%');
    dataTable.setProperty(0, columns.category, 'style', 'width: 15%');

    this.dataTable = dataTable;
  }

  setHasHostOpsChangedEventListener(callback: Function) {
    this.hasHostOpsChanged.subscribe(callback);
  }
}
