import {ChartDataProvider, CustomChartDataProcessor, DataTableOrDataView} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';

import {computeCategoryDiffTable, computeGroupView} from './table_utils';

/** A category diff table data processor. */
export class CategoryDiffTableDataProcessor implements
    CustomChartDataProcessor {
  constructor(
      private readonly diffData: SimpleDataTable|null,
      private readonly filters: google.visualization.DataTableCellFilter[],
      private readonly gColumn: number, private readonly vColumn: number) {}

  process(dataProvider: ChartDataProvider): DataTableOrDataView|null {
    if (!dataProvider) {
      return null;
    }

    const dataTable = dataProvider.getDataTable();
    if (!dataTable) {
      return null;
    }

    if (
      !this.diffData ||
      !this.diffData.cols ||
      this.diffData.cols.length === 0
    ) {
      return null;
    }

    const diffTable = new google.visualization.DataTable(this.diffData);
    if (!diffTable) {
      return null;
    }

    const chart = dataProvider.getChart();
    if (!chart) {
      return null;
    }

    if (this.gColumn < 0 || this.vColumn < 0) {
      return null;
    }

    const maxColumn = Math.max(this.gColumn, this.vColumn);
    const dataNumCols = dataTable.getNumberOfColumns();
    const diffNumCols = diffTable.getNumberOfColumns();
    if (dataNumCols <= maxColumn || diffNumCols <= maxColumn) {
      return null;
    }

    const filters = this.filters || [];
    const validDataFilters = filters.filter(
      (filter) => filter.column >= 0 && filter.column < dataNumCols,
    );
    const validDiffFilters = filters.filter(
      (filter) => filter.column >= 0 && filter.column < diffNumCols,
    );

    return computeCategoryDiffTable(
      computeGroupView(dataTable, validDataFilters, this.gColumn, this.vColumn),
      computeGroupView(diffTable, validDiffFilters, this.gColumn, this.vColumn),
      chart,
    );
  }
}
