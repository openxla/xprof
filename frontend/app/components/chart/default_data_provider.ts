import {EventEmitter} from '@angular/core';
import {
  ChartClass,
  ChartDataProvider,
  ChartOptions,
  DataTableOrDataView,
} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A default chart data provider. */
export class DefaultDataProvider implements ChartDataProvider {
  protected chart?: ChartClass;
  protected dataTable?: google.visualization.DataTable;
  protected sortColumns?: google.visualization.SortByColumn[];
  protected visibleColumns?: number[];
  protected filters?: google.visualization.DataTableCellFilter[];
  protected readonly update = new EventEmitter();

  setChart(chart: ChartClass) {
    this.chart = chart;
  }

  parseData(data: SimpleDataTable | Array<Array<string | number>> | null) {
    if (data && typeof google !== 'undefined' && google.visualization) {
      this.dataTable = new google.visualization.DataTable(
        data as SimpleDataTable,
      );
    }
  }

  setSortColumns(sortColumns: google.visualization.SortByColumn[]) {
    this.sortColumns = sortColumns;
  }

  setVisibleColumns(visibleColumns: number[]) {
    this.visibleColumns = visibleColumns;
  }

  setFilters(filters: google.visualization.DataTableCellFilter[]) {
    this.filters = filters;
  }

  process(): DataTableOrDataView | null {
    if (!this.dataTable) {
      return null;
    }

    if (this.sortColumns && this.sortColumns.length > 0) {
      this.dataTable.sort(this.sortColumns);
    }

    const dataView = new google.visualization.DataView(this.dataTable);

    if (this.visibleColumns && this.visibleColumns.length > 0) {
      dataView.setColumns(this.visibleColumns);
    }

    if (this.filters && this.filters.length > 0) {
      dataView.setRows(this.dataTable.getFilteredRows(this.filters));
    }

    return dataView;
  }

  getChart(): ChartClass | null {
    return this.chart || null;
  }

  getDataTable(): google.visualization.DataTable | null {
    return this.dataTable ? this.dataTable : null;
  }

  getOptions(): ChartOptions | null {
    return null;
  }

  setUpdateEventListener(callback: Function) {
    this.update.subscribe(callback);
  }

  notifyCharts() {
    this.update.emit();
  }
}

/** A chart data provider that accepts array data. */
export class ArrayDataProvider extends DefaultDataProvider {
  override parseData(
    data: SimpleDataTable | Array<Array<string | number>> | null,
  ) {
    if (data) {
      /* tslint:disable no-any */
      this.dataTable = google.visualization.arrayToDataTable(data as any[]);
      /* tslint:enable */
    }
  }
}

/**
 * A chart data provider that accepts SimpleDataTable and allows computing
 * metrics grouped by replica groups.
 */
export class ReplicaGroupDataProvider extends DefaultDataProvider {
  // Communication HLO ops are ops who have the field 'replica_groups'.
  // Ex. all-reduce, all-gather, etc
  communicationOps = new Set();
  // Column indexes
  opCategoryIndex?: number; // 'category' column
  hloOpNameIndex?: number; // 'hlo_op_expression' column
  selfTimeIndex?: number; // 'total_self_time' column

  override parseData(data: SimpleDataTable) {
    const rowWithReplicaGroups: google.visualization.DataObjectRow[] = [];

    if (!data || !data.cols || !data.rows) return;

    // Set the column index member variables.
    for (let i = 0; i < data.cols.length; i++) {
      // TODO(xprof) ids defined in hlo_stats.cc is also hard coded here and in
      // hlo_stats.ts, try to decouple or add "if change then" lint
      if (data.cols[i].id === 'category') this.opCategoryIndex = i;
      if (data.cols[i].id === 'hlo_op_expression') this.hloOpNameIndex = i;
      if (data.cols[i].id === 'total_self_time') this.selfTimeIndex = i;
    }

    if (
      this.opCategoryIndex === undefined ||
      this.hloOpNameIndex === undefined ||
      this.selfTimeIndex === undefined
    ) {
      return;
    }

    // Filter through rows whose hloOpName string has the field
    // 'replica_groups'. Set the value for the hloOpName to only be the
    // 'replica_groups={{...}}' string. This allows computing metrics grouped by
    // replica groups.
    for (const row of data.rows) {
      if (row?.c) {
        const hloOpName = row.c[this.hloOpNameIndex]?.v;

        if (typeof hloOpName !== 'string') return;

        const hasReplicaGroup = hloOpName.match(
          /replica_groups={({(\d,?)+},?)*}/,
        );

        if (hasReplicaGroup !== null) {
          const newRow = {c: [...row.c]};
          newRow.c[this.hloOpNameIndex] = {v: hasReplicaGroup[0]};
          rowWithReplicaGroups.push(newRow);
        }
      }
    }

    // Create a set of the categories of communication HLO ops.
    this.communicationOps.clear();
    for (const row of rowWithReplicaGroups) {
      if (row?.c) {
        this.communicationOps.add(row.c[this.opCategoryIndex].v);
      }
    }

    this.dataTable = new google.visualization.DataTable({
      cols: data.cols,
      rows: rowWithReplicaGroups,
      p: data.p,
    });
  }
}
