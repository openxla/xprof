import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {isGoogleChartsLoaded} from 'org_xprof/frontend/app/common/utils/utils';

/** A base class for a google-charts dashboard. */
export class Dashboard {
  dataTable?: google.visualization.DataTable;

  // Columns from dataTable visible in dataView.
  // If empty, all columns from dataTable are shown.
  columns: Array<number|google.visualization.ColumnSpec> = [];

  // Filters for selecting rows from dataTable to show in dataView.
  // If empty, all rows from dataTable are shown.
  filters = new Map<number, google.visualization.DataTableCellFilter>();

  dataView?: google.visualization.DataView;
  protected pendingData: SimpleDataTable | null = null;
  protected hasPendingLoadCallback = false;

  protected ensureChartPackagesLoaded(data: SimpleDataTable|null): boolean {
    if (
      typeof google !== 'undefined' &&
      typeof google.visualization?.DataTable === 'function'
    ) {
      this.pendingData = null;
      return true;
    }
    this.pendingData = data;
    if (!this.hasPendingLoadCallback) {
      this.hasPendingLoadCallback = true;
      this.loadPendingData();
    }
    return false;
  }

  /**
   * Parses `pendingData` once the Google Charts packages load.
   *
   * Polls while the loader script has not defined `google.charts`, which
   * happens when the app boots first on slow networks. Stops once
   * `pendingData` is cleared, such as when the component is destroyed.
   */
  private loadPendingData() {
    if (!this.pendingData) {
      this.hasPendingLoadCallback = false;
      return;
    }
    if (!isGoogleChartsLoaded()) {
      setTimeout(() => {
        this.loadPendingData();
      }, 100);
      return;
    }
    google.charts.safeLoad({'packages': ['corechart', 'table']});
    google.charts.setOnLoadCallback(() => {
      this.hasPendingLoadCallback = false;
      const latest = this.pendingData;
      this.pendingData = null;
      if (latest) {
        this.parseData(latest);
      }
    });
  }

  parseData(data: SimpleDataTable|null) {
    if (!data) {
      this.pendingData = null;
      this.dataTable = undefined;
      this.dataView = undefined;
      return;
    }
    if (!this.ensureChartPackagesLoaded(data)) return;

    this.dataTable = new google.visualization.DataTable(data);
    this.updateView();
  }

  updateFilters(filter: google.visualization.DataTableCellFilter) {
    if (filter.column === -1) {
      return;
    }
    if (('value' in filter && filter.value !== undefined) ||
        ('minValue' in filter && filter.minValue !== undefined) ||
        ('maxValue' in filter && filter.maxValue !== undefined) ||
        'test' in filter) {
      this.filters.set(filter.column, filter);
    } else {
      this.filters.delete(filter.column);
    }
    this.updateView();
  }

  updateView() {
    if (!this.dataTable) {
      return;
    }
    const dataView = new google.visualization.DataView(this.dataTable);
    if (this.columns.length !== 0) {
      dataView.setColumns(this.columns);
    }
    if (this.filters.size !== 0) {
      dataView.setRows(this.dataTable.getFilteredRows(this.getFilters()));
    }
    this.dataView = dataView;
  }

  getFilters(): google.visualization.DataTableCellFilter[] {
    return Array.from(this.filters.values());
  }

  // Refresh on the dashboard is necessary when the dataTable changes with brand
  // new schema.
  resetDashboard() {
    this.dataTable = undefined;
    this.filters.clear();
    this.columns = [];
  }
}
