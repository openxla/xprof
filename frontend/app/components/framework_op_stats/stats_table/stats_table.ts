import {Component, Input, OnChanges, OnInit, SimpleChanges, ChangeDetectionStrategy} from '@angular/core';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {type FrameworkOpStatsData} from 'org_xprof/frontend/app/common/interfaces/data_table';

import {StatsTableDataProvider} from './stats_table_data_provider';

declare interface SortEvent {
  column: number;
  ascending: boolean;
}

const TABLE_COLUMN_LABEL_EXECUTOR = 'Host/device';
const TABLE_COLUMN_LABEL_TYPE = 'Type';
const TABLE_COLUMN_LABEL_OPERATION = 'Operation';

/** A stats table view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Eager,standalone: false,
  selector: 'stats-table',
  templateUrl: './stats_table.ng.html',
  styleUrls: ['./stats_table.scss']
})
export class StatsTable implements OnChanges, OnInit {
  /**
   * The tensorflow stats data.
   *  TODO(tf-profiler) rename to "frameworkOpStatsData"
   */
  @Input() tensorflowStatsData: FrameworkOpStatsData|null = null;

  /** The tensorflow stats data for diff. */
  @Input() diffData: FrameworkOpStatsData|null = null;

  /** Whether to use diff. */
  @Input() hasDiff = false;

  filterExecutor = '';
  filterType = '';
  filterOperation = '';
  totalOperations = '';
  dataProvider = new StatsTableDataProvider();
  dataInfo: ChartDataInfo = {
    data: null,
    dataProvider: this.dataProvider,
  };

  ngOnInit() {
    this.dataProvider.setTotalOperationsChangedEventListener(
        (totalOperations: string) => {
          this.totalOperations = totalOperations;
        });
  }

  ngOnChanges(changes: SimpleChanges) {
    this.dataProvider.hasDiff = this.hasDiff;
    if (this.hasDiff && this.diffData) {
      this.dataProvider.setDiffData(this.diffData);
    }
    this.dataInfo = {
      ...this.dataInfo,
      data: this.tensorflowStatsData,
    };
  }

  // Use label to choose the index due to lack of id
  getTableColumnIndex(columnLabel: string) {
    switch (columnLabel) {
      case TABLE_COLUMN_LABEL_EXECUTOR:
        return this.hasDiff && this.diffData ? 0 : 2;
      case TABLE_COLUMN_LABEL_TYPE:
        return this.hasDiff && this.diffData ? 1 : 3;
      case TABLE_COLUMN_LABEL_OPERATION:
        return this.hasDiff && this.diffData ? 2 : 4;
      default:
        return -1;
    }
  }

  updateFilters() {
    const filters: google.visualization.DataTableCellFilter[] = [];
    if (this.filterExecutor.trim()) {
      const filter = this.filterExecutor.trim().toLowerCase();
      filters.push({
        'column': this.getTableColumnIndex(TABLE_COLUMN_LABEL_EXECUTOR),
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
      });
    }
    if (this.filterType.trim()) {
      const filter = this.filterType.trim().toLowerCase();
      filters.push({
        'column': this.getTableColumnIndex(TABLE_COLUMN_LABEL_TYPE),
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
      });
    }
    if (this.filterOperation.trim()) {
      const filter = this.filterOperation.trim().toLowerCase();
      filters.push({
        'column': this.getTableColumnIndex(TABLE_COLUMN_LABEL_OPERATION),
        'test': (value: string) => value.toLowerCase().indexOf(filter) >= 0,
      });
    }

    this.dataProvider.setFilters(filters);
  }
}
