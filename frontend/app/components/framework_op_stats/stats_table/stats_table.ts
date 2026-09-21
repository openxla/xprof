import {
  ChangeDetectionStrategy,
  Component,
  OnInit,
  effect,
  input,
} from '@angular/core';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {type FrameworkOpStatsData} from 'org_xprof/frontend/app/common/interfaces/data_table';

import {MatFormField, MatLabel, MatSuffix} from '@angular/material/form-field';
import {MatIcon} from '@angular/material/icon';
import {MatInput} from '@angular/material/input';
import {Chart} from '../../chart/chart';
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
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'stats-table',
  templateUrl: './stats_table.ng.html',
  styleUrls: ['./stats_table.scss'],
  imports: [Chart, MatFormField, MatIcon, MatInput, MatLabel, MatSuffix],
})
export class StatsTable implements OnInit {
  /**
   * The tensorflow stats data.
   *  TODO(tf-profiler) rename to "frameworkOpStatsData"
   */
  readonly tensorflowStatsData = input<FrameworkOpStatsData | null>(null);

  /** The tensorflow stats data for diff. */
  readonly diffData = input<FrameworkOpStatsData | null>(null);

  /** Whether to use diff. */
  readonly hasDiff = input(false);

  filterExecutor = '';
  filterType = '';
  filterOperation = '';
  totalOperations = '';
  dataProvider = new StatsTableDataProvider();
  dataInfo: ChartDataInfo = {
    data: null,
    dataProvider: this.dataProvider,
  };

  constructor() {
    effect(() => {
      const hasDiff = this.hasDiff();
      const diffData = this.diffData();
      const tensorflowStatsData = this.tensorflowStatsData();

      this.dataProvider.hasDiff = hasDiff;
      if (hasDiff && diffData) {
        this.dataProvider.setDiffData(diffData);
      }
      this.dataInfo = {
        ...this.dataInfo,
        data: tensorflowStatsData,
      };
    });
  }

  ngOnInit() {
    this.dataProvider.setTotalOperationsChangedEventListener(
      (totalOperations: string) => {
        this.totalOperations = totalOperations;
      },
    );
  }

  // Use label to choose the index due to lack of id
  getTableColumnIndex(columnLabel: string) {
    switch (columnLabel) {
      case TABLE_COLUMN_LABEL_EXECUTOR:
        return this.hasDiff() && this.diffData() ? 0 : 2;
      case TABLE_COLUMN_LABEL_TYPE:
        return this.hasDiff() && this.diffData() ? 1 : 3;
      case TABLE_COLUMN_LABEL_OPERATION:
        return this.hasDiff() && this.diffData() ? 2 : 4;
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
