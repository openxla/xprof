import {ChangeDetectionStrategy, Component, Input} from '@angular/core';
import {MatDivider} from '@angular/material/divider';
import {Store} from '@ngrx/store';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {getKernelStatsDataState} from 'org_xprof/frontend/app/store/common_data_store/selectors';
import {ExportAsCsv} from '../controls/export_as_csv/export_as_csv';
import {KernelStatsTable} from './kernel_stats_table/kernel_stats_table';

/** A Kernel Stats component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'kernel-stats',
  templateUrl: './kernel_stats.ng.html',
  styleUrls: ['./kernel_stats.scss'],
  imports: [ExportAsCsv, KernelStatsTable, MatDivider],
})
export class KernelStats {
  data: SimpleDataTable | null = null;
  hasDataRow = false;
  @Input() sessionId = '';
  @Input() tool = '';
  @Input() host = '';

  constructor(store: Store<{}>) {
    store
      .select(getKernelStatsDataState)
      .subscribe((data: SimpleDataTable | null) => {
        this.update(data);
      });
  }

  update(data: SimpleDataTable | null) {
    this.data = data;
    this.hasDataRow = !!data && !!data.rows && data.rows.length > 0;
  }
}
