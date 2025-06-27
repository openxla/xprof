import {Component, Input, OnChanges, SimpleChanges} from '@angular/core';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {OpExecutor} from 'org_xprof/frontend/app/common/constants/enums';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {type FrameworkOpStatsData} from 'org_xprof/frontend/app/common/interfaces/data_table';

import {OperationsTableDataProvider} from './operations_table_data_provider';

/** An operations table view component. */
@Component({
  standalone: true,
  selector: 'operations-table',
  templateUrl: 'operations_table.ng.html',
  styleUrls: ['operations_table.scss'],
  imports: [ChartModule],
})
export class OperationsTable implements OnChanges {
  /**
   * The tensorflow stats data.
   *  TODO(tf-profiler) rename to "frameworkOpStatsData"
   */
  @Input() tensorflowStatsData: FrameworkOpStatsData|null = null;

  /** The tensorflow stats data for diff. */
  @Input() diffData: FrameworkOpStatsData|null = null;

  /** Whether to use diff. */
  @Input() hasDiff = false;

  /** The Op executor. */
  @Input() opExecutor: OpExecutor = OpExecutor.NONE;

  title = '';
  dataProvider = new OperationsTableDataProvider();
  dataInfo: ChartDataInfo = {
    data: null,
    dataProvider: this.dataProvider,
  };

  ngOnChanges(changes: SimpleChanges) {
    if (this.opExecutor === OpExecutor.DEVICE) {
      this.title = 'Device-side TensorFlow operations (grouped by TYPE)';
    } else if (this.opExecutor === OpExecutor.HOST) {
      this.title = 'Host-side TensorFlow operations (grouped by TYPE)';
    } else {
      this.title = '';
    }
    this.dataProvider.hasDiff = this.hasDiff;
    if (this.hasDiff && this.diffData) {
      this.dataProvider.setDiffData(this.diffData);
    }
    this.dataProvider.opExecutor = this.opExecutor;
    this.dataInfo = {
      ...this.dataInfo,
      data: this.tensorflowStatsData,
    };
  }
}
