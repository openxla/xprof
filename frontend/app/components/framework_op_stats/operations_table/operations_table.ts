import {ChangeDetectionStrategy, Component, effect, input} from '@angular/core';
import {OpExecutor} from 'org_xprof/frontend/app/common/constants/enums';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {type FrameworkOpStatsData} from 'org_xprof/frontend/app/common/interfaces/data_table';

import {Chart} from '../../chart/chart';
import {OperationsTableDataProvider} from './operations_table_data_provider';

/** An operations table view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'operations-table',
  templateUrl: './operations_table.ng.html',
  styleUrls: ['./operations_table.scss'],
  imports: [Chart],
})
export class OperationsTable {
  /**
   * The tensorflow stats data.
   *  TODO(tf-profiler) rename to "frameworkOpStatsData"
   */
  readonly tensorflowStatsData = input<FrameworkOpStatsData | null>(null);

  /** The tensorflow stats data for diff. */
  readonly diffData = input<FrameworkOpStatsData | null>(null);

  /** Whether to use diff. */
  readonly hasDiff = input(false);

  /** The Op executor. */
  readonly opExecutor = input(OpExecutor.NONE);

  title = '';
  dataProvider = new OperationsTableDataProvider();
  dataInfo: ChartDataInfo = {
    data: null,
    dataProvider: this.dataProvider,
  };

  constructor() {
    effect(() => {
      this.update();
    });
  }

  update() {
    const opExecutor = this.opExecutor();
    const hasDiff = this.hasDiff();
    const diffData = this.diffData();
    const tensorflowStatsData = this.tensorflowStatsData();

    if (opExecutor === OpExecutor.DEVICE) {
      this.title = 'Device-side TensorFlow operations (grouped by TYPE)';
    } else if (opExecutor === OpExecutor.HOST) {
      this.title = 'Host-side TensorFlow operations (grouped by TYPE)';
    } else {
      this.title = '';
    }
    this.dataProvider.hasDiff = hasDiff;
    if (hasDiff && diffData) {
      this.dataProvider.setDiffData(diffData);
    }
    this.dataProvider.opExecutor = opExecutor;
    this.dataInfo = {
      ...this.dataInfo,
      data: tensorflowStatsData,
    };
  }
}
