import {ChangeDetectionStrategy, Component, effect, input} from '@angular/core';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {
  type FrameworkOpStatsData,
  type SimpleDataTable,
} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {COLUMN_CHART_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {XyTableDataProcessor} from 'org_xprof/frontend/app/components/chart/xy_table_data_processor';
import {Chart} from '../../chart/chart';

/** A flop rate chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'flop-rate-chart',
  templateUrl: './flop_rate_chart.ng.html',
  styleUrls: ['./flop_rate_chart.scss'],
  imports: [Chart],
})
export class FlopRateChart {
  /** The input data. */
  readonly data = input<SimpleDataTable | FrameworkOpStatsData | null>(null);
  /** Index of the Column that corresponds to the x-axis. */
  readonly xColumn = input(0);
  /** Index of the Column that corresponds to the y-axis. */
  readonly yColumn = input(0);
  /** The type of the OP, e.g. TensorFlow. */
  readonly opType = input('');

  dataInfo: ChartDataInfo = {
    data: null,
    dataProvider: new DefaultDataProvider(),
  };

  constructor() {
    effect(() => {
      this.update();
    });
  }

  update() {
    const data = this.data();
    const xColumn = this.xColumn();
    const yColumn = this.yColumn();
    const opType = this.opType();

    this.dataInfo.customChartDataProcessor = new XyTableDataProcessor(
      {'fractionDigits': 1},
      [{column: yColumn, minValue: 0.0}],
      xColumn,
      yColumn,
    );
    this.dataInfo.options = {
      ...COLUMN_CHART_OPTIONS,
      hAxis: {
        textPosition: 'none',
        title: opType + ' Op on Device (in decreasing total self-time)',
      },
      vAxis: {title: 'GFLOPs/sec'},
    };
    this.dataInfo = {
      ...this.dataInfo,
      data,
    };
  }
}
