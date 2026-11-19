import {Component, Input, OnChanges, SimpleChanges, ChangeDetectionStrategy} from '@angular/core';
import {OpExecutor} from 'org_xprof/frontend/app/common/constants/enums';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {type FrameworkOpStatsData, type SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {COLUMN_CHART_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {XyTableDataProcessor} from 'org_xprof/frontend/app/components/chart/xy_table_data_processor';

/** A flop rate chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'flop-rate-chart',
  templateUrl: './flop_rate_chart.ng.html',
  styleUrls: ['./flop_rate_chart.scss']
})
export class FlopRateChart implements OnChanges {
  /** The input data. */
  @Input() data: SimpleDataTable|FrameworkOpStatsData|null = null;
  /** Index of the Column that corresponds to the x-axis. */
  @Input() xColumn = 0;
  /** Index of the Column that corresponds to the y-axis. */
  @Input() yColumn = 0;
  /** The type of the OP, e.g. TensorFlow. */
  @Input() opType = '';

  dataInfo: ChartDataInfo = {
    data: null,
    dataProvider: new DefaultDataProvider(),
  };

  ngOnChanges(changes: SimpleChanges) {
    const filters: google.visualization.DataTableCellFilter[] = [
      {column: this.yColumn, minValue: Number.MIN_VALUE},
    ];
    const opExecutorColumn =
        this.data?.cols?.findIndex((col) => col.id === 'host_or_device') ?? -1;
    if (opExecutorColumn >= 0) {
      filters.unshift({column: opExecutorColumn, value: OpExecutor.DEVICE});
    }
    this.dataInfo.customChartDataProcessor = new XyTableDataProcessor(
        {'fractionDigits': 1}, filters,
        this.xColumn, this.yColumn);
    this.dataInfo.options = {
      ...COLUMN_CHART_OPTIONS,
      hAxis: {
        textPosition: 'none',
        title: this.opType + ' Op on Device (in decreasing total self-time)',
      },
      vAxis: {title: 'GFLOPs/sec'},
    };
    this.dataInfo = {
      ...this.dataInfo,
      data: this.data,
    };
  }
}
