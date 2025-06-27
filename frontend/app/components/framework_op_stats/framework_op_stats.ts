import {CommonModule} from '@angular/common';
import {Component, Input} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatDividerModule} from '@angular/material/divider';
import {MatIconModule} from '@angular/material/icon';
import {MatMenuModule} from '@angular/material/menu';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {ExportAsCsv} from 'org_xprof/frontend/app/components/controls/export_as_csv/export_as_csv';
import {FlopRateChart} from 'org_xprof/frontend/app/components/framework_op_stats/flop_rate_chart/flop_rate_chart';
import {ModelProperties} from 'org_xprof/frontend/app/components/framework_op_stats/model_properties/model_properties';
import {OperationsTable} from 'org_xprof/frontend/app/components/framework_op_stats/operations_table/operations_table';
import {StatsTable} from 'org_xprof/frontend/app/components/framework_op_stats/stats_table/stats_table';
import {Store} from '@ngrx/store';
import {IdleOption, OpExecutor, OpKind, OpType} from 'org_xprof/frontend/app/common/constants/enums';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {FrameworkOpStatsData} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {CategoryDiffTableDataProcessor} from 'org_xprof/frontend/app/components/chart/category_diff_table_data_processor';
import {CategoryTableDataProcessor} from 'org_xprof/frontend/app/components/chart/category_table_data_processor';
import {PIE_CHART_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import * as selectors from 'org_xprof/frontend/app/store/framework_op_stats/selectors';

const OP_EXECUTOR_ID = 'host_or_device';
const OP_TYPE_ID = 'type';
const OP_NAME_ID = 'operation';
const SELF_TIME_ID = 'total_self_time';
const MEASURED_FLOP_RATE_ID = 'measured_flop_rate';

/** A TensorFlow Stats component. */
@Component({
  standalone: true,
  selector: 'framework-op-stats',
  templateUrl: 'framework_op_stats.ng.html',
  styleUrls: ['framework_op_stats.css'],
  imports: [
    CommonModule,
    MatButtonModule,
    MatDividerModule,
    MatIconModule,
    MatMenuModule,
    ChartModule,
    ExportAsCsv,
    FlopRateChart,
    ModelProperties,
    OperationsTable,
    StatsTable,
  ],
})
export class FrameworkOpStats {
  @Input() sessionId = '';
  @Input() tool = '';
  @Input() host = '';
  data: FrameworkOpStatsData[]|null = null;
  diffData: FrameworkOpStatsData[]|null = null;
  selectedData: FrameworkOpStatsData|null = null;
  selectedDiffData: FrameworkOpStatsData|null = null;
  idleMenuButtonLabel = IdleOption.NO;
  idleOptionItems = [IdleOption.YES, IdleOption.NO];
  opExecutorDevice = OpExecutor.DEVICE;
  opExecutorHost = OpExecutor.HOST;
  opKindName = OpKind.NAME;
  opKindType = OpKind.TYPE;
  opType = OpType.TENSORFLOW;
  title = '';
  architecture = '';
  task = '';
  devicePprofLink = '';
  hostPprofLink = '';
  hasDeviceData = false;
  hasDiff = false;
  showFlopRateChart = false;
  flopRateChartXColumn = -1;
  flopRateChartYColumn = -1;
  showModelProperties = false;
  showPprofLink = false;
  dataProvider = new DefaultDataProvider();
  dataInfoDeviceByType: ChartDataInfo = {
    data: null,
    dataProvider: this.dataProvider,
    options: PIE_CHART_OPTIONS,
  };
  dataInfoDeviceByName: ChartDataInfo = {
    data: null,
    dataProvider: this.dataProvider,
    options: PIE_CHART_OPTIONS,
  };
  dataInfoHostByType: ChartDataInfo = {
    data: null,
    dataProvider: this.dataProvider,
    options: PIE_CHART_OPTIONS,
  };
  dataInfoHostByName: ChartDataInfo = {
    data: null,
    dataProvider: this.dataProvider,
    options: PIE_CHART_OPTIONS,
  };

  constructor(private readonly store: Store<{}>) {
    this.store.select(selectors.getTitleState).subscribe((title: string) => {
      this.title = title || '';
    });
    this.store.select(selectors.getHasDiffState).subscribe(hasDiff => {
      this.hasDiff = Boolean(hasDiff);
    });
    this.store.select(selectors.getShowFlopRateChartState)
        .subscribe(showFlopRateChart => {
          this.showFlopRateChart = Boolean(showFlopRateChart);
        });
    this.store.select(selectors.getShowModelPropertiesState)
        .subscribe(showModelProperties => {
          this.showModelProperties = Boolean(showModelProperties);
        });
    this.store.select(selectors.getShowPprofLinkState)
        .subscribe(showPprofLink => {
          this.showPprofLink = Boolean(showPprofLink);
        });
    this.store.select(selectors.getDiffDataState)
        .subscribe((diffData: FrameworkOpStatsData[]) => {
          this.diffData = (diffData || []);
          this.setIdleOption();
        });
    this.store.select(selectors.getDataState)
        .subscribe((data: FrameworkOpStatsData[]) => {
          this.data = (data || []);
          this.setIdleOption();
        });
  }

  setIdleOption(option: IdleOption = IdleOption.NO) {
    this.idleMenuButtonLabel = option;
    if ((!this.data || this.data.length === 0) ||
        (this.hasDiff && (!this.diffData || this.diffData.length === 0))) {
      this.selectedData = null;
      this.selectedDiffData = null;
      return;
    }

    if (option === IdleOption.YES) {
      this.selectedData = this.data[0] || null;
      if (this.hasDiff && this.diffData) {
        this.selectedDiffData = this.diffData[0] || null;
      }
    } else {
      this.selectedData = this.data[1] || null;
      if (this.hasDiff && this.diffData) {
        this.selectedDiffData = this.diffData[1] || null;
      }
    }

    // Four charts share one DataProvider. In order to prevent DataTable from
    // being created 4 times, it calls DataProvider function directly.
    this.dataProvider.parseData(this.selectedData);

    const dataTable = this.dataProvider.getDataTable();
    let opTypeIndex = -1;
    let opNameIndex = -1;
    let selfTimeIndex = -1;
    let opExecutorIndex = -1;
    if (dataTable && dataTable.getColumnIndex) {
      this.flopRateChartXColumn = dataTable.getColumnIndex(OP_NAME_ID);
      this.flopRateChartYColumn =
          dataTable.getColumnIndex(MEASURED_FLOP_RATE_ID);
      opTypeIndex = dataTable.getColumnIndex(OP_TYPE_ID);
      opNameIndex = dataTable.getColumnIndex(OP_NAME_ID);
      selfTimeIndex = dataTable.getColumnIndex(SELF_TIME_ID);
      opExecutorIndex = dataTable.getColumnIndex(OP_EXECUTOR_ID);
    }

    const filtersForDevice =
        [{column: opExecutorIndex, value: OpExecutor.DEVICE}];
    const filtersForHost = [{column: opExecutorIndex, value: OpExecutor.HOST}];

    this.dataInfoDeviceByType.customChartDataProcessor = this.selectedDiffData ?
        new CategoryDiffTableDataProcessor(
            this.selectedDiffData, filtersForDevice, opTypeIndex,
            selfTimeIndex) :
        new CategoryTableDataProcessor(
            filtersForDevice, opTypeIndex, selfTimeIndex);
    this.dataInfoDeviceByName.customChartDataProcessor =
        new CategoryTableDataProcessor(
            filtersForDevice, opNameIndex, selfTimeIndex);
    this.dataInfoHostByType.customChartDataProcessor = this.selectedDiffData ?
        new CategoryDiffTableDataProcessor(
            this.selectedDiffData, filtersForHost, opTypeIndex, selfTimeIndex) :
        new CategoryTableDataProcessor(
            filtersForHost, opTypeIndex, selfTimeIndex);
    this.dataInfoHostByName.customChartDataProcessor =
        new CategoryTableDataProcessor(
            filtersForHost, opNameIndex, selfTimeIndex);

    // Since the DataInfo has not been updated, the notifyCharts function is
    // called to redraw the graph.
    this.dataProvider.notifyCharts();

    if (this.selectedData && this.selectedData.p) {
      this.architecture = this.selectedData.p['architecture_type'] || '';
      this.task = this.selectedData.p['task_type'] || '';
      this.devicePprofLink = this.selectedData.p['device_tf_pprof_link'] || '';
      this.hostPprofLink = this.selectedData.p['host_tf_pprof_link'] || '';
    }
    this.hasDeviceData = false;
    if (this.selectedData && this.selectedData.rows) {
      this.hasDeviceData = !!this.selectedData.rows.find(row => {
        return row && row.c && row.c[1] && row.c[1].v === OpExecutor.DEVICE;
      });
    }
  }
}
