import {Component, Input, OnChanges, SimpleChanges} from '@angular/core';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {DEFAULT_SIMPLE_DATA_TABLE, type InputPipelineDeviceAnalysis} from 'org_xprof/frontend/app/common/interfaces/data_table';

import {DeviceSideAnalysisDetailDataProvider} from './device_side_analysis_detail_data_provider';

const INFEED_COLUMN_IDS = [
  'stepnum',
  'infeedPercentAverage',
  'infeedPercentMin',
  'infeedPercentMax',
];
const STEPTIME_COLUMN_IDS_FOR_TPU = [
  'stepnum',
  'computeTimeMs',
  'inputTimeMs',
  'idleTimeMs',
  'tooltip',
];
const STEPTIME_COLUMN_IDS_FOR_GPU = [
  'stepnum',
  'deviceComputeTimeMs',
  'deviceToDeviceTimeMs',
  'deviceCollectivesTimeMs',
  'hostComputeTimeMs',
  'kernelLaunchTimeMs',
  'infeedTimeMs',
  'outfeedTimeMs',
  'compileTimeMs',
  'otherTimeMs',
  'tooltip',
];
const COLORS_FOR_TPU = [
  'green',
  'crimson',
  'blue',
];
const COLORS_FOR_GPU = [
  '#9CCC65',
  '#FFEB3B',
  'pink',
  '#64B5F6',
  '#FF9800',
  '#B71C1C',
  'black',
  '#00695C',
  '#5E35B1',
];

interface DeviceSideAnalysisMetrics {
  average?: string;
  max?: string;
  min?: string;
  stddev?: string;
}

/** A device-side analysis detail view component. */
@Component({
  standalone: false,
  selector: 'device-side-analysis-detail',
  templateUrl: './device_side_analysis_detail.ng.html',
  styleUrls: ['./device_side_analysis_detail.scss']
})
export class DeviceSideAnalysisDetail implements OnChanges {
  /** The input pipeline device analysis data. */
  @Input()
  set deviceAnalysis(analysis: InputPipelineDeviceAnalysis|null) {
    this.inputPipelineDeviceAnalysis = analysis;
    analysis = analysis || DEFAULT_SIMPLE_DATA_TABLE;
    analysis.p = analysis.p || {};
    if (!analysis.rows || analysis.rows.length === 0) {
      return;
    }
    this.isTpu = (analysis.p['hardware_type'] || 'TPU') === 'TPU';
    this.steptimeMsMetrics.average = analysis.p['steptime_ms_average'] || '';
    this.steptimeMsMetrics.max = analysis.p['steptime_ms_maximum'] || '';
    this.steptimeMsMetrics.min = analysis.p['steptime_ms_minimum'] || '';
    this.steptimeMsMetrics.stddev =
        analysis.p['steptime_ms_standard_deviation'] || '';

    this.infeedPercentMetrics.average =
        analysis.p['infeed_percent_average'] || '';
    this.infeedPercentMetrics.max = analysis.p['infeed_percent_maximum'] || '';
    this.infeedPercentMetrics.min = analysis.p['infeed_percent_minimum'] || '';
    this.infeedPercentMetrics.stddev =
        analysis.p['infeed_percent_standard_deviation'] || '';
  }

  /** The default column ids. */
  @Input() columnIds = STEPTIME_COLUMN_IDS_FOR_TPU;

  /** The default column colors. */
  @Input() columnColors = COLORS_FOR_TPU;

  isTpu = true;
  inputPipelineDeviceAnalysis: InputPipelineDeviceAnalysis|null = null;
  infeedPercentMetrics: DeviceSideAnalysisMetrics = {};
  steptimeMsMetrics: DeviceSideAnalysisMetrics = {};
  areaChart: google.visualization.AreaChart|null = null;
  lineChart: google.visualization.LineChart|null = null;
  dataProviderForAreaChart = new DeviceSideAnalysisDetailDataProvider();
  dataInfoForAreaChart: ChartDataInfo = {
    data: null,
    dataProvider: this.dataProviderForAreaChart,
    options: {
      hAxis: {title: 'Step Number'},
      vAxis: {title: 'Milliseconds', format: '###.####ms', minValue: 0},
      chartArea: {
        left: 100,
        top: 10,
        width: '65%',
        height: '90%',
      },
      width: 820,
      height: 300,
      colors: COLORS_FOR_TPU,
      backgroundColor: {fill: 'transparent'},
      isStacked: true,
    },
  };
  dataProviderForLineChart = new DeviceSideAnalysisDetailDataProvider();
  dataInfoForLineChart: ChartDataInfo = {
    data: null,
    dataProvider: this.dataProviderForLineChart,
    options: {
      hAxis: {title: 'Step Number'},
      vAxis: {title: '% of step time', format: '###.###\'%\''},
      chartArea: {
        left: 100,
        top: 10,
        width: '65%',
        height: '90%',
      },
      width: 820,
      height: 300,
      legend: 'none',
      lineWidth: 2,
      backgroundColor: {fill: 'transparent'},
      intervals: {style: 'boxes', color: 'red', lineWidth: 4},
    },
  };

  ngOnChanges(changes: SimpleChanges) {
    this.dataProviderForAreaChart.setColumnIds(
        this.isTpu ? this.columnIds : STEPTIME_COLUMN_IDS_FOR_GPU);
    this.dataInfoForAreaChart = {
      ...this.dataInfoForAreaChart,
      data: this.inputPipelineDeviceAnalysis,
      options: {
        ...this.dataInfoForAreaChart.options,
        colors: this.isTpu ? this.columnColors : COLORS_FOR_GPU,
      },
    };

    this.dataProviderForLineChart.setColumnIds(INFEED_COLUMN_IDS);
    this.dataInfoForLineChart = {
      ...this.dataInfoForLineChart,
      data: this.inputPipelineDeviceAnalysis,
    };
  }
}
