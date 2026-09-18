import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  effect,
  input,
  output,
  viewChild,
} from '@angular/core';
import {MatCard, MatCardContent, MatCardTitle} from '@angular/material/card';
import {STACK_CHART_FILL_COLORS} from 'org_xprof/frontend/app/common/constants/constants';
import {type InputPipelineAnalysis} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {clampDataTableNumericValues} from 'org_xprof/frontend/app/common/utils/chart_utils';

const MAX_CHART_WIDTH = 800;
const COLORS_FOR_GPU = [
  '#4b7b4b',
  '#8d6708',
  '#d252b2',
  '#2a7ab0',
  '#e65722',
  '#8b0000',
  '#000000',
  '#005555',
  '#483d8b',
];

/** A step-time graph view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'step-time-graph',
  templateUrl: './step_time_graph.ng.html',
  styleUrls: ['./step_time_graph.scss'],
  imports: [MatCard, MatCardContent, MatCardTitle],
})
export class StepTimeGraph implements AfterViewInit {
  /** The input pipeline analyis data. */
  readonly inputPipelineAnalysis = input<InputPipelineAnalysis | null>(null);

  /** The default column colors. */
  readonly columnColors = input(STACK_CHART_FILL_COLORS);

  readonly chartRef = viewChild<ElementRef>('chart');
  readonly ready = output<void>();

  title = 'Step-time Graph';
  height = 300;
  width = 0;
  chart: google.visualization.AreaChart | null = null;

  constructor() {
    effect(() => {
      this.inputPipelineAnalysis();
      this.columnColors();
      this.width = 0;
      this.drawChart();
    });
  }

  ngAfterViewInit() {
    this.loadGoogleChart();
  }

  @HostListener('window:resize')
  onResize() {
    this.drawChart();
  }

  drawChart() {
    const chartEl = this.chartRef()?.nativeElement;
    if (!chartEl) {
      return;
    }

    const newWidth = Math.min(MAX_CHART_WIDTH, chartEl.offsetWidth);

    const inputPipelineAnalysis = this.inputPipelineAnalysis();
    if (!this.chart || !inputPipelineAnalysis || this.width === newWidth) {
      return;
    }

    const dataTable = new google.visualization.DataTable(inputPipelineAnalysis);
    const columnsIds = dataTable
      .getTableProperty('step_time_graph_column_ids')
      .split(',');
    let colors = this.columnColors();
    this.height = 300;
    const p = inputPipelineAnalysis.p || {};
    if ((p['hardware_type'] || 'TPU') !== 'TPU') {
      colors = COLORS_FOR_GPU;
      this.height = 400;
    }

    let i = 0;
    while (i < dataTable.getNumberOfColumns()) {
      if (!columnsIds.includes(dataTable.getColumnId(i))) {
        dataTable.removeColumn(i);
        continue;
      }
      i++;
    }

    clampDataTableNumericValues(dataTable, /* startCol= */ 1);

    const showTextEvery = Math.max(
      1,
      Math.floor(dataTable.getNumberOfRows() / 10),
    );
    const options = {
      title: 'Step Time (in milliseconds)',
      titleTextStyle: {bold: true},
      hAxis: {
        title: 'Step Number',
        showTextEvery,
        textStyle: {bold: true},
      },
      vAxis: {
        format: '###.####',
        minValue: 0,
        viewWindow: {min: 0},
        textStyle: {bold: true},
      },
      chartArea: {left: 50, width: '60%'},
      colors: colors,
      height: this.height,
      isStacked: true,
    };
    this.chart.draw(dataTable, options);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
      return;
    }

    google.charts.safeLoad({'packages': ['corechart']});
    google.charts.setOnLoadCallback(() => {
      const chartEl = this.chartRef()?.nativeElement;
      if (!chartEl) return;
      this.chart = new google.visualization.AreaChart(chartEl);
      google.visualization.events.addListener(this.chart, 'ready', () => {
        this.ready.emit();
      });
      this.drawChart();
    });
  }
}
