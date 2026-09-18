import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnInit,
  effect,
  input,
  output,
  viewChild,
} from '@angular/core';
import {KELLY_COLORS} from 'org_xprof/frontend/app/common/constants/constants';
import {PrimitiveTypeNumberString} from 'org_xprof/frontend/app/common/interfaces/data_table';

const BAR_WIDTH = 50;
const DEFAULT_CHART_WIDTH = 500;

/** A stack bar chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'stack-bar-chart',
  templateUrl: './stack_bar_chart.ng.html',
  styleUrls: ['./stack_bar_chart.scss'],
})
export class StackBarChart implements OnInit {
  /** The data to be display. */
  readonly data = input<Array<Array<PrimitiveTypeNumberString | undefined>>>();

  /** The event when the selection of the chart is changed. */
  readonly selected = output<number>();

  readonly chartRef = viewChild<ElementRef>('chart');

  chart: google.visualization.BarChart | null = null;
  chartWidth = DEFAULT_CHART_WIDTH;

  constructor() {
    effect(() => {
      this.data();
      this.drawChart();
    });
  }

  ngOnInit() {
    this.loadGoogleChart();
  }

  drawChart() {
    const data = this.data();
    if (!this.chart || !data) {
      return;
    }

    this.chartWidth = Math.max(DEFAULT_CHART_WIDTH, data.length * BAR_WIDTH);
    const dataTable = window.google.visualization.arrayToDataTable(data);

    const options = {
      backgroundColor: 'transparent',
      chartArea: {
        left: 50,
        right: 20,
        top: 50,
        bottom: 20,
      },
      focusTarget: 'category',
      isStacked: true,
      legend: {
        position: 'top',
        maxLines: 3,
        textStyle: {fontSize: 12},
      },
      hAxis: {textStyle: {fontSize: 12}},
      vAxis: {textStyle: {fontSize: 12}},
      orientation: 'horizontal',
      tooltip: {trigger: 'none'},
      width: this.chartWidth,
      colors: KELLY_COLORS,
    };

    this.chart.draw(dataTable, options as google.visualization.BarChartOptions);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.safeLoad({'packages': ['corechart']});
    google.charts.setOnLoadCallback(() => {
      const chartEl = this.chartRef()?.nativeElement;
      if (!chartEl) return;
      this.chart = new google.visualization.BarChart(chartEl);

      google.visualization.events.addListener(
        this.chart,
        'onmouseover',
        (event: google.visualization.ChartSelection) => {
          event = event || {};
          this.selected.emit(event.row || 0);
        },
      );

      this.drawChart();
    });
  }
}
