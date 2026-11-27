import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  EventEmitter,
  inject,
  Input,
  OnChanges,
  OnInit,
  Output,
  ViewChild,
} from '@angular/core';
import {KELLY_COLORS} from 'org_xprof/frontend/app/common/constants/constants';
import {PrimitiveTypeNumberString} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {loadGoogleChartPackages} from 'org_xprof/frontend/app/common/utils/utils';

const BAR_WIDTH = 50;
const DEFAULT_CHART_WIDTH = 500;

/** A stack bar chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'stack-bar-chart',
  templateUrl: './stack_bar_chart.ng.html',
  styleUrls: ['./stack_bar_chart.scss'],
})
export class StackBarChart implements OnChanges, OnInit {
  /** The data to be display. */
  @Input() data?: Array<Array<PrimitiveTypeNumberString | undefined>>;

  /** The event when the selection of the chart is changed. */
  @Output() selected = new EventEmitter<number>();

  @ViewChild('chart', {static: true}) chartRef!: ElementRef;
  private readonly destroyRef = inject(DestroyRef);

  chart: google.visualization.BarChart | null = null;
  chartWidth = DEFAULT_CHART_WIDTH;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges() {
    this.drawChart();
  }

  drawChart() {
    if (!this.chart) {
      return;
    }
    if (
      !this.data ||
      this.data.length <= 1 ||
      (this.data[0]?.length ?? 0) <= 1
    ) {
      this.chart.clearChart();
      return;
    }

    this.chartWidth = Math.max(
      DEFAULT_CHART_WIDTH,
      this.data.length * BAR_WIDTH,
    );
    const dataTable = window.google.visualization.arrayToDataTable(this.data);

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
    const cancelLoad = loadGoogleChartPackages(['corechart'], () => {
      this.chart = new google.visualization.BarChart(
        this.chartRef.nativeElement,
      );

      google.visualization.events.addListener(
        this.chart,
        'onmouseover',
        (event: google.visualization.ChartSelection) => {
          if (event?.row == null) {
            return;
          }
          this.selected.emit(event.row);
        },
      );

      this.drawChart();
    });
    this.destroyRef.onDestroy(cancelLoad);
  }
}
