import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnInit,
  effect,
  input,
  viewChild,
} from '@angular/core';

/** A organization chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'org-chart',
  templateUrl: './org_chart.ng.html',
  styleUrls: ['./org_chart.scss'],
})
export class OrgChart implements OnInit {
  readonly dataView = input<google.visualization.DataView>();

  chart?: google.visualization.OrgChart;

  readonly chartRef = viewChild<ElementRef>('chart');

  constructor() {
    effect(() => {
      this.dataView();
      this.drawChart();
    });
  }

  ngOnInit() {
    this.loadGoogleChart();
  }

  drawChart() {
    const dataView = this.dataView();
    if (!this.chart || !dataView) {
      return;
    }

    const options: google.visualization.OrgChartOptions = {
      allowHtml: true,
    };

    this.chart.draw(dataView, options);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.safeLoad({'packages': ['orgchart']});
    google.charts.setOnLoadCallback(() => {
      const chartEl = this.chartRef()?.nativeElement;
      if (!chartEl) return;
      this.chart = new google.visualization.OrgChart(chartEl);
      this.drawChart();
    });
  }
}
