import {Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild, ChangeDetectionStrategy} from '@angular/core';

/** A organization chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Eager,standalone: false,
  selector: 'org-chart',
  templateUrl: './org_chart.ng.html',
  styleUrls: ['./org_chart.scss']
})
export class OrgChart implements OnChanges, OnInit {
  @Input() dataView?: google.visualization.DataView;

  chart?: google.visualization.OrgChart;

  @ViewChild('chart', {static: false}) chartRef!: ElementRef;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.drawChart();
  }

  drawChart() {
    if (!this.chart || !this.dataView) {
      return;
    }

    const options: google.visualization.OrgChartOptions = {
      allowHtml: true,
    };

    this.chart.draw(this.dataView, options);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.safeLoad({'packages': ['orgchart']});
    google.charts.setOnLoadCallback(() => {
      this.chart =
          new google.visualization.OrgChart(this.chartRef.nativeElement);
      this.drawChart();
    });
  }
}
