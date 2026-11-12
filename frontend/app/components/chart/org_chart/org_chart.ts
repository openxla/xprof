import {Component, DestroyRef, ElementRef, inject, Input, OnChanges, OnInit, SimpleChanges, ViewChild, ChangeDetectionStrategy} from '@angular/core';
import {loadGoogleChartPackages} from 'org_xprof/frontend/app/common/utils/utils';

/** A organization chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'org-chart',
  templateUrl: './org_chart.ng.html',
  styleUrls: ['./org_chart.scss']
})
export class OrgChart implements OnChanges, OnInit {
  @Input() dataView?: google.visualization.DataView;

  chart?: google.visualization.OrgChart;

  @ViewChild('chart', {static: false}) chartRef!: ElementRef;
  private readonly destroyRef = inject(DestroyRef);

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
    const cancelLoad = loadGoogleChartPackages(['orgchart'], () => {
      this.chart =
          new google.visualization.OrgChart(this.chartRef.nativeElement);
      this.drawChart();
    });
    this.destroyRef.onDestroy(cancelLoad);
  }
}
