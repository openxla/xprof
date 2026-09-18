import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  HostListener,
  OnInit,
  effect,
  input,
  output,
  viewChild,
} from '@angular/core';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';

/** A max heap chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'max-heap-chart',
  templateUrl: './max_heap_chart.ng.html',
  styleUrls: ['./max_heap_chart.scss'],
})
export class MaxHeapChart implements OnInit {
  /** The heap object list. */
  readonly maxHeap = input<HeapObject[]>([]);

  /** The title of view component. */
  readonly title = input<string>('');

  /** The selected item index. */
  readonly selectedIndex = input<number>(-1);

  /** The event when the selection of the chart is changed. */
  readonly selected = output<number>();

  readonly chartRef = viewChild<ElementRef>('chart');

  chart: google.visualization.ColumnChart | null = null;

  constructor() {
    effect(() => {
      this.maxHeap();
      this.drawChart();
    });
    effect(() => {
      this.selectedIndex();
      this.updateSelection();
    });
  }

  @HostListener('window:resize')
  onResize() {
    this.drawChart();
  }

  ngOnInit() {
    this.loadGoogleChart();
  }

  drawChart() {
    if (!this.chart || !this.maxHeap()) {
      return;
    }

    const maxHeap = this.maxHeap();
    const data: Array<string | number> = (
      [''] as Array<string | number>
    ).concat(
      maxHeap.map((heapObject) => {
        return heapObject ? heapObject.sizeMiB || 0 : 0;
      }),
    );
    const chartItemColors = maxHeap.map((heapObject) =>
      utils.getChartItemColorByIndex(heapObject.color || 0),
    );
    const headers = [''].concat(
      maxHeap.map((heapObject) => {
        return heapObject ? heapObject.instructionName || '' : '';
      }),
    );
    const dataTable = google.visualization.arrayToDataTable([headers, data]);

    const options = {
      bar: {groupWidth: '100%'},
      colors: chartItemColors,
      chartArea: {
        left: 0,
        right: 0,
        width: '100%',
        height: '100%',
      },
      isStacked: 'percent',
      legend: {position: 'none'},
      orientation: 'vertical',
      tooltip: {showColorCode: true},
      hAxis: {baselineColor: 'transparent'},
      vAxis: {baselineColor: 'transparent'},
    };

    this.chart.draw(
      dataTable,
      options as google.visualization.ColumnChartOptions,
    );

    google.visualization.events.addListener(this.chart, 'click', () => {
      if (this.chart) {
        this.chart.setSelection([]);
      }
    });

    google.visualization.events.addListener(
      this.chart,
      'onmouseover',
      (event: google.visualization.ChartSelection) => {
        event = event || {};
        const arr = [];
        arr.push(event);
        if (this.chart) {
          this.chart.setSelection(arr);
        }
        this.selected.emit((event.column || 0) - 1);
      },
    );
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
      this.chart = new google.visualization.ColumnChart(chartEl);
      this.drawChart();
    });
  }

  updateSelection() {
    if (!this.chart) {
      return;
    }
    this.chart.setSelection([{row: 0, column: this.selectedIndex() + 1}]);
  }
}
