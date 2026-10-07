import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  ElementRef,
  EventEmitter,
  HostListener,
  inject,
  Input,
  OnChanges,
  OnInit,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';

/** A max heap chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
  selector: 'max-heap-chart',
  templateUrl: './max_heap_chart.ng.html',
  styleUrls: ['./max_heap_chart.scss'],
})
export class MaxHeapChart implements OnChanges, OnInit {
  /** The heap object list. */
  @Input() maxHeap: HeapObject[] = [];

  /** The title of view component. */
  @Input() title: string = '';

  /** The selected item index. */
  @Input() selectedIndex: number = -1;

  /** The event when the selection of the chart is changed. */
  @Output() selected = new EventEmitter<number>();
  private readonly cdr = inject(ChangeDetectorRef);

  @ViewChild('chart', {static: false}) chartRef!: ElementRef;

  chart: google.visualization.ColumnChart | null = null;

  @HostListener('window:resize')
  onResize() {
    this.drawChart();
  }

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['maxHeap']) {
      this.drawChart();
    }
    if (changes['selectedIndex']) {
      this.updateSelection();
    }
  }

  drawChart() {
    if (!this.chart || !this.maxHeap) {
      return;
    }

    const data: Array<string | number> = (
      [''] as Array<string | number>
    ).concat(
      this.maxHeap.map((heapObject) => {
        return heapObject ? heapObject.sizeMiB || 0 : 0;
      }),
    );
    const chartItemColors = this.maxHeap.map((heapObject) =>
      utils.getChartItemColorByIndex(heapObject.color || 0),
    );
    const headers = [''].concat(
      this.maxHeap.map((heapObject) => {
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
      this.cdr.markForCheck();
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
        this.cdr.markForCheck();
      },
    );
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
        this.cdr.markForCheck();
      }, 100);
    }

    google.charts.safeLoad({'packages': ['corechart']});
    google.charts.setOnLoadCallback(() => {
      this.chart = new google.visualization.ColumnChart(
        this.chartRef.nativeElement,
      );
      this.drawChart();
      this.cdr.markForCheck();
    });
  }

  updateSelection() {
    if (!this.chart) {
      return;
    }
    this.chart.setSelection([{row: 0, column: this.selectedIndex + 1}]);
  }
}
