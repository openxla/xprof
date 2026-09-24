import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  EventEmitter,
  HostListener,
  Input,
  OnChanges,
  OnDestroy,
  OnInit,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {HeapObject} from 'org_xprof/frontend/app/common/interfaces/heap_object';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';

/** A max heap chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'max-heap-chart',
  templateUrl: './max_heap_chart.ng.html',
  styleUrls: ['./max_heap_chart.scss'],
})
export class MaxHeapChart implements OnChanges, OnInit, OnDestroy {
  /** The heap object list. */
  @Input() maxHeap: HeapObject[] = [];

  /** The title of view component. */
  @Input() title: string = '';

  /** The selected item index. */
  @Input() selectedIndex: number = -1;

  /** The event when the selection of the chart is changed. */
  @Output() selected = new EventEmitter<number>();

  @ViewChild('chart', {static: false}) chartRef!: ElementRef;

  chart: google.visualization.ColumnChart | null = null;

  /** Whether the chart container is currently hovered by the user cursor. */
  isHovered = false;
  /** The heap item index currently hovered by mouse cursor or chart selection. */
  hoveredIndex = -1;
  /** The mouse cursor horizontal position ratio (0 to 1) within the chart container. */
  hoveredRatio = -1;
  /** The minimum item index bound for the current zoom view window. */
  viewMin = 0;
  /** The maximum item index bound for the current zoom view window. */
  viewMax = -1;

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnDestroy() {
    if (this.chart && google?.visualization?.events) {
      google.visualization.events.removeAllListeners(this.chart);
    }
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['maxHeap']) {
      this.viewMin = 0;
      this.viewMax = this.maxHeap ? this.maxHeap.length - 1 : -1;
      this.hoveredIndex = -1;
      this.hoveredRatio = -1;
      this.drawChart();
    }
    if (changes['selectedIndex']) {
      this.updateSelection();
    }
  }

  onMouseMove(event: MouseEvent) {
    const target = event.currentTarget as HTMLElement;
    if (!target || !target.clientWidth) return;
    const rect = target.getBoundingClientRect();
    const offsetX = event.clientX - rect.left;
    this.hoveredRatio = Math.max(0, Math.min(1, offsetX / target.clientWidth));
  }

  onMouseLeave() {
    this.isHovered = false;
    this.hoveredRatio = -1;
    this.hoveredIndex = -1;
    if (this.chart) {
      this.chart.setSelection([]);
    }
  }

  @HostListener('window:keydown', ['$event'])
  handleKeyDown(event: KeyboardEvent) {
    if (!this.isHovered) return;
    const target = event.target as HTMLElement;
    if (
      target &&
      (target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.isContentEditable)
    ) {
      return;
    }
    const key = event.key.toLowerCase();
    if (['w', 'a', 's', 'd'].includes(key)) {
      event.preventDefault();
      this.handleWasdKey(key);
    }
  }

  handleWasdKey(key: string) {
    if (!this.maxHeap || this.maxHeap.length === 0) return;
    if (this.chart) {
      this.chart.setSelection([]);
    }
    key = key.toLowerCase();
    const total = this.maxHeap.length;
    if (
      this.viewMin < 0 ||
      this.viewMax < 0 ||
      this.viewMax >= total ||
      this.viewMin > this.viewMax
    ) {
      this.viewMin = 0;
      this.viewMax = total - 1;
    }

    if (
      (key === 'a' || key === 'd') &&
      this.viewMin <= 0 &&
      this.viewMax >= total - 1 &&
      total > 1
    ) {
      const initialSpan = Math.max(1, Math.round(total * 0.8));
      const isHoverIndexValid =
        this.hoveredIndex >= 0 && this.hoveredIndex < total;
      const anchorIndex = isHoverIndexValid
        ? this.hoveredIndex
        : Math.floor(total / 2);
      const anchorRatio =
        this.hoveredRatio >= 0 && this.hoveredRatio <= 1
          ? this.hoveredRatio
          : 0.5;

      let newMin = Math.round(anchorIndex - anchorRatio * (initialSpan - 1));
      let newMax = newMin + initialSpan - 1;
      if (newMin < 0) {
        newMin = 0;
        newMax = Math.min(total - 1, initialSpan - 1);
      } else if (newMax >= total) {
        newMax = total - 1;
        newMin = Math.max(0, newMax - initialSpan + 1);
      }
      this.viewMin = newMin;
      this.viewMax = newMax;
    }

    const currentSpan = this.viewMax - this.viewMin + 1;
    const pivotIndex =
      this.hoveredIndex >= 0 && this.hoveredIndex < total
        ? this.hoveredIndex
        : Math.round((this.viewMin + this.viewMax) / 2);
    const relativeRatio =
      currentSpan > 1 ? (pivotIndex - this.viewMin) / (currentSpan - 1) : 0.5;

    if (key === 'w') {
      const newSpan = Math.max(5, Math.round(currentSpan * 0.75));
      let newMin = Math.round(pivotIndex - (newSpan - 1) * relativeRatio);
      let newMax = newMin + newSpan - 1;
      if (newMin < 0) {
        newMin = 0;
        newMax = Math.min(total - 1, newMin + newSpan - 1);
      }
      if (newMax >= total) {
        newMax = total - 1;
        newMin = Math.max(0, newMax - newSpan + 1);
      }
      this.viewMin = newMin;
      this.viewMax = newMax;
      this.drawChart();
    } else if (key === 's') {
      const newSpan = Math.min(total, Math.round(currentSpan * 1.33));
      let newMin = Math.round(pivotIndex - (newSpan - 1) * relativeRatio);
      let newMax = newMin + newSpan - 1;
      if (newMin < 0) {
        newMin = 0;
        newMax = Math.min(total - 1, newMin + newSpan - 1);
      }
      if (newMax >= total) {
        newMax = total - 1;
        newMin = Math.max(0, newMax - newSpan + 1);
      }
      this.viewMin = newMin;
      this.viewMax = newMax;
      this.drawChart();
    } else if (key === 'a') {
      const span = this.viewMax - this.viewMin;
      const step = Math.max(1, Math.round(span * 0.05));
      this.viewMin = Math.max(0, this.viewMin - step);
      this.viewMax = Math.min(total - 1, this.viewMin + span);
      this.drawChart();
    } else if (key === 'd') {
      const span = this.viewMax - this.viewMin;
      const step = Math.max(1, Math.round(span * 0.05));
      this.viewMax = Math.min(total - 1, this.viewMax + step);
      this.viewMin = Math.max(0, this.viewMax - span);
      this.drawChart();
    }
  }

  drawChart() {
    if (!this.chart || !this.maxHeap || this.maxHeap.length === 0) {
      return;
    }

    if (this.viewMax < 0 || this.viewMax >= this.maxHeap.length) {
      this.viewMin = 0;
      this.viewMax = this.maxHeap.length - 1;
    }

    const visibleHeap = this.maxHeap.slice(this.viewMin, this.viewMax + 1);
    const data = [0].concat(
      visibleHeap.map((heapObject) => {
        return heapObject ? heapObject.sizeMiB || 0 : 0;
      }),
    );
    const chartItemColors = visibleHeap.map((heapObject) =>
      utils.getChartItemColorByIndex(heapObject.color || 0),
    );
    const dataTable = google.visualization.arrayToDataTable([
      Array.from<string>({length: data.length}).fill(''),
      data,
    ]);

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
      tooltip: {trigger: 'none'},
      hAxis: {baselineColor: 'transparent'},
      vAxis: {baselineColor: 'transparent'},
      explorer: {
        actions: ['dragToZoom', 'rightClickToReset'],
        maxZoomIn: 0.01,
        maxZoomOut: 1,
      },
    };

    this.chart.draw(
      dataTable,
      options as google.visualization.ColumnChartOptions,
    );

    if (google.visualization && google.visualization.events) {
      google.visualization.events.removeAllListeners(this.chart);
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
          const relativeCol = (event.column || 0) - 1;
          const absoluteIndex = this.viewMin + relativeCol;
          this.hoveredIndex = absoluteIndex;
          const arr = [{row: 0, column: relativeCol + 1}];
          if (this.chart) {
            this.chart.setSelection([]);
            this.chart.setSelection(arr);
          }
          this.selected.emit(absoluteIndex);
        },
      );

      google.visualization.events.addListener(
        this.chart,
        'onmouseleave',
        () => {
          if (this.chart) {
            this.chart.setSelection([]);
          }
          this.hoveredIndex = -1;
        },
      );
    }

    this.updateSelection();
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
      this.chart = new google.visualization.ColumnChart(
        this.chartRef.nativeElement,
      );
      this.drawChart();
    });
  }

  updateSelection() {
    if (!this.chart) {
      return;
    }
    this.chart.setSelection([]);
    const relativeIndex = this.selectedIndex - this.viewMin;
    if (
      this.selectedIndex >= 0 &&
      relativeIndex >= 0 &&
      relativeIndex <= (this.viewMax - this.viewMin)
    ) {
      this.chart.setSelection([{row: 0, column: relativeIndex + 1}]);
    }
  }
}
