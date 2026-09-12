import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  HostListener,
  Input,
  OnChanges,
  OnInit,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {type BufferAllocationInfo} from 'org_xprof/frontend/app/common/interfaces/buffer_allocation_info';
import {ChartDataInfo, ChartType} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Chart} from 'org_xprof/frontend/app/components/chart/chart';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';

/** A program order chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'program-order-chart',
  templateUrl: './program_order_chart.ng.html',
  styleUrls: ['./program_order_chart.scss']
})
export class ProgramOrderChart implements OnChanges, OnInit {
  /** The heap size list. */
  @Input() heapSizes: number[] = [];

  /** The unpadded heap size list. */
  @Input() unpaddedHeapSizes: number[] = [];

  /** The HLO instruction name corresponding to each program order point. */
  @Input() hloInstructionNames: string[] = [];

  /** The peak buffer allocation information. */
  @Input() peakInfo?: BufferAllocationInfo;

  /** The active buffer allocation information. */
  @Input() activeInfo?: BufferAllocationInfo;

  /** Optional timeline URL. */
  @Input() timelineUrl = '';

  @Output() readonly selectedStep = new EventEmitter<number | null>();

  onHeapChartSelected(selection: google.visualization.ChartSelection[]) {
    if (
      selection &&
      selection.length > 0 &&
      selection[0].row !== undefined &&
      selection[0].row !== null
    ) {
      this.selectedStep.emit(selection[0].row);
    } else {
      this.selectedStep.emit(null);
    }
  }

  @HostListener('window:resize')
  onResize() {
    this.updateCharts();
  }

  @ViewChild('activeChart', {static: false}) activeChart?: Chart;

  /** The maximum allocated heap size among all data points. */
  maxSize = 0;
  /** The maximum program order index (total points - 1). */
  maxOrder = 0;
  /** Whether the chart container is currently hovered by the user cursor. */
  isHovered = false;
  /** The program order index currently hovered by mouse cursor. */
  hoveredIndex = -1;
  /** The mouse cursor horizontal position ratio (0 to 1) within the chart container. */
  hoveredRatio = -1;
  /** The minimum program order index bound for the current zoom view window. */
  viewMin = 0;
  /** The maximum program order index bound for the current zoom view window. */
  viewMax = -1;

  activeChartDataInfo: ChartDataInfo = {
    data: null,
    dataProvider: new DefaultDataProvider(),
  };
  peakChartDataInfo: ChartDataInfo = {
    data: null,
    dataProvider: new DefaultDataProvider(),
  };
  heapChartDataInfo: ChartDataInfo = {
    data: null,
    dataProvider: new DefaultDataProvider(),
  };

  readonly AREA_CHART = ChartType.AREA_CHART;
  readonly LINE_CHART = ChartType.LINE_CHART;

  ngOnInit() {
    this.updateCharts();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['heapSizes'] || changes['unpaddedHeapSizes']) {
      this.maxOrder = this.heapSizes.length ? this.heapSizes.length - 1 : 0;
      this.viewMin = 0;
      this.viewMax = this.maxOrder;
      this.hoveredIndex = -1;
      this.hoveredRatio = -1;
      this.drawChart();
    }
    if (changes['peakInfo']) {
      this.drawPeakChart();
    }
    if (changes['activeInfo']) {
      this.drawActiveChart();
    }
  }

  onMouseMove(event: MouseEvent) {
    const target = event.currentTarget as HTMLElement;
    if (!target || !target.clientWidth) return;
    const rect = target.getBoundingClientRect();
    const offsetX = event.clientX - rect.left;
    this.hoveredRatio = Math.max(0, Math.min(1, offsetX / target.clientWidth));
    const effectiveViewMax =
      this.viewMax >= 0 && this.viewMax <= this.maxOrder
        ? this.viewMax
        : this.maxOrder;
    const currentSpan = effectiveViewMax - this.viewMin + 1;
    this.hoveredIndex = Math.round(
      this.viewMin + this.hoveredRatio * (currentSpan - 1),
    );
  }

  onMouseLeave() {
    this.isHovered = false;
    this.hoveredRatio = -1;
    this.hoveredIndex = -1;
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
    if (this.maxOrder <= 0) return;
    key = key.toLowerCase();
    const total = this.maxOrder + 1;
    if (
      this.viewMin < 0 ||
      this.viewMax < 0 ||
      this.viewMax >= total ||
      this.viewMin > this.viewMax
    ) {
      this.viewMin = 0;
      this.viewMax = this.maxOrder;
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
      this.updateCharts();
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
      this.updateCharts();
    } else if (key === 'a') {
      const span = this.viewMax - this.viewMin;
      const step = Math.max(1, Math.round(span * 0.05));
      this.viewMin = Math.max(0, this.viewMin - step);
      this.viewMax = Math.min(total - 1, this.viewMin + span);
      this.updateCharts();
    } else if (key === 'd') {
      const span = this.viewMax - this.viewMin;
      const step = Math.max(1, Math.round(span * 0.05));
      this.viewMax = Math.min(total - 1, this.viewMax + step);
      this.viewMin = Math.max(0, this.viewMax - span);
      this.updateCharts();
    }
  }

  drawActiveChart() {
    if (!this.activeInfo) {
      if (this.activeChart?.chart) {
        this.activeChart.chart.clearChart();
      }
      this.activeChartDataInfo = {
        ...this.activeChartDataInfo,
        data: null,
      };
      return;
    }

    const dataTable = google.visualization.arrayToDataTable([
      ['Schedule', 'Size'],
      [this.activeInfo.alloc, this.activeInfo.size],
      [this.activeInfo.free, this.activeInfo.size],
    ]);

    const options: google.visualization.AreaChartOptions = {
      areaOpacity: 0.7,
      backgroundColor: 'transparent',
      chartArea: {
        height: '80%',
      },
      colors: [this.activeInfo.color || ''],
      hAxis: {
        baselineColor: 'transparent',
        gridlines: {color: 'transparent'},
        textPosition: 'none',
        viewWindow: {
          min: this.viewMin,
          max:
            this.viewMax >= 0 && this.viewMax <= this.maxOrder
              ? this.viewMax
              : this.maxOrder,
        },
      },
      vAxis: {
        baselineColor: 'transparent',
        gridlines: {color: 'transparent'},
        textPosition: 'none',
        viewWindow: {
          min: 0,
          max: this.maxSize,
        },
      },
      legend: {position: 'none'},
      lineWidth: 2,
    };

    this.activeChartDataInfo = {
      ...this.activeChartDataInfo,
      data: JSON.parse(dataTable.toJSON()) as SimpleDataTable,
      options,
    };
  }

  drawChart() {
    if (!this.heapSizes.length || !this.unpaddedHeapSizes.length) {
      return;
    }

    const data = [];
    this.maxOrder = this.heapSizes.length - 1;
    if (this.viewMax < 0 || this.viewMax > this.maxOrder) {
      this.viewMax = this.maxOrder;
    }
    this.maxSize = 0;
    for (let i = 0; i < this.heapSizes.length; i++) {
      this.maxSize = Math.max(
          this.maxSize, Math.max(this.heapSizes[i], this.unpaddedHeapSizes[i]));
      const tooltip = `<div>
        Program Order: ${i}<br>Size: ${this.heapSizes[i].toFixed(1)}<br>
        Unpadded Size: ${this.unpaddedHeapSizes[i].toFixed(1)}<br>
        HLO instruction: ${this.hloInstructionNames[i]}
        </div>`;
      data.push([i,
                this.heapSizes[i], tooltip,
                this.unpaddedHeapSizes[i],tooltip]);
    }
    this.maxSize = Math.round(this.maxSize * 1.1);

    const dataTable = new google.visualization.DataTable();
    dataTable.addColumn('number', 'Schedule');
    dataTable.addColumn('number', 'Size');
    dataTable.addColumn({type: 'string', role: 'tooltip', 'p': {'html': true}});
    dataTable.addColumn('number', 'Unpadded Size');
    dataTable.addColumn({type: 'string', role: 'tooltip', 'p': {'html': true}});
    dataTable.addRows(data);

    const options: google.visualization.LineChartOptions = {
      backgroundColor: 'transparent',
      chartArea: {
        height: '80%',
      },
      hAxis: {
        title: 'Program Order',
        baselineColor: 'transparent',
        viewWindow: {
          min: this.viewMin,
          max:
            this.viewMax >= 0 && this.viewMax <= this.maxOrder
              ? this.viewMax
              : this.maxOrder,
        },
      },
      vAxis: {
        title: 'Allocated Heap Size',
        baselineColor: 'transparent',
        viewWindow: {
          min: 0,
          max: this.maxSize,
        },
      },
      legend: {position: 'top'},
      tooltip: {isHtml: true},
    };

    this.heapChartDataInfo = {
      ...this.heapChartDataInfo,
      data: JSON.parse(dataTable.toJSON()) as SimpleDataTable,
      options,
    };
  }

  drawPeakChart() {
    if (!this.peakInfo) {
      return;
    }

    const peakWidth = Math.max(Math.round(this.maxOrder / 50), 1);
    const peakAlloc =
        Math.max(Math.round(this.peakInfo.alloc - peakWidth / 2), 0);
    const peakFree = Math.min(peakAlloc + peakWidth, this.maxOrder);
    const dataTable = new google.visualization.DataTable();
    dataTable.addColumn('number', 'Schedule');
    dataTable.addColumn('number', 'Allocated Size');
    dataTable.addColumn({type: 'string', role: 'tooltip'});
    dataTable.addRows([
      [
        peakAlloc, this.peakInfo.size,
        `peak memory allocation: ${this.peakInfo.size}`
      ],
      [
        peakFree, this.peakInfo.size,
        `peak memory allocation: ${this.peakInfo.size}`
      ],
    ]);

    const options: google.visualization.AreaChartOptions = {
      backgroundColor: 'transparent',
      chartArea: {
        height: '80%',
      },
      colors: ['#00ff00'],
      hAxis: {
        baselineColor: 'transparent',
        gridlines: {color: 'transparent'},
        textPosition: 'none',
        viewWindow: {
          min: this.viewMin,
          max:
            this.viewMax >= 0 && this.viewMax <= this.maxOrder
              ? this.viewMax
              : this.maxOrder,
        },
      },
      vAxis: {
        baselineColor: 'transparent',
        gridlines: {color: 'transparent'},
        textPosition: 'none',
        viewWindow: {
          min: 0,
          max: this.maxSize,
        },
      },
      legend: {position: 'none'},
      lineWidth: 0,
    };

    this.peakChartDataInfo = {
      ...this.peakChartDataInfo,
      data: JSON.parse(dataTable.toJSON()) as SimpleDataTable,
      options,
    };
  }

  updateCharts() {
    this.drawActiveChart();
    this.drawPeakChart();
    this.drawChart();
  }
}
