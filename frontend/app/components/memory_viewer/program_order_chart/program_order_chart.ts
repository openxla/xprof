import {Component, ElementRef, Input, OnChanges, OnInit, SimpleChanges, ViewChild, ChangeDetectionStrategy} from '@angular/core';
import {type BufferAllocationInfo} from 'org_xprof/frontend/app/common/interfaces/buffer_allocation_info';
import {ChartDataInfo, ChartType} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';

/** A program order chart view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Eager,standalone: false,
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

  @ViewChild('activeChart', {static: false}) activeChartRef!: ElementRef;
  activeChart: google.visualization.AreaChart|null = null;

  maxSize = 0;
  maxOrder = 0;
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
      this.drawChart();
    }
    if (changes['peakInfo']) {
      this.drawPeakChart();
    }
    if (changes['activeInfo']) {
      this.drawActiveChart();
    }
  }

  drawActiveChart() {
    if (!this.activeInfo) {
      if (this.activeChart) {
        this.activeChart.clearChart();
      }
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
          min: 0,
          max: this.maxOrder,
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
          min: 0,
          max: this.maxOrder,
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
          min: 0,
          max: this.maxOrder,
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
