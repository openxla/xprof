import {AfterViewInit, Component, ElementRef, HostListener, Input, OnChanges, SimpleChanges, ViewChild, ChangeDetectionStrategy} from '@angular/core';
import {STACK_CHART_FILL_COLORS} from 'org_xprof/frontend/app/common/constants/constants';
import {type InputPipelineAnalysis} from 'org_xprof/frontend/app/common/interfaces/data_table';

const MAX_CHART_WIDTH = 800;
const COLORS_FOR_GPU = [
  '#4b7b4b',
  '#8d6708',
  '#d252b2',
  '#2a7ab0',
  '#e65722',
  '#8b0000',
  '#000000',
  '#005555',
  '#483d8b',
];

/** A step-time graph view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Eager,standalone: false,
  selector: 'step-time-graph',
  templateUrl: './step_time_graph.ng.html',
  styleUrls: ['./step_time_graph.scss']
})
export class StepTimeGraph implements AfterViewInit, OnChanges {
  /** The input pipeline analyis data. */
  @Input() inputPipelineAnalysis: InputPipelineAnalysis|null = null;

  /** The default column colors. */
  @Input() columnColors = STACK_CHART_FILL_COLORS;

  @ViewChild('chart', {static: false}) chartRef!: ElementRef;

  title = 'Step-time Graph';
  height = 300;
  width = 0;
  chart: google.visualization.AreaChart|null = null;

  ngAfterViewInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.width = 0;
    this.drawChart();
  }

  @HostListener('window:resize')
  onResize() {
    this.drawChart();
  }

  drawChart() {
    if (!this.chartRef) {
      return;
    }

    const newWidth =
        Math.min(MAX_CHART_WIDTH, this.chartRef.nativeElement.offsetWidth);

    if (!this.chart || !this.inputPipelineAnalysis || this.width === newWidth) {
      return;
    }

    const dataTable =
        new google.visualization.DataTable(this.inputPipelineAnalysis);
    const columnsIds =
        dataTable.getTableProperty('step_time_graph_column_ids').split(',');
    let colors = this.columnColors;
    this.height = 300;
    this.inputPipelineAnalysis.p = this.inputPipelineAnalysis.p || {};
    if ((this.inputPipelineAnalysis.p['hardware_type'] || 'TPU') !== 'TPU') {
      colors = COLORS_FOR_GPU;
      this.height = 400;
    }

    let i = 0;
    while (i < dataTable.getNumberOfColumns()) {
      if (!columnsIds.includes(dataTable.getColumnId(i))) {
        dataTable.removeColumn(i);
        continue;
      }
      i++;
    }

    const showTextEvery =
        Math.max(1, Math.floor(dataTable.getNumberOfRows() / 10));
    const options = {
      title: 'Step Time (in milliseconds)',
      titleTextStyle: {bold: true},
      hAxis: {
        title: 'Step Number',
        showTextEvery,
        textStyle: {bold: true},
      },
      vAxis: {
        format: '###.####',
        minValue: 0,
        textStyle: {bold: true},
      },
      chartArea: {left: 50, width: '60%'},
      colors: colors,
      height: this.height,
      isStacked: true,
    };
    this.chart.draw(dataTable, options);
  }

  loadGoogleChart() {
    if (!google || !google.charts) {
      setTimeout(() => {
        this.loadGoogleChart();
      }, 100);
    }

    google.charts.safeLoad({'packages': ['corechart']});
    google.charts.setOnLoadCallback(() => {
      this.chart =
          new google.visualization.AreaChart(this.chartRef.nativeElement);
      this.drawChart();
    });
  }
}
