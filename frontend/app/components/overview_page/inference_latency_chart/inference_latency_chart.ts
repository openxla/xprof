import {CommonModule} from '@angular/common';
import {
  AfterViewInit,
  Component,
  ElementRef,
  HostListener,
  Input,
  OnChanges,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {MatCardModule} from '@angular/material/card';
import {type SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';

const MAX_CHART_WIDTH = 800;

/** An inference latency chart view component. */
@Component({
  standalone: true,
  selector: 'inference-latency-chart',
  templateUrl: 'inference_latency_chart.ng.html',
  styleUrls: ['inference_latency_chart.scss'],
  imports: [CommonModule, MatCardModule],
})
export class InferenceLatencyChart implements AfterViewInit, OnChanges {
  /** The inference latency data. */
  @Input() inferenceLatencyData?: SimpleDataTable;

  @ViewChild('chart', {static: false}) chartRef!: ElementRef;

  title = 'Inference Session Latency Breakdown';
  height = 300;
  width = 0;
  chart: google.visualization.ColumnChart | null = null;

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

    const newWidth = Math.min(
      MAX_CHART_WIDTH,
      this.chartRef.nativeElement.offsetWidth,
    );

    if (!this.chart || !this.inferenceLatencyData || this.width === newWidth) {
      return;
    }
    this.width = newWidth;

    const dataTable = new google.visualization.DataTable(
      this.inferenceLatencyData,
    );
    const dataView = new google.visualization.DataView(dataTable);
    dataView.setColumns([
      0,
      1,
      2,
      3,
      {
        calc: 'stringify',
        sourceColumn: 4,
        type: 'string',
        role: 'annotation',
      },
    ]);

    const options = {
      title: 'Inference Session Latency at Percentile',
      titleTextStyle: {bold: true},
      hAxis: {title: 'Percentile', showTextEvery: 1},
      vAxis: {
        title: 'Session Latency Breakdown',
        format: '###.####ms',
        minValue: 0,
      },
      legend: {position: 'right', maxLines: 3},
      bar: {groupWidth: '35%'},
      chartArea: {left: 100, width: '65%'},
      colors: ['orange', 'blue', 'red'],
      height: this.height,
      isStacked: true,
      annotations: {
        alwaysOutside: true,
        textStyle: {bold: true, color: 'black'},
      },
    };
    this.chart.draw(
      dataView,
      options as google.visualization.ColumnChartOptions,
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
      this.chart = new google.visualization.ColumnChart(
        this.chartRef.nativeElement,
      );
      this.drawChart();
    });
  }
}
