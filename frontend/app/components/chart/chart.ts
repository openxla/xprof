import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  ElementRef,
  EventEmitter,
  HostListener,
  inject,
  Input,
  NgModule,
  OnChanges,
  OnInit,
  Output,
  SimpleChanges,
} from '@angular/core';
import {
  ChartClass,
  type ChartDataInfo,
  ChartType,
  CustomChartDataProcessor,
  DataTableOrDataView,
} from 'org_xprof/frontend/app/common/interfaces/chart';
import {loadGoogleChartPackages} from 'org_xprof/frontend/app/common/utils/utils';

/** A common chart component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'chart',
  template: '',
  // Below 960px wide charts scroll sideways instead of widening the page. Only
  // horizontal overflow scrolls: the off-screen accessibility element Google
  // Charts adds under each chart would otherwise add a vertical scrollbar.
  styles: [
    ':host {display: block;}',
    `@media (max-width: 959.98px) {
      :host {max-width: 100%; overflow-x: auto; overflow-y: hidden;}
    }`,
  ],
})
export class Chart implements OnChanges, OnInit {
  /** The type of chart. */
  @Input() chartType?: ChartType;

  /** The information of chart data. */
  @Input() dataInfo?: ChartDataInfo;

  /** The event for the number of rows of processed data. */
  @Output() readonly processedNumberOfRows = new EventEmitter<number>();

  /** The event when the selection of the chart is changed. */
  @Output() readonly selected = new EventEmitter<
    google.visualization.ChartSelection[]
  >();

  chart?: ChartClass;

  private readonly elementRef: ElementRef = inject(ElementRef);
  private readonly destroyRef = inject(DestroyRef);
  private isSmallScreen = window.innerWidth < 960;

  @HostListener('window:resize')
  onResize() {
    const isSmallScreen = window.innerWidth < 960;
    if (isSmallScreen !== this.isSmallScreen) {
      this.isSmallScreen = isSmallScreen;
      this.draw();
    }
  }

  ngOnInit() {
    this.loadGoogleChart();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['dataInfo'] && this.dataInfo && this.dataInfo.dataProvider) {
      const previousInfo = changes['dataInfo'].previousValue as
        | ChartDataInfo
        | undefined;
      // Clear a provider only when this chart filled it before, so a provider
      // shared through `data: null` keeps data deferred until charts load.
      const clearsOwnData =
        previousInfo?.dataProvider === this.dataInfo.dataProvider &&
        Boolean(previousInfo?.data);
      if (this.dataInfo.data || clearsOwnData) {
        this.dataInfo.dataProvider.parseData(this.dataInfo.data);
      }
      this.dataInfo.dataProvider.setFilters(this.dataInfo.filters || []);
    }
    this.draw();
  }

  draw() {
    this.isSmallScreen = window.innerWidth < 960;
    if (!this.chart) {
      return;
    }

    if (!this.dataInfo || !this.dataInfo.dataProvider) {
      this.chart.clearChart();
      return;
    }

    const processedData = this.getProcessedData(
      this.dataInfo.customChartDataProcessor,
    );

    const options =
      this.dataInfo.dataProvider.getOptions() || this.dataInfo.options;
    const drawOptions =
      this.isSmallScreen && options && 'tooltip' in options && options.tooltip
        ? {...options, tooltip: {...options.tooltip, ignoreBounds: false}}
        : options;

    if (processedData) {
      // tslint:disable-next-line:no-any
      this.chart.draw(processedData, drawOptions as any);
    }

    this.processedNumberOfRows.emit(
      processedData ? processedData.getNumberOfRows() : 0,
    );
  }

  getProcessedData(
    customChartDataProcessor: CustomChartDataProcessor | undefined,
  ): DataTableOrDataView | null {
    if (!this.dataInfo || !this.dataInfo.dataProvider) {
      return null;
    }

    if (customChartDataProcessor && customChartDataProcessor.process) {
      return customChartDataProcessor.process(this.dataInfo.dataProvider);
    }

    return this.dataInfo.dataProvider.process();
  }

  loadGoogleChart() {
    const packages = ['corechart', 'table', 'line'];
    const cancelLoad = loadGoogleChartPackages(packages, () => {
      this.initChart();
      this.initDataProvider();
      this.draw();
      if (this.chart) {
        google.visualization.events.addListener(this.chart, 'select', () => {
          this.selected.emit(this.chart?.getSelection() || []);
        });
      }
    });
    this.destroyRef.onDestroy(cancelLoad);
  }

  initChart() {
    switch (this.chartType) {
      case ChartType.AREA_CHART:
        this.chart = new google.visualization.AreaChart(
          this.elementRef.nativeElement,
        );
        break;
      case ChartType.BAR_CHART:
        this.chart = new google.visualization.BarChart(
          this.elementRef.nativeElement,
        );
        break;
      case ChartType.BUBBLE_CHART:
        this.chart = new google.visualization.BubbleChart(
          this.elementRef.nativeElement,
        );
        break;
      case ChartType.CANDLESTICK_CHART:
        this.chart = new google.visualization.CandlestickChart(
          this.elementRef.nativeElement,
        );
        break;
      case ChartType.COLUMN_CHART:
        this.chart = new google.visualization.ColumnChart(
          this.elementRef.nativeElement,
        );
        break;
      case ChartType.COMBO_CHART:
        this.chart = new google.visualization.ComboChart(
          this.elementRef.nativeElement,
        );
        break;
      case ChartType.HISTOGRAM:
        this.chart = new google.visualization.Histogram(
          this.elementRef.nativeElement,
        );
        break;
      case ChartType.LINE_CHART:
        this.chart = new google.visualization.LineChart(
          this.elementRef.nativeElement,
        );
        break;
      case ChartType.PIE_CHART:
        this.chart = new google.visualization.PieChart(
          this.elementRef.nativeElement,
        );
        break;
      case ChartType.SCATTER_CHART:
        this.chart = new google.visualization.ScatterChart(
          this.elementRef.nativeElement,
        );
        break;
      case ChartType.STEPPED_AREA_CHART:
        this.chart = new google.visualization.SteppedAreaChart(
          this.elementRef.nativeElement,
        );
        break;
      case ChartType.TABLE:
        this.chart = new google.visualization.Table(
          this.elementRef.nativeElement,
        );
        break;
      default:
        this.chart = undefined;
        break;
    }
  }

  initDataProvider() {
    if (!this.dataInfo || !this.dataInfo.dataProvider) {
      return;
    }

    if (this.chart) {
      this.dataInfo.dataProvider.setChart(this.chart);
    }
    if (this.dataInfo.data) {
      this.dataInfo.dataProvider.parseData(this.dataInfo.data);
    } else {
      // A shared provider can hold data that arrived before Google Charts did.
      this.dataInfo.dataProvider.flushPendingData?.();
    }
    this.dataInfo.dataProvider.setFilters(this.dataInfo.filters || []);
    this.dataInfo.dataProvider.setUpdateEventListener(() => {
      this.draw();
    });
  }
}

@NgModule({declarations: [Chart], exports: [Chart]})
export class ChartModule {}
