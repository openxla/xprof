import {Component, ElementRef, EventEmitter, inject, Input, NgZone, OnChanges, OnInit, Output, Renderer2, SimpleChanges, ViewChild} from '@angular/core';
import {PIE_CHART_PALETTE} from 'org_xprof/frontend/app/common/constants/roofline_model_constants';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {CategoryTableDataProcessor} from 'org_xprof/frontend/app/components/chart/category_table_data_processor';
import {PIE_CHART_OPTIONS, SCATTER_CHART_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {Table} from 'org_xprof/frontend/app/components/chart/table/table';

type ColumnIdxArr = Array<number|google.visualization.ColumnSpec>;

/**
 * An operation level analysis table view component (step appregation: total).
 */
@Component({
  standalone: false,
  selector: 'operation-level-analysis',
  templateUrl: './operation_level_analysis.ng.html',
  styleUrls: ['./operation_level_analysis.scss'],
})
export class OperationLevelAnalysis extends Dashboard implements OnInit,
                                                                 OnChanges {
  private readonly zone = inject(NgZone);
  /** The roofline model data, original dataset */
  // used for table chart and pie chart
  @Input() rooflineModelData?: google.visualization.DataTable|null = null;
  @Input() viewColumns: ColumnIdxArr = [];
  // data for scatter chart, heavey data preprocessing handled in parent
  @Input() rooflineSeriesData?: google.visualization.DataTable|null = null;
  @Input() scatterChartOptions: google.visualization.ScatterChartOptions = {};
  // Op name prepopulated from url
  @Input() selectedOp = '';
  // Whether the source code service is available.
  @Input() sourceCodeServiceIsAvailable = false;

  @Output()
  readonly filterUpdated =
      new EventEmitter<google.visualization.DataTableCellFilter[]>();

  pieChartDataProvider = new DefaultDataProvider();
  scatterChartDataProvider = new DefaultDataProvider();
  dataInfoCategoryPieChart: ChartDataInfo = {
    data: null,
    dataProvider: this.pieChartDataProvider,
    options: {
      ...PIE_CHART_OPTIONS,
      width: 400,
      height: 400,
      chartArea: {
        width: '70%',
        height: '70%',
      },
      title: 'Percentage of self time per HLO op category',
      colors: PIE_CHART_PALETTE,
      sliceVisibilityThreshold: 0.01,
    },
  };
  dataInfoRooflineScatterChart: ChartDataInfo = {
    data: null,
    dataProvider: this.scatterChartDataProvider,
    options: SCATTER_CHART_OPTIONS,
  };

  @ViewChild('table', {read: Table, static: false})
  tableRef: Table|undefined = undefined;
  @ViewChild('table', {read: ElementRef, static: false})
  chartElementRef: ElementRef|undefined = undefined;
  private readonly renderer: Renderer2 = inject(Renderer2);
  sourceFileAndLineNumber = '';
  stackTrace = '';
  showStackTrace = false;

  constructor() {
    super();
  }

  ngOnInit() {
    this.update();
  }

  private addSourceInfoClickListener() {
    const chart = this.tableRef?.table;
    const chartElement = this.chartElementRef?.nativeElement;
    if (!chart || !chartElement) {
      // TODO: b/429036372 - Using setTimeout to detect change is inefficient.
      setTimeout(() => {
        this.addSourceInfoClickListener();
      }, 100);
      return;
    }
    google.visualization.events.addListener(chart, 'ready', () => {
      this.renderer.listen(chartElement, 'click', (event: Event) => {
        const target = event.target;
        if (target instanceof HTMLElement) {
          if (target.classList.contains('source-info-cell')) {
            this.zone.run(() => {
              this.sourceFileAndLineNumber = target.textContent || '';
              this.stackTrace = target.getAttribute('title') || '';
            });
          }
        }
      });
    });
  }

  toggleShowStackTrace() {
    this.showStackTrace = !this.showStackTrace;
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['sourceCodeServiceIsAvailable'] &&
        this.sourceCodeServiceIsAvailable) {
      this.addSourceInfoClickListener();
    }
    this.update();
  }

  update() {
    this.parseData();
    // call inheried method to update table chart view
    this.updateView();
  }

  override parseData() {
    // base data already preprocessed in parent component
    if (!this.rooflineModelData) {
      return;
    }

    // process data for table chart
    // columns are used in parent logic to set the dataView
    this.columns = this.viewColumns;
    this.dataTable = this.rooflineModelData;

    // process data for pie chart
    this.pieChartDataProvider.parseData(
        JSON.parse(this.dataTable.toJSON()) as SimpleDataTable,
    );
    this.updateAndDrawPieCharts();

    // process data for roofline scatter chart
    if (this.rooflineSeriesData) {
      this.scatterChartDataProvider.parseData(
          JSON.parse(this.rooflineSeriesData.toJSON()) as SimpleDataTable,
      );
      this.updateAndDrawScatterChart();
    }
  }

  /**
   * Triggered when filter update event is emited
   * this is a temp solutino to make other charts view updated as well as the
   * table chart when filters are changed
   * TODO: remove this function when the Dashboard generalization is done
   * building dashboard with multiple charts
   */
  onUpdateFilters(filter: google.visualization.DataTableCellFilter) {
    this.updateFilters(filter);
    this.updateAndDrawPieCharts();
    this.updateAndDrawScatterChart();
    this.filterUpdated.emit(this.getFilters());
  }

  /**
   * Helper functiont to update data for pie chart and refresh view
   * TODO: update either chart component or Dashboard base class to generalize
   * building dashboard with multiple charts this is a temp solutino to make
   */
  updateAndDrawPieCharts() {
    if (!this.dataTable) return;
    const opCategoryIndex = this.dataTable.getColumnIndex('category');
    const opTotalSelfTimeIndex =
        this.dataTable.getColumnIndex('total_self_time');
    this.dataInfoCategoryPieChart.customChartDataProcessor =
        new CategoryTableDataProcessor(
            this.getFilters(),
            opCategoryIndex,
            opTotalSelfTimeIndex,
        );
  }

  updateAndDrawScatterChart() {
    if (!this.rooflineSeriesData) return;
    this.dataInfoRooflineScatterChart.options = Object.assign(
        {},
        this.dataInfoRooflineScatterChart.options,
        this.scatterChartOptions,
    );
    this.dataInfoRooflineScatterChart.dataProvider.notifyCharts();
  }
}
