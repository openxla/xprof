import {
  Component,
  EventEmitter,
  Input,
  OnChanges,
  OnInit,
  Output,
  SimpleChanges,
} from '@angular/core';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {SCATTER_CHART_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {CommonModule} from '@angular/common';
import {CategoryFilter} from 'org_xprof/frontend/app/components/controls/category_filter/category_filter';
import {ChartModule} from 'org_xprof/frontend/app/components/chart/chart';
import {Table} from 'org_xprof/frontend/app/components/chart/table/table';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';

type ColumnIdxArr = Array<number | google.visualization.ColumnSpec>;

/** An program level analysis table view component. */
@Component({
  standalone: true,
  selector: 'program-level-analysis',
  templateUrl: 'program_level_analysis.ng.html',
  styleUrls: ['program_level_analysis.scss'],
  imports: [CommonModule, CategoryFilter, ChartModule, Table],
})
export class ProgramLevelAnalysis
  extends Dashboard
  implements OnInit, OnChanges
{
  /** The roofline model data */
  @Input() rooflineModelData?: google.visualization.DataTable | null = null;
  @Input() viewColumns: ColumnIdxArr = [];
  // data for scatter chart, heavey data preprocessing handled in parent
  @Input() rooflineSeriesData?: google.visualization.DataTable | null = null;
  @Input() scatterChartOptions: google.visualization.ScatterChartOptions = {};

  @Output()
  readonly filterUpdated = new EventEmitter<
    google.visualization.DataTableCellFilter[]
  >();

  scatterChartDataProvider = new DefaultDataProvider();
  dataInfoRooflineScatterChart: ChartDataInfo = {
    data: null,
    dataProvider: this.scatterChartDataProvider,
    options: {...SCATTER_CHART_OPTIONS, width: 800},
  };

  constructor() {
    super();
  }

  ngOnInit() {
    this.update();
  }

  ngOnChanges(changes: SimpleChanges) {
    this.update();
  }

  update() {
    this.parseData();
    this.updateView();
  }

  override parseData() {
    // base data already preprocessed in parent component
    if (!this.rooflineModelData) {
      return;
    }

    // process data for table chart
    this.columns = this.viewColumns;
    this.dataTable = this.rooflineModelData;

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
    this.updateAndDrawScatterChart();
    this.filterUpdated.emit(this.getFilters());
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
