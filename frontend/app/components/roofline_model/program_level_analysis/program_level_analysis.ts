import {
  ChangeDetectionStrategy,
  Component,
  OnInit,
  effect,
  input,
  output,
} from '@angular/core';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {SCATTER_CHART_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {Chart} from '../../chart/chart';
import {Table} from '../../chart/table/table';
import {CategoryFilter} from '../../controls/category_filter/category_filter';

type ColumnIdxArr = Array<number | google.visualization.ColumnSpec>;

/** An program level analysis table view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'program-level-analysis',
  templateUrl: './program_level_analysis.ng.html',
  styleUrls: ['./program_level_analysis.scss'],
  imports: [CategoryFilter, Chart, Table],
})
export class ProgramLevelAnalysis extends Dashboard implements OnInit {
  /** The roofline model data */
  readonly rooflineModelData = input<google.visualization.DataTable | null>(
    null,
  );
  readonly viewColumns = input<ColumnIdxArr>([]);
  // data for scatter chart, heavey data preprocessing handled in parent
  readonly rooflineSeriesData = input<google.visualization.DataTable | null>(
    null,
  );
  readonly scatterChartOptions =
    input<google.visualization.ScatterChartOptions>({});

  readonly filterUpdated = output<google.visualization.DataTableCellFilter[]>();

  scatterChartDataProvider = new DefaultDataProvider();
  dataInfoRooflineScatterChart: ChartDataInfo = {
    data: null,
    dataProvider: this.scatterChartDataProvider,
    options: {...SCATTER_CHART_OPTIONS, width: 800},
  };

  constructor() {
    super();
    effect(() => {
      this.rooflineModelData();
      this.viewColumns();
      this.rooflineSeriesData();
      this.scatterChartOptions();
      this.update();
    });
  }

  ngOnInit() {
    this.update();
  }

  update() {
    this.parseData();
    this.updateView();
  }

  override parseData() {
    const rooflineModelData = this.rooflineModelData();
    // base data already preprocessed in parent component
    if (!rooflineModelData) {
      return;
    }

    // process data for table chart
    this.columns = this.viewColumns();
    this.dataTable = rooflineModelData;

    // process data for roofline scatter chart
    const rooflineSeriesData = this.rooflineSeriesData();
    if (rooflineSeriesData) {
      this.scatterChartDataProvider.parseData(
        JSON.parse(rooflineSeriesData.toJSON()) as SimpleDataTable,
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
    if (!this.rooflineSeriesData()) return;
    this.dataInfoRooflineScatterChart.options = Object.assign(
      {},
      this.dataInfoRooflineScatterChart.options,
      this.scatterChartOptions(),
    );
    this.dataInfoRooflineScatterChart.dataProvider.notifyCharts();
  }
}
