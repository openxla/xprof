import {DataTableUnion} from './data_table';

/** The enumerator for Google Chart type. */
export enum ChartType {
  UNKNOWN = '',
  AREA_CHART = 'AreaChart',
  BAR_CHART = 'BarChart',
  BUBBLE_CHART = 'BubbleChart',
  CANDLESTICK_CHART = 'CandlestickChart',
  COLUMN_CHART = 'ColumnChart',
  COMBO_CHART = 'ComboChart',
  HISTOGRAM = 'Histogram',
  LINE_CHART = 'LineChart',
  PIE_CHART = 'PieChart',
  SCATTER_CHART = 'ScatterChart',
  STEPPED_AREA_CHART = 'SteppedAreaChart',
  TABLE = 'Table',
}

/** The type for Google Chart class. */
export type ChartClass =
  | google.visualization.AreaChart
  | google.visualization.BarChart
  | google.visualization.BubbleChart
  | google.visualization.CandlestickChart
  | google.visualization.ColumnChart
  | google.visualization.ComboChart
  | google.visualization.Histogram
  | google.visualization.LineChart
  | google.visualization.PieChart
  | google.visualization.ScatterChart
  | google.visualization.SteppedAreaChart
  | google.visualization.Table;

/** All chart options type. */
export type ChartOptions =
  | google.visualization.AreaChartOptions
  | google.visualization.BarChartOptions
  | google.visualization.BubbleChartOptions
  | google.visualization.CandlestickChartOptions
  | google.visualization.ColumnChartOptions
  | google.visualization.ComboChartOptions
  | google.visualization.HistogramOptions
  | google.visualization.LineChartOptions
  | google.visualization.PieChartOptions
  | google.visualization.ScatterChartOptions
  | google.visualization.SteppedAreaChartOptions
  | google.visualization.TableOptions;

/** The data type of DataTable, DataView. */
export type DataTableOrDataView =
  | google.visualization.DataTable
  | google.visualization.DataView;

/** The base interface for an information of chart data. */
export interface ChartDataInfo {
  data: DataTableUnion | Array<Array<string | number>> | null;
  dataProvider: ChartDataProvider;
  filters?: google.visualization.DataTableCellFilter[];
  options?: ChartOptions;
  customChartDataProcessor?: CustomChartDataProcessor;
}

/** The base interface for a chart data provider. */
export interface ChartDataProvider {
  setChart(chart: ChartClass): void;
  // Create a DataTable from JSON data or arrays.
  parseData(data: DataTableUnion | Array<Array<string | number>> | null): void;
  setFilters(filters: google.visualization.DataTableCellFilter[]): void;
  process(): DataTableOrDataView | null;
  // When using the chart function in customChartDataProcessor, get the chart
  // through this function.
  getChart(): ChartClass | null;
  getDataTable(): google.visualization.DataTable | null;
  getOptions(): ChartOptions | null;
  setUpdateEventListener(callback: Function): void;
  setVisibleColumns(visibleColumns: number[]): void;
  notifyCharts(): void;
}

/** The base interface for a class with custom process method. */
export interface CustomChartDataProcessor {
  process(dataProvider: ChartDataProvider): DataTableOrDataView | null;
}
