/// <reference types="google.visualization" />

/** The basic options of Google Charts table. */
export const TABLE_OPTIONS: google.visualization.TableOptions = {
  allowHtml: true,
  alternatingRowStyle: false,
  cssClassNames: {
    'headerCell': 'google-chart-table-header-cell',
    'tableCell': 'google-chart-table-table-cell',
  },
};

/** The basic options of Google Charts pie chart. */
export const PIE_CHART_OPTIONS: google.visualization.PieChartOptions = {
  backgroundColor: 'transparent',
  width: 400,
  height: 200,
  chartArea: {
    left: 0,
    width: '100%',
    height: '80%',
  },
  legend: {textStyle: {fontSize: 10}},
  sliceVisibilityThreshold: 0.01,
};

/** The basic options of Google Charts column chart. */
export const COLUMN_CHART_OPTIONS: google.visualization.ColumnChartOptions = {
  backgroundColor: 'transparent',
  width: 550,
  height: 200,
  chartArea: {
    left: 70,
    top: 10,
    width: '80%',
    height: '80%',
  },
  legend: {position: 'none'},
  tooltip: {
    isHtml: true,
    ignoreBounds: true,
  },
};

/** The basic options of Google Charts bar chart. */
export const BAR_CHART_OPTIONS: google.visualization.BarChartOptions = {
  backgroundColor: 'transparent',
  width: 550,
  height: 200,
  chartArea: {
    left: 70,
    top: 10,
    width: '80%',
    height: '80%',
  },
  legend: {position: 'none'},
  tooltip: {
    isHtml: true,
    ignoreBounds: true,
  },
};

/** The basic options for Google Charts Scatter chart */
export const SCATTER_CHART_OPTIONS: google.visualization.ScatterChartOptions = {
  backgroundColor: 'transparent',
  width: 550,
  height: 200,
  chartArea: {
    left: 70,
    top: 10,
    width: '60%',
    height: '80%',
  },
  legend: {position: 'none'},
  tooltip: {
    isHtml: true,
    ignoreBounds: true,
  },
};
