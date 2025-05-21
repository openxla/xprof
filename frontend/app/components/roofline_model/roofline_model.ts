import {Component, inject, OnDestroy, ViewChild} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {DEVICE_INFO, NUMERIC_DATA_FORMAT, PIE_CHART_PALETTE, ROOFLINE_STYLES, SCATTER_CHART_AXIS, SCATTER_CHART_OPTIONS,} from 'org_xprof/frontend/app/common/constants/roofline_model_constants';
import {RooflineModelData} from 'org_xprof/frontend/app/common/interfaces/roofline_model';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {OperationLevelAnalysis} from './operation_level_analysis/operation_level_analysis';
import {ProgramLevelAnalysis} from './program_level_analysis/program_level_analysis';

interface DeviceInfoData {
  id: string;
  label: string;
  type?: string;
  value?: string | number;
  unit?: string;
  context?: string;
  display?: boolean;
}
declare interface DeviceIndicators {
  hasMergedVmem: boolean;
  hasCmem: boolean;
  hasMegacore: boolean;
  isGpu: boolean;
}
type ColumnIdxArr = Array<number | google.visualization.ColumnSpec>;

interface TooltipRow {
  id: string;
  label: string;
  operation?: (val: string | number) => string;
}

const NVIDIA_GPU_TYPE_PREFIX = 'Nvidia GPU';

/** A roofline model component. */
@Component({
  standalone: false,
  selector: 'roofline-model',
  templateUrl: './roofline_model.ng.html',
  styleUrls: ['./roofline_model.scss'],
})
export class RooflineModel implements OnDestroy {
  sessionId = '';
  tool = 'roofline_model';

  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  @ViewChild('programLevelAnalysis')
  programLevelAnalysis?: ProgramLevelAnalysis;
  @ViewChild('opLevelAnalysis') opLevelAnalysis?: OperationLevelAnalysis;

  host = '';
  // Device Information section data
  deviceInfoArray: DeviceInfoData[] = [];
  // Some critical indicators
  deviceIndicators: DeviceIndicators = {
    hasMergedVmem: false,
    hasCmem: false,
    hasMegacore: false,
    isGpu: false,
  };

  // dataTableRaw from the raw roofline model data
  // DataTable data format makes a lot data manipulation easier
  dataTableRaw: google.visualization.DataTable | null = null;

  /** Program level section variables */
  // DataTable data for underlying table chart filtered on category for program
  dataTableProgram: google.visualization.DataTable | null = null;
  // visible columns for the table chart view, if empty all columns are shown
  columnsIdxProgram: ColumnIdxArr = [];
  // preprocessed data for underlying roofline scatter chart
  scatterDataProgram: google.visualization.DataTable | null = null;
  readonly scatterChartOptionsProgram:
      google.visualization.ScatterChartOptions = {
    ...SCATTER_CHART_OPTIONS,
    tooltip: {
      ...(SCATTER_CHART_OPTIONS.tooltip || {}),
      trigger: 'selection',
    },
    series: [],
  };
  readonly programLevelAgg = ['Total', 'Total (HW)', 'Average', 'Step'];

  /** Operation level section variables */
  dataTableOp?: google.visualization.DataTable | null = null;
  columnsIdxOp: ColumnIdxArr = [];
  scatterDataOp?: google.visualization.DataTable | null = null;
  readonly scatterChartOptionsOp: google.visualization.ScatterChartOptions = {
    ...SCATTER_CHART_OPTIONS,
    tooltip: {
      ...(SCATTER_CHART_OPTIONS.tooltip || {}),
      trigger: 'selection',
    },
    series: [],
  };
  // Prepopulated op name from url
  selectedOpName = '';

  constructor(
      route: ActivatedRoute,
      private readonly store: Store<{}>,
  ) {
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.processQuery(params);
      this.update();
    });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  parseUrlParams() {
    this.selectedOpName =
      this.dataService.searchParams?.get('roofline_op_name') || '';
  }

  refreshDashboards() {
    this.programLevelAnalysis?.resetDashboard();
    this.opLevelAnalysis?.resetDashboard();
  }

  processQuery(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || params['tool'] || this.tool;
    this.host = params['host'] || this.host;
  }

  update() {

    setLoadingState(true, this.store, 'Loading roofline model data');
    this.refreshDashboards();

    // get tool data
    this.dataService.getData(this.sessionId, this.tool, this.host)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          setLoadingState(false, this.store);
          this.parseData(data as RooflineModelData[]);
          // TODO(muditgokhale): Add support for roofline model link from trace
          // viewer in 3P. Merge parseUrlParams with processQuery method once
          // done.
          this.parseUrlParams();
        });
  }

  parseData(data?: RooflineModelData[]) {
    if (!google?.visualization) {
      console.log('gviz lib is not loaded yet.');
      setTimeout(() => {
        this.parseData(data);
      }, 100);
      return;
    }
    if (data === null || !Array.isArray(data) || data.length < 1) {
      return;
    }
    this.dataTableRaw = new google.visualization.DataTable(data[0]);

    this.parseDeviceInfoData(this.dataTableRaw);
    this.parseBaseOpAndProgramTableData();

    // process section 1 data
    this.setColumnsIdxProgram();
    this.processScatterDataProgram();

    // process section 2 data
    this.setColumnsIdxOp();
    this.processScatterDataOp();
  }

  /** parse the device information from the original dataset */
  parseDeviceInfoData(dataTableRaw: google.visualization.DataTable) {
    // TODO(b/401596855): Update this to be truthy/falthy check instead of
    // string comparison.
    this.deviceIndicators = {
      hasMergedVmem:
          dataTableRaw.getTableProperties().hasOwnProperty('has_merged_vmem') &&
          !(dataTableRaw.getTableProperty('has_merged_vmem') === '0'),
      hasCmem: dataTableRaw.getTableProperties().hasOwnProperty('has_cmem') &&
          !(dataTableRaw.getTableProperty('has_cmem') === '0'),
      hasMegacore:
          dataTableRaw.getTableProperties().hasOwnProperty('megacore') &&
          !(dataTableRaw.getTableProperty('megacore') === '0'),
      isGpu: dataTableRaw.getTableProperty('device_type')
                 .startsWith(NVIDIA_GPU_TYPE_PREFIX),
    };

    this.deviceInfoArray = DEVICE_INFO.reduce(
      (acc: DeviceInfoData[], cur: DeviceInfoData) => {
        // copy cur to avoid mutating the original object
        // when switch between GPU and TPU runs
        const curInfo = {...cur};
        // deal with category of specific context
        if (this.deviceIndicators.isGpu) {
          if (cur.id === 'peak_flop_rate') {
            curInfo.label = 'Peak FLOP Rate per GPU';
          } else if (cur.id === 'peak_hbm_bw') {
            curInfo.label = 'Peak HBM Bandwidth per GPU';
          } else if (cur.id.startsWith('peak_cmem')) {
            curInfo.display = false;
          } else if (cur.id === 'megacore') {
            curInfo.display = false;
          } else if (cur.id === 'peak_vmem_read_bw') {
            // TODO(b/374835204): Better refactor proto for GPU roofline
            // model and refine related code. including ids like this
            // peak_vmem_read_bw, and peak_vmem_write_bw, megacore, etc.
            curInfo.label = 'Peak L2 cache Bandwidth per GPU';
            curInfo.display = false;
          } else if (cur.id === 'peak_vmem_write_bw') {
            curInfo.label = 'Peak Shared Memory / L1 Cache Bandwidth per GPU';
          }
        } else {
          if (cur.id.startsWith('peak_vmem')) {
            if (!this.deviceIndicators.hasMergedVmem) {
              curInfo.display = false;
            }
          } else if (cur.id.startsWith('peak_cmem')) {
            if (!this.deviceIndicators.hasCmem) {
              curInfo.display = false;
            }
          } else if (cur.id === 'megacore') {
            curInfo.context +=
                '(if yes, the analysis assumes Megacore where an HLO runs on both TensorCores utilizing the full chip\'s resources so that the rooflines are twice higher)';
            curInfo.value = this.deviceIndicators.hasMegacore ? 'Yes' : 'No';
          }
        }
        const value = this.dataTableRaw!.getTableProperty(cur.id);
        acc.push({
          // convert numeric value to numbers, as some ridge numbers will be
          // used as axis values in chart
          value: cur.type === 'number' ? Number(value) : value,
          // put cur at last to overwrite with preprocessed data
          ...curInfo,
        });
        return acc;
      },
      [] as DeviceInfoData[],
    );
  }

  /** Filter and get DataTable data for op and program secions */
  parseBaseOpAndProgramTableData() {
    if (!this.dataTableRaw) {
      return;
    }
    const gViewProgram = new google.visualization.DataView(this.dataTableRaw);
    gViewProgram.setRows(
      this.dataTableRaw.getFilteredRows([
        {
          column: this.dataTableRaw.getColumnIndex('category'),
          value: 'Program',
        },
      ]),
    );
    this.dataTableProgram = gViewProgram.toDataTable();
    this.formatTableData(this.dataTableProgram);

    const gViewOp = new google.visualization.DataView(this.dataTableRaw);
    gViewOp.setRows(
      this.dataTableRaw.getFilteredRows([
        {column: this.dataTableRaw.getColumnIndex('step'), value: 'Total'},
      ]),
    );
    // TODO(b/359276801) Enable injecting Graph Viewer crosslink after
    // dispatching host list to global store, so we can infer module name from
    // program_id given the module list (aka host list in graph viewer)
    const dataTableOp = gViewOp.toDataTable();
    this.dataTableOp =
        this.injectGraphViewerLinksForOpTable(dataTableOp) || null;
    this.formatTableData(this.dataTableOp);
  }

  injectGraphViewerLinksForOpTable(
      dataTableOp: google.visualization.DataTable,
  ) {
    if (!dataTableOp) return;

    const operationIndex = dataTableOp.getColumnIndex('operation');
    const programIdIndex =
        this.dataTableProgram!.getColumnIndex('hlo_module_id');
    const numRows = dataTableOp.getNumberOfRows();
    if (!operationIndex || !programIdIndex || !numRows) return;
    for (let i = 0; i < numRows; ++i) {
      const opName = dataTableOp.getValue(i, operationIndex);
      const programId = dataTableOp.getValue(i, programIdIndex);
      if (!programId || programId === '0' || !opName) continue;
      const graphViewerLink = this.dataService.getGraphViewerLink(
          this.sessionId, '', opName, programId);
      const hyperlinkValue = graphViewerLink ?
          `<a href="${graphViewerLink}" target="_blank">${opName}</a>` :
          opName;
      dataTableOp.setCell(i, operationIndex, hyperlinkValue);
    }
    return dataTableOp;
  }

  /** Get the index array of columns that is visible on the table view */
  getColumnIdx(baseColumnsIds: string[]) {
    const cmemColumnsIds = this.deviceIndicators.hasCmem
      ? ['measured_memory_bw', 'cmem_read_bw', 'cmem_write_bw']
      : [];
    const coreColumnsIds = [
      'roofline_efficiency',
      'compute_efficiency',
      'max_mem_bw_utilization',
    ];
    const columnsIds = [
      ...baseColumnsIds,
      ...cmemColumnsIds,
      ...coreColumnsIds,
    ];

    const getColumnIdxes = (columnIds: string[]) => {
      return columnIds.reduce((acc: ColumnIdxArr, cur: string) => {
        acc.push(this.dataTableRaw!.getColumnIndex(cur));
        return acc;
      }, [] as ColumnIdxArr);
    };
    return getColumnIdxes(columnsIds);
  }

  setColumnsIdxProgram() {
    const baseColumnsIds = [
      'step',
      'total_time_per_core',
      'measured_flop_rate',
      'bound_by',
      'hbm_bw',
    ];
    this.columnsIdxProgram = this.getColumnIdx(baseColumnsIds);
  }

  setColumnsIdxOp() {
    const baseColumnIds = [
      'step',
      'rank',
      'hlo_module_id',
      'category',
      'operation',
      'occurrences',
      'total_time',
      'measured_flop_rate',
      'model_flop_rate',
      'bound_by',
      'hbm_bw',
      'source_info',
    ];
    this.columnsIdxOp = this.getColumnIdx(baseColumnIds);
  }

  formatTableData(data: google.visualization.DataTable | null) {
    if (!data) return;
    let dataFormatter = null;
    for (
      let columnIdx = 0;
      columnIdx < data.getNumberOfColumns();
      ++columnIdx
    ) {
      const id = data.getColumnId(columnIdx);
      const formattedColumnIds = Object.keys(NUMERIC_DATA_FORMAT);
      if (!formattedColumnIds.includes(id)) {
        continue;
      }
      switch (NUMERIC_DATA_FORMAT[id].type) {
        case 'decimal':
          dataFormatter = new google.visualization.NumberFormat({
            fractionDigits: NUMERIC_DATA_FORMAT[id].digit,
          });
          dataFormatter.format(data, columnIdx);
          break;
        case 'percent':
          const pattern = `##.${'#'.repeat(
            NUMERIC_DATA_FORMAT[id].digit || 2,
          )}%`;
          dataFormatter = new google.visualization.NumberFormat({pattern});
          dataFormatter.format(data, columnIdx);
          break;
        default:
          console.log(`Cannot identify format config for column ${id}`);
      }
    }
  }

  /**
   * Helper function to get operation name.
   * It can be either:
   * (1) graph viewer link that encoded with the operation name
   * (2) direct operation name if the crosslink is not implemented
   */
  getOpName(operationValueOrLink: string) {
    const regex = '<a .*>(.*?)</a>';
    const match = operationValueOrLink.match(regex);
    const opName = match?.[1] || operationValueOrLink;
    return this.truncateOperationName(opName);
  }

  /** Helper function to truncate operation name for up to 30 chars */
  truncateOperationName(operationName: string) {
    if (operationName.length > 30) {
      return operationName.substring(0, 30) + '...';
    } else {
      return operationName;
    }
  }

  /**
   * Helper function to add columns to the scatter plot data
   * General for program and operation levels
   * # columns = (1 y value + 1 tooltip) * #series + 1 X axis value
   */
  addScatterDataColumns(
    seriesNames: string[],
    scatterData: google.visualization.DataTable,
  ) {
    // add columns: x axis, series data + corresponding tooltip
    scatterData.addColumn('number', 'Bottleneck Operational Intensity');
    // create 1 value + 1 tooltip column for each series
    seriesNames.forEach((s: string) => {
      scatterData.addColumn('number', s);
      scatterData.addColumn({
        type: 'string',
        role: 'tooltip',
        'p': {'html': true},
      });
    });
  }

  /**
   * Helper function to construct data rows for the scatter chart
   * scatter chart includes the rooflines and other clustered points
   */
  makeScatterRow(
    numColumns: number,
    xIndex: number,
    yIndex: number,
    xVal: number,
    yVal: number,
    tooltip: string,
  ) {
    const newRow = Array.from<number | string | null>({
      length: numColumns,
    }).fill(null);
    newRow[xIndex] = xVal;
    newRow[yIndex] = yVal;
    newRow[yIndex + 1] = tooltip;
    return newRow;
  }

  /** Helper function to add a data row for the scatter chart */
  addSeriesRow(
      sourceDataTable: google.visualization.DataTable,
      scatterDataTable: google.visualization.DataTable,
      rowIndex: number,
      columnIndex: number,
  ) {
    if (rowIndex < 0 || columnIndex < 0) {
      return;
    }
    const numScatterDataColumns = scatterDataTable.getNumberOfColumns();
    const xValue = sourceDataTable.getValue(
      rowIndex,
      sourceDataTable.getColumnIndex('bottleneck_operational_intensity'),
    );
    const yValue = sourceDataTable.getValue(
      rowIndex,
      sourceDataTable.getColumnIndex('measured_flop_rate'),
    );
    // xValue is always assigned to the first column
    // yValue is assigned to the given Step agg level column (columnIdx)
    scatterDataTable.addRow(
      this.makeScatterRow(
        numScatterDataColumns,
        0,
        columnIndex,
        xValue,
        yValue,
        this.makeTooltip(sourceDataTable, rowIndex),
      ),
    );
  }

  /** Helper function to add data rows for a single roofline */
  addRoofline(
      rooflineName: string,
      seriesIndex: number,
      peakFlopRate: number,
      peakMemoryBw: number,
      ridgePoint: number,
      scatterData: google.visualization.DataTable,
  ) {
    if (seriesIndex < 0) {
      return;
    }
    const numColumns = scatterData.getNumberOfColumns();
    // Roofline before the ridge point.
    scatterData.addRow(
      this.makeScatterRow(
        numColumns,
        0,
        seriesIndex,
        SCATTER_CHART_AXIS.minX,
        SCATTER_CHART_AXIS.minX * peakMemoryBw,
        this.makeRooflineTooltip(
          'Roofline',
          SCATTER_CHART_AXIS.minX,
          SCATTER_CHART_AXIS.minX * peakMemoryBw,
        ),
      ),
    );
    // Ridge point.
    scatterData.addRow(
      this.makeScatterRow(
        numColumns,
        0,
        seriesIndex,
        ridgePoint,
        peakFlopRate,
        this.makeRooflineTooltip(
          rooflineName + ' Ridge Point',
          ridgePoint,
          peakFlopRate,
        ),
      ),
    );
    // Roofline after the ridge point.
    scatterData.addRow(
      this.makeScatterRow(
        numColumns,
        0,
        seriesIndex,
        SCATTER_CHART_AXIS.maxX,
        peakFlopRate,
        this.makeRooflineTooltip(
          'Roofline',
          SCATTER_CHART_AXIS.maxX,
          peakFlopRate,
        ),
      ),
    );
  }

  /** Callback function when filterUpdated in child is triggered */
  updateDataTableOp(newFilters: google.visualization.DataTableCellFilter[]) {
    this.processScatterDataOp(newFilters);
  }

  /** Callback function when filterUpdated in child is triggered */
  updateDataTableProgram(
    newFilters: google.visualization.DataTableCellFilter[],
  ) {
    this.processScatterDataProgram(newFilters);
  }

  /**
   * Parse dataset for program level roofline scatter chart
   * With series of data, operation scatter plot =
   * rooflines (line) plot + program level step cluster(scatter) plot
   */
  processScatterDataProgram(
    filters?: google.visualization.DataTableCellFilter[],
  ) {
    if (!this.dataTableProgram) {
      return;
    }
    const filteredDataTableProgram = this.getFilteredDataTable(
      this.dataTableProgram,
      filters,
    );
    // TODO: update the programSeries based on data received
    const programSeries = this.getProgramSeries();
    // clear and recreate the scatter data
    this.scatterDataProgram = new google.visualization.DataTable();
    this.addScatterDataColumns(programSeries, this.scatterDataProgram);
    this.addRooflinesSeriesRows(this.scatterDataProgram);
    this.addProgramSeriesRows(programSeries, filteredDataTableProgram);
    this.updateProgramScatterStyles(programSeries.length);
  }

  /**
   * Parse dataset for operation level roofline scatter chart
   * With series of data, operation scatter plot =
   * rooflines (line) plot + op categoreis cluster(scatter) plot
   */
  processScatterDataOp(filters?: google.visualization.DataTableCellFilter[]) {
    if (!this.dataTableOp) {
      return;
    }
    const filteredDataTableOp = this.getFilteredDataTable(
      this.dataTableOp,
      filters,
    );

    const opCategories = this.getOpCategories(filteredDataTableOp);
    const opSeries = this.getOpSeries(opCategories);

    // clear the original scatter data
    this.scatterDataOp = new google.visualization.DataTable();
    this.addScatterDataColumns(opSeries, this.scatterDataOp);
    this.addRooflinesSeriesRows(this.scatterDataOp);
    this.addOpSeriesRows(opSeries, filteredDataTableOp);
    this.updateOpScatterStyles(opSeries.length);
  }

  /**
   * Helper function to get filtered DataTable given base op/proram DataTable,
   * and feed to child component as source data for roofline scatter chart.
   * Because scatter chart DataTable is in a different structure than the table
   * chart DataTable.
   * Filteres are passed from child filters.
   */
  getFilteredDataTable(
    dataTable: google.visualization.DataTable,
    filters?: google.visualization.DataTableCellFilter[],
  ) {
    // apply filters if any, filters are emitted from child component
    // because the scatter dataTable is restructured and cannot be applied in
    // child directly
    let filteredDataTable: google.visualization.DataTable | null = null;
    if (filters && filters.length > 0) {
      const filteredDataView = new google.visualization.DataView(dataTable);
      filteredDataView.setRows(dataTable.getFilteredRows(filters));
      filteredDataTable = filteredDataView.toDataTable();
    } else {
      filteredDataTable = dataTable;
    }
    return filteredDataTable;
  }

  /**
   * Helper function to get operation categories, with data filtered on
   * "step == Total", the list is sorted by total_self_time in order to make the
   * scatter chart style in consistent with the pie chart
   */
  getOpCategories(filteredDataTableOp: google.visualization.DataTable) {
    const sortedOpCategories: string[] = [];

    // sort the categories given frequency
    const chartView = google.visualization.data.group(
      filteredDataTableOp,
      [filteredDataTableOp.getColumnIndex('category')],
      [
        {
          'column': filteredDataTableOp.getColumnIndex('total_self_time'),
          'aggregation': google.visualization.data.sum,
          'type': 'number',
        },
      ],
    );
    // sort categories on sum of total_self_time
    chartView.sort({column: 1, desc: true});
    for (let i = 0; i < chartView.getNumberOfRows(); ++i) {
      const category = chartView.getValue(i, 0);
      // Program will be appended separately
      if (category !== 'Program') {
        sortedOpCategories.push(category);
      }
    }
    return sortedOpCategories;
  }

  /**
   * The roofline chart consists of a seris of data (roofline series +
   * appregation series)
   * This helper function gets the roofline base series
   */
  getRooflineBaseSeries() {
    let series: string[] = [];
    if (this.deviceIndicators.isGpu) {
      series = series.concat(['Shared Mem / L1 Roofline']);
    } else {
      if (this.deviceIndicators.hasMergedVmem) {
        series = series.concat(['VMEM Read Roofline', 'VMEM Write Roofline']);
      } else if (this.deviceIndicators.hasCmem) {
        series = series.concat(['CMEM Read Roofline', 'CMEM Write Roofline']);
      }
    }
    return [...series, 'HBM Roofline'];
  }

  /**
   * #series = #roofline(line) + 4 program level aggregation series
   */
  getProgramSeries() {
    let series: string[] = this.getRooflineBaseSeries();
    series = series.concat(this.programLevelAgg);
    return series;
  }

  /**
   * #series = #roofline(line) + #operation level aggregation series
   * (categories) + 2 'Program' datapoints
   * The sereis list will decide the style of the scatter chart
   */
  getOpSeries(opCategories: string[]) {
    let series: string[] = this.getRooflineBaseSeries();
    // the first program is to make it's legend shows on top
    // the second program is to show marker on top layer on the chart
    series = series.concat(['Program', ...opCategories, 'Program']);
    return series;
  }

  /**
   * Helper function to add data rows for roofline plot - vmem, cmem, hbm
   * generalized function for both op & program
   */
  addRooflinesSeriesRows(scatterData: google.visualization.DataTable) {
    const rooflineInfo = this.deviceInfoArray.reduce(
      (acc, item) => {
        acc[item.id] = Number(item.value || 0);
        return acc;
      },
      {} as {[key: string]: number},
    );
    let columnIndex = 1;

    if (!this.deviceIndicators.isGpu) {
      const addRooflinePairs = (memType: 'cmem' | 'vmem') => {
        ['read', 'write'].forEach((opType) => {
          this.addRoofline(
            `${memType.toUpperCase()} ${opType.charAt(0).toUpperCase() + opType.slice(1)}`,
            columnIndex,
            rooflineInfo['peak_flop_rate'],
            rooflineInfo[`peak_${memType}_${opType}_bw`],
            rooflineInfo[`${memType}_${opType}_ridge_point`],
            scatterData,
          );
          columnIndex += 2; // value col + tooltip col
        });
      };
      if (this.deviceIndicators.hasMergedVmem) {
        addRooflinePairs('vmem');
      }
      if (this.deviceIndicators.hasCmem) {
        addRooflinePairs('cmem');
      }
    } else {
      // Just use vmem_read for gpu SHM/L1
      this.addRoofline(
        'Shared Mem / L1',
        columnIndex,
        rooflineInfo['peak_flop_rate'],
        rooflineInfo['peak_vmem_write_bw'],
        rooflineInfo['vmem_write_ridge_point'],
        scatterData,
      );
      columnIndex += 2; // value col + tooltip col
    }

    this.addRoofline(
      'HBM',
      columnIndex,
      rooflineInfo['peak_flop_rate'],
      rooflineInfo['peak_hbm_bw'],
      rooflineInfo['hbm_ridge_point'],
      scatterData,
    );
  }

  /**
   * Poluplate program level scatter chart data rows with series using filtered
   * operation DataTable data
   */
  addProgramSeriesRows(
    programSeries: string[],
    filteredDataTableProgram: google.visualization.DataTable,
  ) {
    for (
      let rowIndex = 0;
      rowIndex < filteredDataTableProgram.getNumberOfRows();
      ++rowIndex
    ) {
      let step = filteredDataTableProgram.getValue(
        rowIndex,
        filteredDataTableProgram.getColumnIndex('step'),
      );
      // Assgin 'Step' as value if the step field is numeric string
      if (!this.programLevelAgg.includes(step)) {
        step = 'Step';
      }
      const columnIndex = 1 + 2 * programSeries.lastIndexOf(step);
      this.addSeriesRow(
        filteredDataTableProgram,
        this.scatterDataProgram!,
        rowIndex,
        columnIndex,
      );
    }
  }

  /**
   * Poluplate operation level scatter chart data rows with series using
   * filtered operation DataTable data
   */
  addOpSeriesRows(
    opSeries: string[],
    filteredDataTableOp: google.visualization.DataTable,
  ) {
    for (
      let rowIndex = 0;
      rowIndex < filteredDataTableOp.getNumberOfRows();
      ++rowIndex
    ) {
      const category = filteredDataTableOp.getValue(
        rowIndex,
        filteredDataTableOp.getColumnIndex('category'),
      );
      const columnIndex = 1 + 2 * opSeries.lastIndexOf(category);
      if (
        columnIndex > 0 &&
        filteredDataTableOp.getValue(
          rowIndex,
          filteredDataTableOp.getColumnIndex('bound_by'),
        ) !== 'Unknown'
      ) {
        this.addSeriesRow(
          filteredDataTableOp,
          this.scatterDataOp!,
          rowIndex,
          columnIndex,
        );
      }
    }
  }

  /** Make tooltip for rooflines series in the scatter chart */
  makeRooflineTooltip(
    rooflineName: string,
    operationIntensity: number,
    flopRate: number,
  ) {
    return (
      '<div style="padding:5px;">' +
      '<b>' +
      rooflineName +
      '</b><br/>' +
      '<b>Operational Intensity (FLOP/Byte): </b>' +
      operationIntensity.toLocaleString(undefined, {maximumFractionDigits: 2}) +
      '<br/>' +
      '<b>Flop Rate (GFLOP/s): </b>' +
      flopRate.toLocaleString(undefined, {maximumFractionDigits: 2}) +
      '<br/>' +
      '</div>'
    );
  }

  /** Make tooltip for the clustered series (points) in the scatter chart */
  makeTooltip(dataTable: google.visualization.DataTable, rowIndex: number) {
    // Prepare column index to make easier access
    const columns: {[columnKey: string]: number} = {};
    for (let i = 0; i < dataTable.getNumberOfColumns(); i++) {
      columns[dataTable.getColumnId(i)] = i;
    }
    // TODO(jihochoi): fix the utilization numbers for TPU V4.
    // '<b> - Percent relative to optimal: </b>'
    // + (100 * dataTable.getValue(rowIndex,
    // columns.roofline_efficiency)).toLocaleString(undefined, {maximumFractionDigits:2})
    // + '%<br/>' +
    // '<b> - Percent relative to HW limit: </b>'
    // + (100 * dataTable.getValue(rowIndex,
    // columns.compute_efficiency)).toLocaleString(undefined, {maximumFractionDigits:2})
    // + '%<br/>' +
    // '<b> - Percent relative to HW limit: </b>'
    // + (100 * dataTable.getValue(rowIndex,
    // columns.hbm_bw_utilization)).toLocaleString(undefined, {maximumFractionDigits:2})
    // + '%<br/>' +
    const tooltipRows: TooltipRow[] = [
      {
        id: 'step',
        label: 'Step',
      },
      {
        id: 'rank',
        label: 'Rank',
      },
      {
        id: 'hlo_module_id',
        label: 'Program ID',
      },
      {
        id: 'category',
        label: 'Category',
      },
      {
        id: 'operation',
        label: 'Operation',
        operation: (val) => this.getOpName(val as string),
      },
      {
        id: 'occurrences',
        label: '# of Occurrences',
      },
      {
        id: 'total_time_per_core',
        label: 'Total Time per core (us)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 2}),
      },
      {
        id: 'total_time_in_percentage',
        label: 'Total Time / Program',
        operation: (val) => `${100 * Number(Number(val).toFixed(4))}%`,
      },
      {
        id: 'measured_flop_rate',
        label: 'Normalized FLOP Rate (GFLOP/s)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'model_flop_rate',
        label: 'Model FLOP Rate (GFLOP/s)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'hbm_bw',
        label: 'HBM BW (GiB/s)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'cmem_read_bw',
        label: 'CMEM Read BW (GiB/s)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'cmem_write_bw',
        label: 'CMEM Write BW (GiB/s)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'vmem_read_bw',
        label: 'VMEM Read BW (GiB/s)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'vmem_write_bw',
        label: 'VMEM Write BW (GiB/s)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'operational_intensity',
        label: 'Operational Intensity (FLOP/Byte)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'hbm_operational_intensity',
        label: 'HBM Operational Intensity (FLOP/Byte)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'cmem_read_operational_intensity',
        label: 'CMEM Read Operational Intensity (FLOP/Byte)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'cmem_write_operational_intensity',
        label: 'CMEM Write Operational Intensity (FLOP/Byte)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'vmem_read_operational_intensity',
        label: 'VMEM Read Operational Intensity (FLOP/Byte)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'vmem_write_operational_intensity',
        label: 'VMEM Write Operational Intensity (FLOP/Byte)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'bottleneck_operational_intensity',
        label: 'Bottleneck Operational Intensity (FLOP/Byte)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {id: 'boundy_by', label: 'Bound By'},
    ];
    const gpuTooltipRows: TooltipRow[] = [
      {
        id: 'step',
        label: 'Step',
      },
      {
        id: 'rank',
        label: 'Rank',
      },
      {
        id: 'hlo_module_id',
        label: 'Program ID',
      },
      {
        id: 'category',
        label: 'Category',
      },
      {
        id: 'operation',
        label: 'Operation',
        operation: (val) => this.getOpName(val as string),
      },
      {
        id: 'occurrences',
        label: '# of Occurrences',
      },
      {
        id: 'total_time_per_core',
        label: 'Total Time per gpu (us)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 2}),
      },
      {
        id: 'total_time_in_percentage',
        label: 'Total Time / Program',
        operation: (val) => `${100 * Number(Number(val).toFixed(4))}%`,
      },
      {
        id: 'measured_flop_rate',
        label: 'Normalized FLOP Rate (GFLOP/s)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'model_flop_rate',
        label: 'Model FLOP Rate (GFLOP/s)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'hbm_bw',
        label: 'HBM BW (GiB/s)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'vmem_write_bw',
        label: 'Shm/L1 BW (GiB/s)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'operational_intensity',
        label: 'Operational Intensity (FLOP/Byte)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'hbm_operational_intensity',
        label: 'HBM Operational Intensity (FLOP/Byte)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {
        id: 'bottleneck_operational_intensity',
        label: 'Bottleneck Operational Intensity (FLOP/Byte)',
        operation: (val) =>
          val.toLocaleString(undefined, {maximumFractionDigits: 4}),
      },
      {id: 'boundy_by', label: 'Bound By'},
    ];

    const tooltipBodyHtml = (
      this.deviceIndicators.isGpu ? gpuTooltipRows : tooltipRows
    ).reduce((acc: string, row: TooltipRow) => {
      if (!columns.hasOwnProperty(row.id)) {
        return acc;
      }
      const val: string | number = dataTable.getValue(
        rowIndex,
        columns[row.id],
      );
      acc += `<b>${row.label}: </b> ${
        row.operation ? row.operation(val) : val
      }<br>`;
      return acc;
    }, '');
    return `<div style="padding: 5px">${tooltipBodyHtml}</div>`;
  }

  // TODO(yinzz) remove the style updating dependency on the series order
  // make it a k-v based format
  formatRooflineSeriesStyle(
    seriesIndex: number,
    chartOptions: google.visualization.ScatterChartOptions,
  ) {
    if (this.deviceIndicators.isGpu) {
      chartOptions.series[seriesIndex++] = ROOFLINE_STYLES.write;
    } else {
      if (this.deviceIndicators.hasMergedVmem) {
        chartOptions.series[seriesIndex++] = ROOFLINE_STYLES.read;
        chartOptions.series[seriesIndex++] = ROOFLINE_STYLES.write;
      }
      if (this.deviceIndicators.hasCmem) {
        chartOptions.series[seriesIndex++] = ROOFLINE_STYLES.read;
        chartOptions.series[seriesIndex++] = ROOFLINE_STYLES.write;
      }
    }
    chartOptions.series[seriesIndex++] = ROOFLINE_STYLES.hbm;
    return seriesIndex;
  }

  updateProgramScatterStyles(numSeries: number) {
    let seriesIndex = 0;
    seriesIndex = this.formatRooflineSeriesStyle(
      seriesIndex,
      this.scatterChartOptionsProgram,
    );
    for (; seriesIndex < numSeries; ++seriesIndex) {
      this.scatterChartOptionsProgram.series[seriesIndex] = {pointSize: 4};
    }
  }

  // TODO(yinzz) remove the style updating dependency on the series order
  updateOpScatterStyles(numSeries: number) {
    let seriesIndex = 0;
    seriesIndex = this.formatRooflineSeriesStyle(
      seriesIndex,
      this.scatterChartOptionsOp,
    );
    // extra series style record for the Program legend
    this.scatterChartOptionsOp.series[seriesIndex++] = {
      pointSize: 20,
      color: '#FF0000',
      pointShape: 'star',
    };
    // Other ops are colored in the same order as in the pie chart, cmem, vmem,
    // hbm, program
    const numSeriesBeforeOps =
      2 * (this.deviceIndicators.hasCmem ? 1 : 0) +
      2 * (this.deviceIndicators.hasMergedVmem ? 1 : 0) +
      2;
    for (; seriesIndex < numSeries - 1; ++seriesIndex) {
      this.scatterChartOptionsOp.series[seriesIndex] = {
        pointSize: 3,
        // make sure the color of series matches the pie chart
        color:
          PIE_CHART_PALETTE[
            (seriesIndex - numSeriesBeforeOps) % PIE_CHART_PALETTE.length
          ],
      };
    }
    // Real series for program which does not show in the legend.
    // This is added at the end to make it plotted at the top and not buried by
    // other op points.
    this.scatterChartOptionsOp.series[numSeries - 1] = {
      pointSize: 20,
      color: '#FF0000',
      pointShape: 'star',
      visibleInLegend: false,
    };
  }

  ngOnDestroy() {
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
