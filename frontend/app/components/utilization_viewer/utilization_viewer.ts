import {
  ChangeDetectionStrategy,
  Component,
  inject,
  OnDestroy,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {
  ChartDataInfo,
  ChartType,
} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {alignTables} from 'org_xprof/frontend/app/common/utils/diff_utils';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {
  BAR_CHART_OPTIONS,
  PIE_CHART_OPTIONS,
} from 'org_xprof/frontend/app/components/chart/chart_options';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {FilterDataProcessor} from 'org_xprof/frontend/app/components/chart/filter_data_processor';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {BaseDiffService} from 'org_xprof/frontend/app/services/data_service_v2/diff_service';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

const UNIT_CHART_OPTIONS: google.visualization.BarChartOptions = {
  ...BAR_CHART_OPTIONS,
  width: 800,
  height: 700,
  chartArea: {left: '25%', width: '65%', height: 650},
  hAxis: {format: 'percent', minValue: 0.0, maxValue: 1.0},
};

const UNIT_CHART_OPTIONS_BASELINE: google.visualization.BarChartOptions = {
  ...UNIT_CHART_OPTIONS,
  height: 1050,
  chartArea: {...UNIT_CHART_OPTIONS.chartArea, height: 975},
  colors: ['#1a73e8', '#adc6fa'],
  legend: {position: 'top'},
};

const BANDWIDTH_CHART_OPTIONS: google.visualization.BarChartOptions = {
  ...BAR_CHART_OPTIONS,
  width: 800,
  height: 400,
  chartArea: {left: '25%', width: '65%', height: 300},
  hAxis: {format: 'percent', minValue: 0.0, maxValue: 1.0},
};

const BANDWIDTH_CHART_OPTIONS_BASELINE: google.visualization.BarChartOptions = {
  ...BANDWIDTH_CHART_OPTIONS,
  height: 600,
  chartArea: {...BANDWIDTH_CHART_OPTIONS.chartArea, height: 450},
  colors: ['#1a73e8', '#adc6fa'],
  legend: {position: 'top'},
};

const HBM_CHART_OPTIONS: google.visualization.PieChartOptions = {
  ...PIE_CHART_OPTIONS,
  width: 800,
  height: 400,
};

const CORE_ID = 'node';
const NAME_ID = 'name';
const ACHIEVED_ID = 'achieved';
const PEAK_ID = 'peak';
const UNIT_ID = 'unit';
const HBM_READ_RATIO_NAME = 'HBM Read Ratio';
const HBM_WRITE_RATIO_NAME = 'HBM Write Ratio';

declare interface NodeChartDataInfoMap {
  [index: number]: ChartDataInfo;
}

declare interface NodeFilterDataProcessorMap {
  [index: number]: FilterDataProcessor | null;
}

/**
 * Utilization viewer component.
 * The utilization viewer displays unit and bandwidth utiization for each tensor
 * node in a TPU chip.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'utilization-viewer',
  templateUrl: './utilization_viewer.ng.html',
  styleUrls: ['./utilization_viewer.scss'],
  providers: [BaseDiffService],
})
export class UtilizationViewer extends Dashboard implements OnDestroy {
  readonly tool = 'utilization_viewer';
  readonly ChartType = ChartType;
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly throbber = new Throbber(this.tool);

  sessionId = '';
  host = '';
  dataService: DataServiceV2Interface = inject(DATA_SERVICE_INTERFACE_TOKEN);
  diffService: BaseDiffService = inject(BaseDiffService);
  dataProvider = new DefaultDataProvider();
  dataInfoTensorNodesUnit: Partial<NodeChartDataInfoMap> = {};
  dataProcessorTensorNodesUnit: Partial<NodeFilterDataProcessorMap> = {};
  dataInfoTensorNodesBandwidth: Partial<NodeChartDataInfoMap> = {};
  dataProcessorTensorNodesBandwidth: Partial<NodeFilterDataProcessorMap> = {};
  dataInfoHBMRatio: Partial<NodeChartDataInfoMap> = {};
  dataProcessorHBMRatio: Partial<NodeFilterDataProcessorMap> = {};
  coreIndexes: number[] = [];
  hbmCoreIndexes: number[] = [];

  constructor(
    route: ActivatedRoute,
    private readonly store: Store<{}>,
  ) {
    super();
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.sessionId = (params || {})['sessionId'] || '';
    });
    combineLatest([route.params, route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        this.sessionId = params['sessionId'] || this.sessionId;
        this.processQueryParams(queryParams);
        this.update();
      });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  processQueryParams(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.host = params['host'] || this.host;
  }

  update() {
    setLoadingState(true, this.store, 'Loading utilization viewer data');
    this.throbber.start();

    this.diffService
      .getDiffData(this.sessionId, this.tool)
      .pipe(takeUntil(this.destroyed))
      .subscribe(({active, baseline}) => {
        this.throbber.stop();
        setLoadingState(false, this.store);
        this.parseData(
          this.mergeTables(
            active as SimpleDataTable | null,
            baseline as SimpleDataTable | null,
          ),
        );
      });
  }

  initDataProcessors(hasBaseline = false) {
    const unitOptions = hasBaseline
      ? UNIT_CHART_OPTIONS_BASELINE
      : UNIT_CHART_OPTIONS;
    const bandwidthOptions = hasBaseline
      ? BANDWIDTH_CHART_OPTIONS_BASELINE
      : BANDWIDTH_CHART_OPTIONS;
    this.coreIndexes.forEach((index: number) => {
      this.dataInfoTensorNodesUnit[index] = {
        data: null,
        dataProvider: this.dataProvider,
        options: unitOptions,
      };
      this.dataProcessorTensorNodesUnit[index] = null;
      this.dataInfoTensorNodesBandwidth[index] = {
        data: null,
        dataProvider: this.dataProvider,
        options: bandwidthOptions,
      };
      this.dataProcessorTensorNodesBandwidth[index] = null;
      this.dataInfoHBMRatio[index] = {
        data: null,
        dataProvider: this.dataProvider,
        options: HBM_CHART_OPTIONS,
      };
      this.dataProcessorHBMRatio[index] = null;
    });
  }

  updateDataProcessors(
    visibleColumns: Array<number | google.visualization.ColumnSpec>,
    coreCol: number,
    unitCol: number,
  ) {
    this.coreIndexes.forEach((index: number) => {
      if (this.dataInfoTensorNodesUnit[index] !== undefined) {
        this.dataInfoTensorNodesUnit[index]!.customChartDataProcessor =
          this.dataProcessorTensorNodesUnit[index] = new FilterDataProcessor(
            visibleColumns,
            [
              {column: coreCol, value: index},
              // A range filter intended to include "cycles" and
              // "instructions", but exclude "bytes"
              {
                column: unitCol,
                minValue: 'cycles',
                maxValue: 'instructions',
              },
            ],
          );
      }
      if (this.dataInfoTensorNodesBandwidth.hasOwnProperty(index)) {
        this.dataInfoTensorNodesBandwidth[index]!.customChartDataProcessor =
          this.dataProcessorTensorNodesBandwidth[index] =
            new FilterDataProcessor(visibleColumns, [
              {column: coreCol, value: index},
              {column: unitCol, value: 'bytes'},
            ]);
      }
    });
  }

  updateHBMDataProcessors(
    nameCol: number,
    achievedCol: number,
    coreCol: number,
  ) {
    const visibleColumns = [nameCol, achievedCol];
    this.hbmCoreIndexes.forEach((index: number) => {
      this.dataInfoHBMRatio[index]!.customChartDataProcessor =
        this.dataProcessorHBMRatio[index] = new FilterDataProcessor(
          visibleColumns,
          [
            {column: coreCol, value: index},
            // A range filter to select rows with HBM Read Ratio and HBM Write Ratio.
            {
              column: nameCol,
              minValue: HBM_READ_RATIO_NAME,
              maxValue: HBM_WRITE_RATIO_NAME,
            },
          ],
        );
    });
  }

  mergeTables(
    active: SimpleDataTable | null,
    baseline: SimpleDataTable | null,
  ): SimpleDataTable | null {
    if (!active) return null;
    if (!baseline) return active;

    const activeCols = active.cols || [];
    const activeRows = active.rows || [];
    const baselineRows = baseline.rows || [];

    const getColIndex = (id: string) =>
      activeCols.findIndex((c) => c.id === id);
    const coreCol = getColIndex(CORE_ID);
    const nameCol = getColIndex(NAME_ID);
    const achievedCol = getColIndex(ACHIEVED_ID);
    const peakCol = getColIndex(PEAK_ID);
    const unitCol = getColIndex(UNIT_ID);
    const hostCol = getColIndex('host');
    const deviceCol = getColIndex('device');
    const sampleCol = getColIndex('sample');

    if (
      coreCol === -1 ||
      nameCol === -1 ||
      achievedCol === -1 ||
      peakCol === -1 ||
      unitCol === -1
    ) {
      return active;
    }

    const getRowKey = (row: google.visualization.DataObjectRow) => {
      const host = hostCol !== -1 ? (row.c?.[hostCol]?.v ?? '') : '';
      const device = deviceCol !== -1 ? (row.c?.[deviceCol]?.v ?? '') : '';
      const sample = sampleCol !== -1 ? (row.c?.[sampleCol]?.v ?? '') : '';
      const node = row.c?.[coreCol]?.v ?? '';
      const name = row.c?.[nameCol]?.v ?? '';
      const unit = row.c?.[unitCol]?.v ?? '';
      return `${host}_${device}_${sample}_${node}_${name}_${unit}`;
    };

    const aligned = alignTables(activeRows, baselineRows, getRowKey);

    const newCols = [...activeCols];
    const baselineAchievedColIndex = newCols.length;
    newCols.push({
      id: 'achieved_baseline',
      label: 'Baseline Achieved',
      type: 'number',
    });
    const baselinePeakColIndex = newCols.length;
    newCols.push({
      id: 'peak_baseline',
      label: 'Baseline Peak',
      type: 'number',
    });

    const newRows: google.visualization.DataObjectRow[] = [];

    for (const compRow of aligned.values()) {
      const activeRow = compRow.active;
      const baselineRow = compRow.baseline;

      if (activeRow) {
        const newCells = activeRow.c ? [...activeRow.c] : [];
        newCells[baselineAchievedColIndex] = {
          v: baselineRow?.c?.[achievedCol]?.v ?? 0,
          f: baselineRow?.c?.[achievedCol]?.f,
        };
        newCells[baselinePeakColIndex] = {
          v: baselineRow?.c?.[peakCol]?.v ?? 0,
          f: baselineRow?.c?.[peakCol]?.f,
        };
        newRows.push({c: newCells});
      } else if (baselineRow) {
        const newCells = baselineRow.c ? [...baselineRow.c] : [];
        newCells[achievedCol] = {v: 0};
        newCells[peakCol] = {v: 0};

        newCells[baselineAchievedColIndex] = {
          v: baselineRow.c?.[achievedCol]?.v ?? 0,
          f: baselineRow.c?.[achievedCol]?.f,
        };
        newCells[baselinePeakColIndex] = {
          v: baselineRow.c?.[peakCol]?.v ?? 0,
          f: baselineRow.c?.[peakCol]?.f,
        };
        newRows.push({c: newCells});
      }
    }

    return {
      cols: newCols,
      rows: newRows,
      p: active.p,
    };
  }

  override parseData(data: SimpleDataTable | null) {
    if (!data) return;
    this.dataProvider.parseData(data);
    const dataTable = this.dataProvider.getDataTable();
    if (dataTable) {
      this.dataTable = dataTable;

      const coreCol = dataTable.getColumnIndex(CORE_ID);
      const nameCol = dataTable.getColumnIndex(NAME_ID);
      const achievedCol = dataTable.getColumnIndex(ACHIEVED_ID);
      const peakCol = dataTable.getColumnIndex(PEAK_ID);
      const unitCol = dataTable.getColumnIndex(UNIT_ID);
      this.coreIndexes = dataTable.getDistinctValues(coreCol);

      // Determine which cores have HBM Read/Write Ratio data.
      const hbmCoreSet = new Set<number>();
      for (let i = 0; i < dataTable.getNumberOfRows(); i++) {
        const name = dataTable.getValue(i, nameCol);
        if (name === HBM_READ_RATIO_NAME || name === HBM_WRITE_RATIO_NAME) {
          const coreIndex = dataTable.getValue(i, coreCol);
          hbmCoreSet.add(coreIndex);
        }
      }
      this.hbmCoreIndexes = Array.from(hbmCoreSet).sort((a, b) => a - b);

      const hasBaseline = !!this.diffService.getBaseSessionId() &&
        dataTable.getColumnIndex('achieved_baseline') !== -1;
      const baselineAchievedCol = dataTable.getColumnIndex('achieved_baseline');
      const baselinePeakCol = dataTable.getColumnIndex('peak_baseline');

      const visibleColumns: Array<number | google.visualization.ColumnSpec> = [
        nameCol,
        {
          calc: (data: google.visualization.DataTable, row: number) => {
            const achieved = data.getValue(row, achievedCol);
            const peak = data.getValue(row, peakCol);
            return peak !== 0 ? achieved / peak : 0;
          },
          type: 'number',
          label: hasBaseline ? 'Active % Active' : '% Active',
          id: 'active',
        },
        {
          calc: (data: google.visualization.DataTable, row: number) => {
            const achieved = data.getValue(row, achievedCol);
            if (achieved === 0) return undefined;
            const peak = data.getValue(row, peakCol);
            return ((100 * achieved) / peak).toFixed(2) + '%';
          },
          type: 'string',
          role: 'annotation',
        },
      ];

      if (hasBaseline) {
        visibleColumns.push({
          calc: (data: google.visualization.DataTable, row: number) => {
            const achieved = data.getValue(row, baselineAchievedCol);
            const peak = data.getValue(row, baselinePeakCol);
            return peak !== 0 ? achieved / peak : 0;
          },
          type: 'number',
          label: 'Baseline % Active',
          id: 'baseline',
        });
        visibleColumns.push({
          calc: (data: google.visualization.DataTable, row: number) => {
            const achieved = data.getValue(row, baselineAchievedCol);
            if (achieved === 0) return undefined;
            const peak = data.getValue(row, baselinePeakCol);
            return ((100 * achieved) / peak).toFixed(2) + '%';
          },
          type: 'string',
          role: 'annotation',
        });
      }

      visibleColumns.push({
        calc: (data: google.visualization.DataTable, row: number) => {
          const achieved = data.getValue(row, achievedCol);
          const peak = data.getValue(row, peakCol);
          const unit = data.getValue(row, unitCol);
          let tooltip = `Active Achieved: ${achieved.toLocaleString()} ${unit} (Peak: ${peak.toLocaleString()} ${unit})`;
          if (hasBaseline) {
            const baseAchieved = data.getValue(row, baselineAchievedCol);
            const basePeak = data.getValue(row, baselinePeakCol);
            tooltip += `\nBaseline Achieved: ${baseAchieved.toLocaleString()} ${unit} (Peak: ${basePeak.toLocaleString()} ${unit})`;
          }
          return tooltip;
        },
        type: 'string',
        role: 'tooltip',
      });

      this.initDataProcessors(hasBaseline);
      this.updateDataProcessors(visibleColumns, coreCol, unitCol);
      this.updateHBMDataProcessors(nameCol, achievedCol, coreCol);
      this.updateView();
    }
  }

  override updateView() {
    const filters = this.getFilters();
    for (const processor of Object.values(this.dataProcessorTensorNodesUnit)) {
      if (processor) {
        processor.setFilters(filters);
      }
    }
    for (const processor of Object.values(
      this.dataProcessorTensorNodesBandwidth,
    )) {
      if (processor) {
        processor.setFilters(filters);
      }
    }
    this.hbmCoreIndexes.forEach((index: number) => {
      const processor = this.dataProcessorHBMRatio[index];
      if (processor) {
        processor.setFilters(filters);
      }
    });
    this.dataProvider.notifyCharts();
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
