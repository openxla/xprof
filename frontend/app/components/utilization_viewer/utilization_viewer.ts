import {Component, inject, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {ChartDataInfo, ChartType} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {BAR_CHART_OPTIONS, PIE_CHART_OPTIONS, } from 'org_xprof/frontend/app/components/chart/chart_options';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {FilterDataProcessor} from 'org_xprof/frontend/app/components/chart/filter_data_processor';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
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

const BANDWIDTH_CHART_OPTIONS: google.visualization.BarChartOptions = {
  ...BAR_CHART_OPTIONS,
  width: 800,
  height: 400,
  chartArea: {left: '25%', width: '65%', height: 300},
  hAxis: {format: 'percent', minValue: 0.0, maxValue: 1.0},
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
  [index: number]: FilterDataProcessor|null;
}

/**
 * Utilization viewer component.
 * The utilization viewer displays unit and bandwidth utiization for each tensor
 * node in a TPU chip.
 */
@Component({
  standalone: false,
  selector: 'utilization-viewer',
  templateUrl: './utilization_viewer.ng.html',
  styleUrls: ['./utilization_viewer.scss'],
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
      this.update();
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

    this.dataService.getData(this.sessionId, this.tool)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          this.throbber.stop();
          setLoadingState(false, this.store);
          this.parseData(data as SimpleDataTable | null);
        });
  }

  initDataProcessors() {
    this.coreIndexes.forEach((index: number) => {
      this.dataInfoTensorNodesUnit[index] = {
        data: null,
        dataProvider: this.dataProvider,
        options: UNIT_CHART_OPTIONS,
      };
      this.dataProcessorTensorNodesUnit[index] = null;
      this.dataInfoTensorNodesBandwidth[index] = {
        data: null,
        dataProvider: this.dataProvider,
        options: BANDWIDTH_CHART_OPTIONS,
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
      visibleColumns: Array<number|google.visualization.ColumnSpec>,
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

  override parseData(data: SimpleDataTable|null) {
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

      const visibleColumns: Array<number|google.visualization.ColumnSpec> = [
        nameCol,
        {
          calc: (data: google.visualization.DataTable, row: number) => {
            const achieved = data.getValue(row, achievedCol);
            const peak = data.getValue(row, peakCol);
            return peak !== 0 ? achieved / peak : 0;
          },
          type: 'number',
          label: '% Active',
          id: 'active',
        },
        {
          calc: (data: google.visualization.DataTable, row: number) => {
            const achieved = data.getValue(row, achievedCol);
            const peak = data.getValue(row, peakCol);
            const unit = data.getValue(row, 7);
            return `Achieved: ${achieved.toLocaleString()} ${unit}\n(Peak: ${
                peak.toLocaleString()} ${unit})`;
          },
          type: 'string',
          role: 'tooltip',
        },
        {
          calc: (data: google.visualization.DataTable, row: number) => {
            const achieved = data.getValue(row, achievedCol);
            if (achieved === 0) return undefined;
            const peak = data.getValue(row, 6);
            return ((100 * achieved) / peak).toFixed(2) + '%';
          },
          type: 'string',
          role: 'annotation',
        },
      ];

      this.initDataProcessors();
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
             this.dataProcessorTensorNodesBandwidth)) {
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
