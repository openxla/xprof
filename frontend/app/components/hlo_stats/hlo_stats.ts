import {Component, OnDestroy} from '@angular/core';
import {FormControl} from '@angular/forms';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {OpType} from 'org_xprof/frontend/app/common/constants/enums';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {CategoryTableDataProcessor} from 'org_xprof/frontend/app/components/chart/category_table_data_processor';
import {PIE_CHART_OPTIONS, TABLE_OPTIONS,} from 'org_xprof/frontend/app/components/chart/chart_options';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {DefaultDataProvider, ReplicaGroupDataProvider,} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {DataService} from 'org_xprof/frontend/app/services/data_service/data_service';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

const OP_CATEGORY_ID = 'category';
const OP_NAME_ID = 'hlo_op_name';
const PROGRAM_ID = 'program_id';
const OP_EXPRESSION_ID = 'hlo_op_expression';
const SELF_TIME_ID = 'total_self_time';
const HLO_REMAT_ID = 'hlo_rematerialization';
const OUTSIDE_COMPILATION_ID = 'outside_compilation';
const MEASURED_FLOP_RATE_ID = 'model_flop_rate';

/** A Hlo Stats component. */
@Component({
  standalone: false,
  selector: 'hlo-stats',
  templateUrl: './hlo_stats.ng.html',
  styleUrls: ['./hlo_stats.css'],
})
export class HloStats extends Dashboard implements OnDestroy {
  readonly tool = 'hlo_stats';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  sessionId = '';
  data: SimpleDataTable|null = null;
  hloOpNameSelected = '';
  programIdSelected = '';
  // Flop rate chart properties.
  readonly opType = OpType.XLA_HLO;
  flopRateChartXColumn = -1;
  flopRateChartYColumn = -1;
  // Pie charts properties.
  pieChartDataProvider = new DefaultDataProvider();
  replicaGroupDataProvider = new ReplicaGroupDataProvider();
  dataInfoCategoryChart: ChartDataInfo = {
    data: null,
    dataProvider: this.pieChartDataProvider,
    options: PIE_CHART_OPTIONS,
  };
  dataInfoOpChart: ChartDataInfo = {
    data: null,
    dataProvider: this.pieChartDataProvider,
    options: PIE_CHART_OPTIONS,
  };
  communicationOps = new Set();
  selectedCommOp = '';
  dataInfoOpReplicaGroupChart: ChartDataInfo = {
    data: null,
    dataProvider: this.replicaGroupDataProvider,
    options: PIE_CHART_OPTIONS,
  };
  dataInfoRematerializationChart: ChartDataInfo = {
    data: null,
    dataProvider: this.pieChartDataProvider,
    options: PIE_CHART_OPTIONS,
  };
  dataInfoRematerializationCategoryChart: ChartDataInfo = {
    data: null,
    dataProvider: this.pieChartDataProvider,
    options: PIE_CHART_OPTIONS,
  };
  dataInfoOutsideCompilationChart: ChartDataInfo = {
    data: null,
    dataProvider: this.pieChartDataProvider,
    options: PIE_CHART_OPTIONS,
  };
  // Table properties.
  dataInfoForTable: ChartDataInfo = {
    data: null,
    dataProvider: new DefaultDataProvider(),
    filters: [],
    options: {
      ...TABLE_OPTIONS,
      showRowNumber: false,
      page: 'enable',
      pageSize: 100,
      sortAscending: true,
      sortColumn: 0,
    },
  };
  showChartSection = true;
  tableColumnsControl = new FormControl([0, 1, 2, 3, 4, 5, 6, 7, 8]);
  tableColumns: Array<{index: number; label: string}> = [];

  constructor(
      route: ActivatedRoute,
      private readonly dataService: DataService,
      private readonly store: Store<{}>,
  ) {
    super();
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.update(params as NavigationEvent);
    });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
    this.tableColumnsControl.valueChanges.subscribe((newValue) => {
      this.updateTableColumns(newValue || []);
    });
  }

  update(event: NavigationEvent) {
    const run = event.run || '';
    const tag = event.tag || 'hlo_stats';
    const host = event.host || '';

    setLoadingState(true, this.store, 'Loading hlo data');

    this.dataService.getData(run, tag, host)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          setLoadingState(false, this.store);
          this.data = data as SimpleDataTable | null;
          this.process(this.data);
          this.onCheckInputParams();
        });
  }

  onCheckInputParams() {
    this.hloOpNameSelected =
        this.dataService.searchParams?.get('hlo_op_name') || '';
    // Assumption: the program_id is in format like 'main(<program_id>)'
    // parsing with a regex to match content in the bracket
    const programIdParsed =
        this.dataService.searchParams?.get('program_id')?.match(/\((.*)\)/);
    this.programIdSelected =
        programIdParsed?.length === 2 ? programIdParsed[1] : '';
  }

  // Iterate through the table data
  // and inject graph link to the hlo op text cell
  addGraphViewerLinkInTableData(data: SimpleDataTable) {
    const programIdColumnIdx =
        data.cols?.findIndex((col) => col.id === PROGRAM_ID) ?? -1;
    const hloOpExpressionColumnIdx =
        data.cols?.findIndex((col) => col.id === OP_EXPRESSION_ID) ?? -1;
    const hloOpNameColumnIdx =
        data.cols?.findIndex((col) => col.id === OP_NAME_ID) ?? -1;
    if (programIdColumnIdx === -1 || hloOpExpressionColumnIdx === -1 ||
        hloOpNameColumnIdx === -1) {
      return data;
    }

    const updatedData = {
      ...data,
      rows: data?.rows!.map((row, index) => {
        const programId = (row.c![programIdColumnIdx].v as string).trim() || '';
        const hloOpName = (row.c![hloOpNameColumnIdx].v as string).trim() || '';
        const hloOpExpression =
            (row.c![hloOpExpressionColumnIdx].v as string) || '';
        const graphViewerLink = `/graph_viewer/${this.sessionId}?program_id=${
            programId}&node_name=${hloOpName}`;
        return {
          ...row,
          c: [
            ...row.c!.slice(0, hloOpExpressionColumnIdx),
            {
              ...row.c![hloOpExpressionColumnIdx],
              v: `<a href="${graphViewerLink}" target="_blank">${
                  hloOpExpression}</a>`,
            },
            ...row.c!.slice(hloOpExpressionColumnIdx + 1),
          ],
        };
      }),
    };
    return updatedData;
  }

  private process(data: SimpleDataTable|null) {
    if (!data) return;

    this.parseData(data);
    this.drawFlopRateChart();
    this.updateOpReplicaGroupChart();

    const updatedData = this.addGraphViewerLinkInTableData(data);
    this.dataInfoForTable = {
      ...this.dataInfoForTable,
      data: updatedData,
    };
  }

  override updateView() {
    this.dataInfoForTable = {
      ...this.dataInfoForTable,
      filters: this.getFilters(),
    };
  }

  updateOpReplicaGroupChart() {
    if (!this.replicaGroupDataProvider.opCategoryIndex ||
        !this.replicaGroupDataProvider.hloOpNameIndex ||
        !this.replicaGroupDataProvider.selfTimeIndex) {
      return;
    }

    const filtersForReplicaGroup = [
      {
        column: this.replicaGroupDataProvider.opCategoryIndex,
        value: this.selectedCommOp,
      },
    ];

    this.dataInfoOpReplicaGroupChart.customChartDataProcessor =
        new CategoryTableDataProcessor(
            filtersForReplicaGroup,
            this.replicaGroupDataProvider.hloOpNameIndex,
            this.replicaGroupDataProvider.selfTimeIndex,
        );

    // Since the DataInfo has not been updated, the notifyCharts function is
    // called to redraw the graph.
    this.replicaGroupDataProvider.notifyCharts();
  }

  processTableColumns(dataTable: google.visualization.DataTable) {
    const numColumns = dataTable.getNumberOfColumns();
    this.tableColumns = [];
    for (let i = 0; i < numColumns; i++) {
      this.tableColumns.push({
        index: i,
        label: dataTable.getColumnLabel(i),
      });
    }
    this.updateTableColumns(
        this.tableColumnsControl.value || [0, 1, 2, 3, 4, 5, 6, 7, 8],
    );
  }

  updateTableColumns(newValue: number[]) {
    if (newValue.length === 0) return;
    this.dataInfoForTable.dataProvider.setVisibleColumns(newValue);
    this.dataInfoForTable.dataProvider.notifyCharts();
  }

  override parseData(data: SimpleDataTable|null) {
    if (!data) return;
    // Five charts share one DataProvider. In order to prevent DataTable from
    // being created multiple times, it calls DataProvider function directly.
    this.pieChartDataProvider.parseData(data);
    const dataTable = this.pieChartDataProvider.getDataTable();
    if (!dataTable) return;

    this.dataTable = dataTable;
    this.processTableColumns(dataTable);
    this.updateView();

    const hloOpNameIndex = dataTable.getColumnIndex(OP_EXPRESSION_ID);
    const opCategoryIndex = dataTable.getColumnIndex(OP_CATEGORY_ID);
    const selfTimeIndex = dataTable.getColumnIndex(SELF_TIME_ID);
    const hloRematIndex = dataTable.getColumnIndex(HLO_REMAT_ID);
    const outsideCompilationIndex = dataTable.getColumnIndex(
        OUTSIDE_COMPILATION_ID,
    );

    const filtersForRemat = [{column: hloRematIndex, value: 'Yes'}];

    this.dataInfoCategoryChart.customChartDataProcessor =
        new CategoryTableDataProcessor([], opCategoryIndex, selfTimeIndex);
    this.dataInfoOpChart.customChartDataProcessor =
        new CategoryTableDataProcessor([], hloOpNameIndex, selfTimeIndex);
    this.dataInfoRematerializationChart.customChartDataProcessor =
        new CategoryTableDataProcessor([], hloRematIndex, selfTimeIndex, false);
    this.dataInfoRematerializationCategoryChart.customChartDataProcessor =
        new CategoryTableDataProcessor(
            filtersForRemat,
            opCategoryIndex,
            selfTimeIndex,
        );
    this.dataInfoOutsideCompilationChart.customChartDataProcessor =
        new CategoryTableDataProcessor(
            [],
            outsideCompilationIndex,
            selfTimeIndex,
            false,
        );

    // Since the DataInfo has not been updated, the notifyCharts function is
    // called to redraw the graph.
    this.pieChartDataProvider.notifyCharts();

    // Create a DataProvider in which the row string value for hloOpName column
    // is truncated to only be the 'replica_groups={{...}}' string.
    this.replicaGroupDataProvider.parseData(data);
    this.communicationOps = this.replicaGroupDataProvider.communicationOps;

    if (this.communicationOps.size) {
      // Set value to the first communication Op in the set.
      this.selectedCommOp = this.communicationOps.values().next().value;
    }
  }

  private drawFlopRateChart() {
    if (!this.dataTable || !this.dataTable.getColumnIndex) return;
    this.flopRateChartXColumn = this.dataTable.getColumnIndex(OP_EXPRESSION_ID);
    this.flopRateChartYColumn = this.dataTable.getColumnIndex(
        MEASURED_FLOP_RATE_ID,
    );
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
