import {Component, inject, OnDestroy} from '@angular/core';
import {FormControl} from '@angular/forms';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {OpType} from 'org_xprof/frontend/app/common/constants/enums';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {CategoryTableDataProcessor} from 'org_xprof/frontend/app/components/chart/category_table_data_processor';
import {PIE_CHART_OPTIONS, TABLE_OPTIONS,} from 'org_xprof/frontend/app/components/chart/chart_options';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {DefaultDataProvider, ReplicaGroupDataProvider,} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

const AVG_TIME_ID = 'avg_time';
const HLO_REMAT_ID = 'hlo_rematerialization';
const MEASURED_FLOP_RATE_ID = 'model_flop_rate';
const OCCURRENCES_ID = 'occurrences';
const OP_CATEGORY_ID = 'category';
const OP_EXPRESSION_ID = 'hlo_op_expression';
const OP_NAME_ID = 'hlo_op_name';
const OUTSIDE_COMPILATION_ID = 'outside_compilation';
const PROGRAM_ID = 'program_id';
const RANK_ID = 'rank';
const SELF_TIME_ID = 'total_self_time';
const SOURCE_INFO_ID = 'source_info';
const SOURCE_STACK_ID = 'source_stack';
const TF_OP_NAME_ID = 'tf_op_name';
const TOTAL_TIME_ID = 'total_time';

/** A Hlo Stats component. */
@Component({
  standalone: false,
  selector: 'hlo-stats',
  templateUrl: './hlo_stats.ng.html',
  styleUrls: ['./hlo_stats.css'],
})
export class HloStats extends Dashboard implements OnDestroy {
  tool = 'hlo_op_stats';
  sessionId = '';
  host = '';
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
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
  tableColumnsControl = new FormControl<number[]>([]);
  tableColumns: Array<{index: number; label: string}> = [];

  constructor(
      route: ActivatedRoute,
      private readonly store: Store<{}>,
  ) {
    super();
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.processQuery(params);
      this.update();
    });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
    this.tableColumnsControl.valueChanges.subscribe((newValue) => {
      this.updateTableColumns(newValue || []);
    });
  }

  processQuery(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || this.tool;
    this.host = params['host'] || this.host;
  }

  update() {
    setLoadingState(true, this.store, 'Loading hlo data');

    this.dataService.getData(this.sessionId, this.tool, this.host)
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
        const graphViewerLink = this.dataService.getGraphViewerLink(
            this.sessionId,
            '',
            hloOpName,
            programId,
        );
        const hyperlinkValue = graphViewerLink ?
            `<a href="${graphViewerLink}" target="_blank">${
                hloOpExpression}</a>` :
            hloOpExpression;
        return {
          ...row,
          c: [
            ...row.c!.slice(0, hloOpExpressionColumnIdx),
            {
              ...row.c![hloOpExpressionColumnIdx],
              v: hyperlinkValue,
            },
            ...row.c!.slice(hloOpExpressionColumnIdx + 1),
          ],
        };
      }),
    };
    return updatedData;
  }

  /**
   * Merges the source info and stack columns into a single column.
   *
   * If any of the following conditions are met, the original data is returned:
   * - `data.cols` is `null`.
   * - `data.cols` does not contain a column with ID `SOURCE_INFO_ID`.
   * - `data.cols` does not contain a column with ID `SOURCE_STACK_ID`.
   *
   * If the original data is not returned, then a shallow copy is returned where
   * the column with ID `SOURCE_STACK_ID` is removed and its content is merged
   * into the column with ID `SOURCE_INFO_ID`.
   *
   * TODO(b/411696636): Simplify the logic after new changes are applied.
   */
  private mergeSourceInfoColumns(data: SimpleDataTable): SimpleDataTable {
    const infoColIdx =
        data.cols?.findIndex((col) => col.id === SOURCE_INFO_ID) ?? -1;
    const stackColIdx =
        data.cols?.findIndex((col) => col.id === SOURCE_STACK_ID) ?? -1;

    if (infoColIdx === -1 || stackColIdx === -1) {
      return data;
    }

    function removeSourceStackColumn<T>(array: T[]): T[] {
      return array.filter((_, idx) => idx !== stackColIdx);
    }

    function sourceInfoCell(
        sourceInfo: string,
        sourceStack: string,
        ): google.visualization.DataObjectCell {
      return {
        v: sourceInfo,
        // We show the source stack in a tooltip. Also, we assume that neither
        // `sourceStack` nor `sourceInfo` contains HTML tags. In other words,
        // we don't need to escape them.
        f: `<div title="${sourceStack}">${sourceInfo}</div>`,
      };
    }

    const updatedData = {
      ...data,
      cols: removeSourceStackColumn(data?.cols || []),
      rows: (data?.rows || []).map((row, index) => {
        const cells = row.c || [];
        const sourceInfo = (cells[infoColIdx]?.v as string) || '';
        const sourceStack = (cells[stackColIdx]?.v as string) || '';
        return {
          ...row,
          c: removeSourceStackColumn([
            ...cells.slice(0, infoColIdx),
            sourceInfoCell(sourceInfo, sourceStack),
            ...cells.slice(infoColIdx + 1),
          ]),
        };
      }),
    };
    return updatedData;
  }

  private process(data: SimpleDataTable|null) {
    if (!data) return;

    // `mergeSourceInfoColumns` needs to be called before `parseData`, because
    // it removes a column.
    const dataWithSourceInfo = this.mergeSourceInfoColumns(data);

    this.parseData(dataWithSourceInfo);
    this.drawFlopRateChart();
    this.updateOpReplicaGroupChart();

    const updatedData = this.addGraphViewerLinkInTableData(dataWithSourceInfo);
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
    this.tableColumns = [];
    const numColumns = dataTable.getNumberOfColumns();
    const defaultVisibleColumns = [];
    const defaultVisibleColumnIds = new Set([
      AVG_TIME_ID,
      OCCURRENCES_ID,
      OP_CATEGORY_ID,
      OP_EXPRESSION_ID,
      OP_NAME_ID,
      PROGRAM_ID,
      RANK_ID,
      SOURCE_INFO_ID,
      TF_OP_NAME_ID,
      TOTAL_TIME_ID,
    ]);
    for (let i = 0; i < numColumns; i++) {
      this.tableColumns.push({
        index: i,
        label: dataTable.getColumnLabel(i),
      });
      if (defaultVisibleColumnIds.has(dataTable.getColumnId(i))) {
        defaultVisibleColumns.push(i);
      }
    }
    if (this.tableColumnsControl?.value?.length === 0) {
      this.tableColumnsControl.setValue(defaultVisibleColumns);
    }
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
