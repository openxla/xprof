import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  inject,
  Injector,
  NgZone,
  OnDestroy,
  Renderer2,
  ViewChild,
} from '@angular/core';
import {FormControl} from '@angular/forms';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {OpType} from 'org_xprof/frontend/app/common/constants/enums';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {alignTables} from 'org_xprof/frontend/app/common/utils/diff_utils';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {CategoryTableDataProcessor} from 'org_xprof/frontend/app/components/chart/category_table_data_processor';
import {Chart} from 'org_xprof/frontend/app/components/chart/chart';
import {
  PIE_CHART_OPTIONS,
  TABLE_OPTIONS,
} from 'org_xprof/frontend/app/components/chart/chart_options';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {
  DefaultDataProvider,
  ReplicaGroupDataProvider,
} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {BaseDiffService} from 'org_xprof/frontend/app/services/data_service_v2/diff_service';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, ReplaySubject} from 'rxjs';
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
const TF_OP_NAME_ID = 'tf_op_name';
const TOTAL_TIME_ID = 'total_time';
const CORE_TYPE_ID = 'core_type';
const SPARSE_CORE_VALUE = 'SparseCore';
const VDD_ENERGY_ID = 'vdd_energy';

/** A Hlo Stats component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'hlo-stats',
  templateUrl: './hlo_stats.ng.html',
  styleUrls: ['./hlo_stats.css'],
})
export class HloStats extends Dashboard implements OnDestroy {
  tool = 'hlo_op_stats';
  sessionId = '';
  host = '';
  private lastBaseSessionId = '';
  private readonly injector = inject(Injector);
  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );
  private readonly diffService = inject(BaseDiffService);
  private readonly zone = inject(NgZone);
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly throbber = new Throbber(this.tool);
  data: SimpleDataTable | null = null;
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
      pageSize: 10,
      sortAscending: true,
      sortColumn: 0,
    },
  };
  dataInfoForSparseCoreTable: ChartDataInfo = {
    data: null,
    dataProvider: new DefaultDataProvider(),
    filters: [],
    options: {
      ...TABLE_OPTIONS,
      showRowNumber: false,
      page: 'enable',
      pageSize: 10,
      sortAscending: true,
      sortColumn: 0,
    },
  };
  showChartSection = true;
  tableColumnsControl = new FormControl<number[]>([]);
  tableColumns: Array<{index: number; label: string}> = [];
  hasSparseCoreData = false;

  // We add a listener to `chart` and manipulate multiple elements of
  // `chartElement`. Knowing that `Chart.elementRef` is private, we use
  // `ViewChild` twice to access both. See `addSourceInfoClickListener` for
  // more details.
  @ViewChild('table', {read: Chart, static: false})
  chartRef: Chart | undefined = undefined;
  @ViewChild('table', {read: ElementRef, static: false})
  chartElementRef: ElementRef | undefined = undefined;
  @ViewChild('sparseCoreTable', {read: Chart, static: false})
  sparseCoreChartRef: Chart | undefined = undefined;
  @ViewChild('sparseCoreTable', {read: ElementRef, static: false})
  sparseCoreChartElementRef: ElementRef | undefined = undefined;
  private readonly renderer: Renderer2 = inject(Renderer2);
  sourceFileAndLineNumber = '';
  stackTrace = '';
  showStackTrace = false;
  sourceCodeServiceIsAvailable = false;

  constructor(
    route: ActivatedRoute,
    private readonly store: Store<{}>,
  ) {
    super();
    combineLatest([route.params, route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        const oldSessionId = this.sessionId;
        const oldTool = this.tool;
        const oldHost = this.host;
        const oldHloOpName = this.hloOpNameSelected;
        const oldProgramId = this.programIdSelected;

        this.sessionId = params['sessionId'] || this.sessionId;
        this.processQueryParams(queryParams);
        const currentBaseSessionId = this.diffService.getBaseSessionId() || '';
        // Trigger update only if the parameters actually changed.
        const hasChanged =
          this.sessionId !== oldSessionId ||
          this.tool !== oldTool ||
          this.host !== oldHost ||
          this.hloOpNameSelected !== oldHloOpName ||
          this.programIdSelected !== oldProgramId ||
          currentBaseSessionId !== this.lastBaseSessionId;
        this.lastBaseSessionId = currentBaseSessionId;
        if (hasChanged) {
          this.update();
        }
      });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
    this.tableColumnsControl.valueChanges.subscribe((newValue) => {
      this.updateTableColumns(newValue || []);
    });

    // We don't need the source code service to be persistently available.
    // We temporarily use the service to check if it is available and show
    // UI accordingly.
    const sourceCodeService = this.injector.get(
      SOURCE_CODE_SERVICE_INTERFACE_TOKEN,
      null,
    );
    sourceCodeService
      ?.isAvailable()
      .pipe(takeUntil(this.destroyed))
      .subscribe((isAvailable) => {
        this.sourceCodeServiceIsAvailable = isAvailable;
        if (this.sourceCodeServiceIsAvailable) {
          this.addSourceInfoClickListener(this.chartRef, this.chartElementRef);
          this.addSourceInfoClickListener(
            this.sparseCoreChartRef,
            this.sparseCoreChartElementRef,
          );
        }
      });
  }

  processQueryParams(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || this.tool;
    this.host = params['host'] || this.host;
  }

  update() {
    setLoadingState(true, this.store, 'Loading hlo data');
    this.throbber.start();

    this.diffService
      .getDiffData<SimpleDataTable>(this.sessionId, this.tool, {
        host: this.host,
      })
      .pipe(takeUntil(this.destroyed))
      .subscribe(
        ({
          active,
          baseline,
        }: {
          active: SimpleDataTable | null;
          baseline: SimpleDataTable | null;
        }) => {
          this.throbber.stop();
          setLoadingState(false, this.store);
          this.data = this.mergeTables(active, baseline);
          this.process(this.data);
          this.onCheckInputParams();
        },
      );
  }

  private mergeTables(
    active: SimpleDataTable | null,
    baseline: SimpleDataTable | null,
  ): SimpleDataTable | null {
    if (
      !active ||
      !baseline ||
      !active.cols ||
      !active.rows ||
      !baseline.cols ||
      !baseline.rows
    ) {
      return active;
    }

    const activeCols = active.cols;
    const baselineCols = baseline.cols;

    const programIdCol = activeCols.findIndex((col) => col.id === PROGRAM_ID);
    const categoryCol = activeCols.findIndex(
      (col) => col.id === OP_CATEGORY_ID,
    );
    const nameCol = activeCols.findIndex((col) => col.id === OP_NAME_ID);
    const exprCol = activeCols.findIndex((col) => col.id === OP_EXPRESSION_ID);
    const coreTypeCol = activeCols.findIndex((col) => col.id === CORE_TYPE_ID);

    const totalTimeCol = activeCols.findIndex(
      (col) => col.id === TOTAL_TIME_ID,
    );
    const selfTimeCol = activeCols.findIndex((col) => col.id === SELF_TIME_ID);
    const flopRateCol = activeCols.findIndex(
      (col) => col.id === MEASURED_FLOP_RATE_ID,
    );
    const occurrencesCol = activeCols.findIndex(
      (col) => col.id === OCCURRENCES_ID,
    );
    const avgTimeCol = activeCols.findIndex((col) => col.id === AVG_TIME_ID);

    const baselineProgramIdCol = baselineCols.findIndex(
      (col) => col.id === PROGRAM_ID,
    );
    const baselineCategoryCol = baselineCols.findIndex(
      (col) => col.id === OP_CATEGORY_ID,
    );
    const baselineNameCol = baselineCols.findIndex(
      (col) => col.id === OP_NAME_ID,
    );
    const baselineExprCol = baselineCols.findIndex(
      (col) => col.id === OP_EXPRESSION_ID,
    );
    const baselineCoreTypeCol = baselineCols.findIndex(
      (col) => col.id === CORE_TYPE_ID,
    );

    const baselineTotalTimeCol = baselineCols.findIndex(
      (col) => col.id === TOTAL_TIME_ID,
    );
    const baselineSelfTimeCol = baselineCols.findIndex(
      (col) => col.id === SELF_TIME_ID,
    );
    const baselineFlopRateCol = baselineCols.findIndex(
      (col) => col.id === MEASURED_FLOP_RATE_ID,
    );
    const baselineOccurrencesCol = baselineCols.findIndex(
      (col) => col.id === OCCURRENCES_ID,
    );
    const baselineAvgTimeCol = baselineCols.findIndex(
      (col) => col.id === AVG_TIME_ID,
    );

    if (nameCol === -1 || baselineNameCol === -1) {
      return active;
    }

    const getActiveRowKey = (row: google.visualization.DataObjectRow) => {
      const programId =
        programIdCol !== -1 ? (row.c?.[programIdCol]?.v ?? '') : '';
      const category =
        categoryCol !== -1 ? (row.c?.[categoryCol]?.v ?? '') : '';
      const name = row.c?.[nameCol]?.v ?? '';
      const expr = exprCol !== -1 ? (row.c?.[exprCol]?.v ?? '') : '';
      const coreType =
        coreTypeCol !== -1 ? (row.c?.[coreTypeCol]?.v ?? '') : '';
      return `${programId}_${category}_${name}_${expr}_${coreType}`;
    };

    const getBaselineRowKey = (row: google.visualization.DataObjectRow) => {
      const programId =
        baselineProgramIdCol !== -1
          ? (row.c?.[baselineProgramIdCol]?.v ?? '')
          : '';
      const category =
        baselineCategoryCol !== -1
          ? (row.c?.[baselineCategoryCol]?.v ?? '')
          : '';
      const name = row.c?.[baselineNameCol]?.v ?? '';
      const expr =
        baselineExprCol !== -1 ? (row.c?.[baselineExprCol]?.v ?? '') : '';
      const coreType =
        baselineCoreTypeCol !== -1
          ? (row.c?.[baselineCoreTypeCol]?.v ?? '')
          : '';
      return `${programId}_${category}_${name}_${expr}_${coreType}`;
    };

    const aligned = alignTables(
      active.rows,
      baseline.rows,
      getActiveRowKey,
      undefined,
      getBaselineRowKey,
    );

    const activeToBaselineColMap = activeCols.map((col) =>
      baselineCols.findIndex((bCol) => bCol.id === col.id),
    );

    const newCols = [...activeCols];
    const baselineTotalTimeColIndex = totalTimeCol !== -1 ? newCols.length : -1;
    if (totalTimeCol !== -1) {
      newCols.push({
        id: `${TOTAL_TIME_ID}_baseline`,
        label: 'Baseline Total Time (ms)',
        type: 'number',
      });
    }
    const baselineSelfTimeColIndex = selfTimeCol !== -1 ? newCols.length : -1;
    if (selfTimeCol !== -1) {
      newCols.push({
        id: `${SELF_TIME_ID}_baseline`,
        label: 'Baseline Duration (ms)',
        type: 'number',
      });
    }
    const baselineFlopRateColIndex = flopRateCol !== -1 ? newCols.length : -1;
    if (flopRateCol !== -1) {
      newCols.push({
        id: `${MEASURED_FLOP_RATE_ID}_baseline`,
        label: 'Baseline Measured GFLOP/s',
        type: 'number',
      });
    }
    const baselineOccurrencesColIndex =
      occurrencesCol !== -1 ? newCols.length : -1;
    if (occurrencesCol !== -1) {
      newCols.push({
        id: `${OCCURRENCES_ID}_baseline`,
        label: 'Baseline Occurrences',
        type: 'number',
      });
    }
    const baselineAvgTimeColIndex = avgTimeCol !== -1 ? newCols.length : -1;
    if (avgTimeCol !== -1) {
      newCols.push({
        id: `${AVG_TIME_ID}_baseline`,
        label: 'Baseline Average Time (ms)',
        type: 'number',
      });
    }

    const getBaselineCellValue = (
      row: google.visualization.DataObjectRow | null | undefined,
      colIdx: number,
    ) => {
      if (!row || colIdx === -1) return 0;
      return row.c?.[colIdx]?.v ?? 0;
    };

    const getBaselineCellFormat = (
      row: google.visualization.DataObjectRow | null | undefined,
      colIdx: number,
    ) => {
      if (!row || colIdx === -1) return undefined;
      return row.c?.[colIdx]?.f;
    };

    const newRows: google.visualization.DataObjectRow[] = [];

    for (const compRow of aligned.values()) {
      const activeRow = compRow.active;
      const baselineRow = compRow.baseline;

      if (activeRow) {
        const newCells = activeRow.c ? [...activeRow.c] : [];
        if (baselineTotalTimeColIndex !== -1) {
          newCells[baselineTotalTimeColIndex] = {
            v: getBaselineCellValue(baselineRow, baselineTotalTimeCol),
            f: getBaselineCellFormat(baselineRow, baselineTotalTimeCol),
          };
        }
        if (baselineSelfTimeColIndex !== -1) {
          newCells[baselineSelfTimeColIndex] = {
            v: getBaselineCellValue(baselineRow, baselineSelfTimeCol),
            f: getBaselineCellFormat(baselineRow, baselineSelfTimeCol),
          };
        }
        if (baselineFlopRateColIndex !== -1) {
          newCells[baselineFlopRateColIndex] = {
            v: getBaselineCellValue(baselineRow, baselineFlopRateCol),
            f: getBaselineCellFormat(baselineRow, baselineFlopRateCol),
          };
        }
        if (baselineOccurrencesColIndex !== -1) {
          newCells[baselineOccurrencesColIndex] = {
            v: getBaselineCellValue(baselineRow, baselineOccurrencesCol),
            f: getBaselineCellFormat(baselineRow, baselineOccurrencesCol),
          };
        }
        if (baselineAvgTimeColIndex !== -1) {
          newCells[baselineAvgTimeColIndex] = {
            v: getBaselineCellValue(baselineRow, baselineAvgTimeCol),
            f: getBaselineCellFormat(baselineRow, baselineAvgTimeCol),
          };
        }
        newRows.push({c: newCells});
      } else if (baselineRow) {
        const newCells: google.visualization.DataObjectCell[] = [];
        for (let i = 0; i < activeCols.length; i++) {
          const bColIdx = activeToBaselineColMap[i];
          newCells[i] =
            bColIdx !== -1 && baselineRow.c?.[bColIdx]
              ? {...baselineRow.c[bColIdx]}
              : {v: ''};
        }

        if (totalTimeCol !== -1) newCells[totalTimeCol] = {v: 0};
        if (selfTimeCol !== -1) newCells[selfTimeCol] = {v: 0};
        if (flopRateCol !== -1) newCells[flopRateCol] = {v: 0};
        if (occurrencesCol !== -1) newCells[occurrencesCol] = {v: 0};
        if (avgTimeCol !== -1) newCells[avgTimeCol] = {v: 0};

        if (baselineTotalTimeColIndex !== -1) {
          newCells[baselineTotalTimeColIndex] = {
            v: getBaselineCellValue(baselineRow, baselineTotalTimeCol),
            f: getBaselineCellFormat(baselineRow, baselineTotalTimeCol),
          };
        }
        if (baselineSelfTimeColIndex !== -1) {
          newCells[baselineSelfTimeColIndex] = {
            v: getBaselineCellValue(baselineRow, baselineSelfTimeCol),
            f: getBaselineCellFormat(baselineRow, baselineSelfTimeCol),
          };
        }
        if (baselineFlopRateColIndex !== -1) {
          newCells[baselineFlopRateColIndex] = {
            v: getBaselineCellValue(baselineRow, baselineFlopRateCol),
            f: getBaselineCellFormat(baselineRow, baselineFlopRateCol),
          };
        }
        if (baselineOccurrencesColIndex !== -1) {
          newCells[baselineOccurrencesColIndex] = {
            v: getBaselineCellValue(baselineRow, baselineOccurrencesCol),
            f: getBaselineCellFormat(baselineRow, baselineOccurrencesCol),
          };
        }
        if (baselineAvgTimeColIndex !== -1) {
          newCells[baselineAvgTimeColIndex] = {
            v: getBaselineCellValue(baselineRow, baselineAvgTimeCol),
            f: getBaselineCellFormat(baselineRow, baselineAvgTimeCol),
          };
        }
        newRows.push({c: newCells});
      }
    }

    return {
      cols: newCols,
      rows: newRows,
      p: active.p,
    };
  }

  onCheckInputParams() {
    this.hloOpNameSelected =
      this.dataService.getSearchParams().get('hlo_op_name') || '';
    // Assumption: the program_id is in format like 'main(<program_id>)'
    // parsing with a regex to match content in the bracket
    const programIdParsed = this.dataService
      .getSearchParams()
      .get('program_id')
      ?.match(/\((.*)\)/);
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
    if (
      programIdColumnIdx === -1 ||
      hloOpExpressionColumnIdx === -1 ||
      hloOpNameColumnIdx === -1
    ) {
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
        const hyperlinkValue = graphViewerLink
          ? `<a href="${graphViewerLink}" target="_blank">${
              hloOpExpression
            }</a>`
          : hloOpExpression;
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
   * Adds a click listener to the source info cells.
   *
   * If "Show Source Code" is checked, then whenever user clicks on the source
   * info cell, we show snippets of source code around the stack trace.
   *
   * Unfortunately, `google.visualization.Table` does not provide any API to
   * listen to click events on *cells*. So we manually add the click listener
   * to the items in this table (see
   * https://developers.google.com/chart/interactive/docs/gallery/table#events
   * as a reference).
   *
   * Unfortunately, `google.visualization.Table` does not provide enough
   * extension points to add interactive elements to cells. Therefore, we go
   * to the native elements of the table and add the click listener to the
   * cells with class `source-info-cell`.
   */
  private addSourceInfoClickListener(
    chartRef: Chart | undefined,
    chartElementRef: ElementRef | undefined,
  ) {
    const chart = chartRef?.chart;
    const chartElement = chartElementRef?.nativeElement;
    if (!chart || !chartElement) {
      // TODO: b/429036372 - Using setTimeout to detect change is inefficient.
      setTimeout(() => {
        this.addSourceInfoClickListener(chartRef, chartElementRef);
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

  private process(data: SimpleDataTable | null) {
    if (!data) return;

    const coreTypeIdx =
      data.cols?.findIndex((col) => col.id === CORE_TYPE_ID) ?? -1;
    let tensorCoreDataRows = data.rows;
    let sparseCoreRows: google.visualization.DataObjectRow[] = [];

    if (coreTypeIdx !== -1) {
      tensorCoreDataRows =
        data.rows?.filter(
          (row) => row.c![coreTypeIdx]?.v !== SPARSE_CORE_VALUE,
        ) || [];
      sparseCoreRows =
        data.rows?.filter(
          (row) => row.c![coreTypeIdx]?.v === SPARSE_CORE_VALUE,
        ) || [];
    }

    const tensorCoreData = {...data, rows: tensorCoreDataRows};
    const sparseCoreData = {...data, rows: sparseCoreRows};

    this.parseData(tensorCoreData);
    this.drawFlopRateChart();
    this.updateOpReplicaGroupChart();

    const updatedMainData = this.addGraphViewerLinkInTableData(tensorCoreData);
    this.dataInfoForTable = {
      ...this.dataInfoForTable,
      data: updatedMainData,
    };

    const updatedSparseCoreData =
      this.addGraphViewerLinkInTableData(sparseCoreData);
    this.dataInfoForSparseCoreTable = {
      ...this.dataInfoForSparseCoreTable,
      data: updatedSparseCoreData,
    };
    this.hasSparseCoreData = sparseCoreRows.length > 0;

    this.dataInfoForSparseCoreTable.dataProvider.parseData(sparseCoreData);
  }

  override updateView() {
    this.dataInfoForTable = {
      ...this.dataInfoForTable,
      filters: this.getFilters(),
    };
    this.dataInfoForSparseCoreTable = {
      ...this.dataInfoForSparseCoreTable,
      filters: this.getFilters(),
    };
  }

  updateOpReplicaGroupChart() {
    if (
      !this.replicaGroupDataProvider.opCategoryIndex ||
      !this.replicaGroupDataProvider.hloOpNameIndex ||
      !this.replicaGroupDataProvider.selfTimeIndex
    ) {
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
      VDD_ENERGY_ID,
      `${TOTAL_TIME_ID}_baseline`,
      `${SELF_TIME_ID}_baseline`,
      `${MEASURED_FLOP_RATE_ID}_baseline`,
      `${OCCURRENCES_ID}_baseline`,
      `${AVG_TIME_ID}_baseline`,
    ]);
    for (let i = 0; i < numColumns; i++) {
      const colId = dataTable.getColumnId(i);
      if (colId === CORE_TYPE_ID) continue;
      this.tableColumns.push({
        index: i,
        label: dataTable.getColumnLabel(i),
      });
      if (defaultVisibleColumnIds.has(colId)) {
        defaultVisibleColumns.push(i);
      }
    }
    if (this.tableColumnsControl?.value?.length === 0) {
      this.tableColumnsControl.setValue(defaultVisibleColumns);
    }
  }

  updateTableColumns(newValue: number[]) {
    if (newValue.length === 0 || !this.dataTable) return;

    const coreTypeIdx = this.dataTable.getColumnIndex(CORE_TYPE_ID);
    const rankIdx = this.dataTable.getColumnIndex(RANK_ID);

    const visibleColumns = newValue.filter((index) => index !== coreTypeIdx);
    this.dataInfoForTable.dataProvider.setVisibleColumns(visibleColumns);

    const sparseCoreVisibleColumns = visibleColumns.filter(
      (index) => index !== rankIdx,
    );
    this.dataInfoForSparseCoreTable.dataProvider.setVisibleColumns(
      sparseCoreVisibleColumns,
    );

    this.dataInfoForTable.dataProvider.notifyCharts();
    this.dataInfoForSparseCoreTable.dataProvider.notifyCharts();
  }

  override parseData(data: SimpleDataTable | null) {
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
