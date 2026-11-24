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
import {
  createSessionIdResolver,
  escapeHtml,
  getSessionIdFromParams,
  loadGoogleChartPackages,
  resolveHostFromParams,
  setLoadingState,
} from 'org_xprof/frontend/app/common/utils/utils';
import {CategoryDiffTableDataProcessor} from 'org_xprof/frontend/app/components/chart/category_diff_table_data_processor';
import {CategoryTableDataProcessor} from 'org_xprof/frontend/app/components/chart/category_table_data_processor';
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
import {combineLatest, ReplaySubject, Subscription} from 'rxjs';
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

const BASELINE_DEFAULT_VISIBLE_COLUMN_IDS = new Set([
  `${TOTAL_TIME_ID}_baseline`,
  `${SELF_TIME_ID}_baseline`,
  `${MEASURED_FLOP_RATE_ID}_baseline`,
  `${OCCURRENCES_ID}_baseline`,
  `${AVG_TIME_ID}_baseline`,
]);

const DEFAULT_VISIBLE_COLUMN_IDS = new Set([
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
  ...BASELINE_DEFAULT_VISIBLE_COLUMN_IDS,
]);

function getSingleQueryParam(value: unknown): string | undefined {
  if (Array.isArray(value)) {
    const last = value[value.length - 1];
    return typeof last === 'string' ? last : undefined;
  }
  return typeof value === 'string' ? value : undefined;
}

function parseProgramId(rawProgramId: string | null | undefined): string {
  const trimmed = (rawProgramId ?? '').trim();
  if (!trimmed) {
    return '';
  }
  const match = trimmed.match(/\(([^)]*)\)/);
  return match ? match[1].trim() : trimmed;
}

/** A Hlo Stats component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'hlo-stats',
  templateUrl: './hlo_stats.ng.html',
  styleUrls: ['./hlo_stats.scss'],
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
  private dataSubscription?: Subscription;
  private cancelChartLoad?: () => void;
  private latestQueryParams: Params = {};
  private readonly throbber: Throbber;
  data: SimpleDataTable | null = null;
  baselineData: SimpleDataTable | null = null;
  tensorCoreBaselineData: SimpleDataTable | null = null;
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
  communicationOps = new Set<string>();
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
  tableColumnsControl = new FormControl<string[]>([]);
  tableColumns: Array<{id: string; index: number; label: string}> = [];
  private knownColumnIds = new Set<string>();
  hasSparseCoreData = false;

  private chartElementRefInternal?: ElementRef<HTMLElement>;
  private sparseCoreChartElementRefInternal?: ElementRef<HTMLElement>;
  private readonly sourceInfoClickListeners = new Map<
    'table' | 'sparseCoreTable',
    {element: HTMLElement; unlisten: () => void}
  >();

  @ViewChild('table', {read: ElementRef, static: false})
  set chartElementRef(ref: ElementRef<HTMLElement> | undefined) {
    this.chartElementRefInternal = ref;
    this.syncSourceInfoClickListener('table', ref);
  }

  get chartElementRef(): ElementRef<HTMLElement> | undefined {
    return this.chartElementRefInternal;
  }

  @ViewChild('sparseCoreTable', {read: ElementRef, static: false})
  set sparseCoreChartElementRef(ref: ElementRef<HTMLElement> | undefined) {
    this.sparseCoreChartElementRefInternal = ref;
    this.syncSourceInfoClickListener('sparseCoreTable', ref);
  }

  get sparseCoreChartElementRef(): ElementRef<HTMLElement> | undefined {
    return this.sparseCoreChartElementRefInternal;
  }

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
    const routeTool = route.routeConfig?.path?.split('/')[0];
    if (routeTool === 'hlo_stats' || routeTool === 'hlo_op_stats') {
      this.tool = routeTool;
    } else if (window.location.pathname.includes('/data/plugin/profile')) {
      this.tool = 'hlo_stats';
    }
    this.throbber = new Throbber(this.tool);
    const resolveSessionId = createSessionIdResolver();
    combineLatest([route.params, route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        this.latestQueryParams = queryParams;
        const oldSessionId = this.sessionId;
        const oldTool = this.tool;
        const oldHost = this.host;
        const oldHloOpName = this.hloOpNameSelected;
        const oldProgramId = this.programIdSelected;

        this.sessionId = resolveSessionId(params, queryParams, this.sessionId);
        this.processQueryParams(queryParams, oldSessionId);
        this.onCheckInputParams(queryParams);
        if (this.tool !== oldTool) {
          this.store.dispatch(
            setCurrentToolStateAction({currentTool: this.tool}),
          );
        }
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
    this.tableColumnsControl.valueChanges
      .pipe(takeUntil(this.destroyed))
      .subscribe((newValue) => {
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
        this.syncSourceInfoClickListeners();
      });
  }

  processQueryParams(params: Params, prevSessionId = this.sessionId) {
    this.sessionId = getSessionIdFromParams(undefined, params, this.sessionId);
    this.tool = params['tag'] || this.tool;
    this.host = resolveHostFromParams(
      params,
      prevSessionId,
      this.sessionId,
      this.host,
    );
  }

  update() {
    this.dataSubscription?.unsubscribe();
    this.cancelChartLoad?.();
    this.cancelChartLoad = undefined;
    setLoadingState(true, this.store, 'Loading hlo data');
    this.throbber.start();

    this.dataSubscription = this.diffService
      .getDiffData<SimpleDataTable>(this.sessionId, this.tool, {
        host: this.host,
      })
      .pipe(takeUntil(this.destroyed))
      .subscribe({
        next: ({
          active,
          baseline,
        }: {
          active: SimpleDataTable | null;
          baseline: SimpleDataTable | null;
        }) => {
          this.whenChartsLoaded(() => {
            this.throbber.stop();
            setLoadingState(false, this.store);
            this.baselineData = baseline;
            this.data = this.mergeTables(active, baseline);
            this.process(this.data);
            this.onCheckInputParams();
          });
        },
        error: () => {
          this.throbber.stop();
          setLoadingState(false, this.store);
          this.baselineData = null;
          this.data = null;
          this.process(null);
        },
      });
  }

  /**
   * Runs `callback` once the Google Charts packages have loaded.
   *
   * `process()` and the charts bound to `data` build
   * `google.visualization.DataTable`s. When HLO Op Stats is the first page
   * opened, its data can arrive before the packages load.
   */
  private whenChartsLoaded(callback: () => void) {
    this.cancelChartLoad?.();
    this.cancelChartLoad = undefined;
    if (
      typeof google !== 'undefined' &&
      typeof google.visualization?.DataTable === 'function'
    ) {
      callback();
      return;
    }
    // The page may be closed or superseded while the loader script and
    // packages load.
    this.cancelChartLoad = loadGoogleChartPackages(
      ['corechart', 'table'],
      () => {
        this.cancelChartLoad = undefined;
        this.zone.run(callback);
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
      active.cols.length === 0 ||
      !active.rows ||
      !baseline.cols ||
      baseline.cols.length === 0 ||
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
        label: 'Baseline Total Time (us)',
        type: 'number',
      });
    }
    const baselineSelfTimeColIndex = selfTimeCol !== -1 ? newCols.length : -1;
    if (selfTimeCol !== -1) {
      newCols.push({
        id: `${SELF_TIME_ID}_baseline`,
        label: 'Baseline Total Self Time (us)',
        type: 'number',
      });
    }
    const baselineFlopRateColIndex = flopRateCol !== -1 ? newCols.length : -1;
    if (flopRateCol !== -1) {
      newCols.push({
        id: `${MEASURED_FLOP_RATE_ID}_baseline`,
        label: 'Baseline Model GFLOP/s',
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
        label: 'Baseline Average Time (us)',
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

  onCheckInputParams(queryParams: Params = this.latestQueryParams) {
    const searchParams = this.dataService.getSearchParams?.();
    const rawHloOpName =
      getSingleQueryParam(queryParams['hlo_op_name']) ??
      searchParams?.get('hlo_op_name') ??
      '';
    this.hloOpNameSelected = rawHloOpName;

    const rawProgramId =
      getSingleQueryParam(queryParams['program_id']) ??
      searchParams?.get('program_id') ??
      '';
    this.programIdSelected = parseProgramId(rawProgramId);
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
      !data.rows ||
      programIdColumnIdx === -1 ||
      hloOpExpressionColumnIdx === -1 ||
      hloOpNameColumnIdx === -1
    ) {
      return data;
    }

    const updatedData = {
      ...data,
      rows: data.rows.map((row) => {
        if (!row.c) {
          return row;
        }
        const programId = String(row.c[programIdColumnIdx]?.v ?? '').trim();
        const hloOpName = String(row.c[hloOpNameColumnIdx]?.v ?? '').trim();
        const exprCell = row.c[hloOpExpressionColumnIdx];
        const hloOpExpression = String(exprCell?.v ?? '');
        const graphViewerLink =
          programId && programId !== '0' && hloOpName
            ? this.dataService.getGraphViewerLink(
                this.sessionId,
                '',
                hloOpName,
                programId,
              )
            : '';
        const formattedCell = graphViewerLink
          ? {
              ...exprCell,
              v: hloOpExpression,
              f:
                `<a href="${escapeHtml(graphViewerLink)}" target="_blank">` +
                `${escapeHtml(hloOpExpression)}</a>`,
            }
          : {
              ...exprCell,
              v: hloOpExpression,
            };
        return {
          ...row,
          c: [
            ...row.c.slice(0, hloOpExpressionColumnIdx),
            formattedCell,
            ...row.c.slice(hloOpExpressionColumnIdx + 1),
          ],
        };
      }),
    };
    return updatedData;
  }

  private syncSourceInfoClickListeners() {
    this.syncSourceInfoClickListener('table', this.chartElementRefInternal);
    this.syncSourceInfoClickListener(
      'sparseCoreTable',
      this.sparseCoreChartElementRefInternal,
    );
  }

  /**
   * Attaches a single delegated click listener to a rendered table container.
   *
   * `google.visualization.Table` does not provide a cell-click API, so clicks
   * on `.source-info-cell` elements bubble to the host element regardless of
   * table redraws or chart load timing.
   */
  private syncSourceInfoClickListener(
    key: 'table' | 'sparseCoreTable',
    chartElementRef: ElementRef<HTMLElement> | undefined,
  ) {
    const chartElement = this.sourceCodeServiceIsAvailable
      ? chartElementRef?.nativeElement
      : undefined;
    const existing = this.sourceInfoClickListeners.get(key);
    if (existing?.element === chartElement) {
      return;
    }
    if (existing) {
      existing.unlisten();
      this.sourceInfoClickListeners.delete(key);
    }
    if (!chartElement) {
      return;
    }
    const unlisten = this.renderer.listen(
      chartElement,
      'click',
      (event: Event) => {
        const target = event.target;
        const cell =
          target instanceof Element
            ? target.closest('.source-info-cell')
            : null;
        if (cell) {
          this.zone.run(() => {
            this.sourceFileAndLineNumber = cell.textContent || '';
            this.stackTrace = cell.getAttribute('title') || '';
          });
        }
      },
    );
    this.sourceInfoClickListeners.set(key, {element: chartElement, unlisten});
  }

  private disposeSourceInfoClickListeners() {
    for (const listener of this.sourceInfoClickListeners.values()) {
      listener.unlisten();
    }
    this.sourceInfoClickListeners.clear();
  }

  toggleShowStackTrace() {
    this.showStackTrace = !this.showStackTrace;
  }

  private clearData() {
    this.baselineData = null;
    this.tensorCoreBaselineData = null;
    this.flopRateChartXColumn = -1;
    this.flopRateChartYColumn = -1;
    this.parseData(null);

    this.dataInfoForTable.dataProvider.setVisibleColumns([]);
    this.dataInfoForTable = {
      ...this.dataInfoForTable,
      data: null,
      filters: [],
    };

    this.dataInfoForSparseCoreTable.dataProvider.setVisibleColumns([]);
    this.dataInfoForSparseCoreTable.dataProvider.parseData(null);
    this.dataInfoForSparseCoreTable = {
      ...this.dataInfoForSparseCoreTable,
      data: null,
      filters: [],
    };
    this.hasSparseCoreData = false;
    this.sourceFileAndLineNumber = '';
    this.stackTrace = '';
  }

  private process(data: SimpleDataTable | null) {
    if (!data || !data.cols || data.cols.length === 0) {
      this.data = null;
      this.clearData();
      return;
    }

    const coreTypeIdx =
      data.cols?.findIndex((col) => col.id === CORE_TYPE_ID) ?? -1;
    let tensorCoreDataRows = data.rows;
    let sparseCoreRows: google.visualization.DataObjectRow[] = [];

    if (coreTypeIdx !== -1) {
      tensorCoreDataRows =
        data.rows?.filter(
          (row) => row.c?.[coreTypeIdx]?.v !== SPARSE_CORE_VALUE,
        ) || [];
      sparseCoreRows =
        data.rows?.filter(
          (row) => row.c?.[coreTypeIdx]?.v === SPARSE_CORE_VALUE,
        ) || [];
    }

    const tensorCoreData = {...data, rows: tensorCoreDataRows};
    const sparseCoreData = {...data, rows: sparseCoreRows};

    let tensorCoreBaselineData: SimpleDataTable | null = null;
    if (
      this.baselineData &&
      this.baselineData.cols &&
      this.baselineData.cols.length > 0 &&
      this.baselineData.rows
    ) {
      const baselineCoreTypeIdx = this.baselineData.cols.findIndex(
        (col) => col.id === CORE_TYPE_ID,
      );
      let tensorCoreBaselineRows = this.baselineData.rows;
      if (baselineCoreTypeIdx !== -1) {
        tensorCoreBaselineRows = this.baselineData.rows.filter(
          (row) =>
            row.c?.[baselineCoreTypeIdx] != null &&
            row.c[baselineCoreTypeIdx].v !== SPARSE_CORE_VALUE,
        );
      }
      tensorCoreBaselineData = {
        ...this.baselineData,
        rows: tensorCoreBaselineRows,
      };
    }
    this.tensorCoreBaselineData = tensorCoreBaselineData;

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
      this.replicaGroupDataProvider.opCategoryIndex === undefined ||
      this.replicaGroupDataProvider.hloOpNameIndex === undefined ||
      this.replicaGroupDataProvider.selfTimeIndex === undefined
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
    const nextTableColumns: Array<{id: string; index: number; label: string}> =
      [];
    const nextColumnIds = new Set<string>();
    const defaultVisibleIds: string[] = [];
    const numColumns = dataTable.getNumberOfColumns();
    for (let i = 0; i < numColumns; i++) {
      const colId = dataTable.getColumnId(i) || String(i);
      if (colId === CORE_TYPE_ID) continue;
      nextTableColumns.push({
        id: colId,
        index: i,
        label: dataTable.getColumnLabel(i),
      });
      nextColumnIds.add(colId);
      if (DEFAULT_VISIBLE_COLUMN_IDS.has(colId)) {
        defaultVisibleIds.push(colId);
      }
    }
    this.tableColumns = nextTableColumns;

    const currentSelectedIds = this.tableColumnsControl.value ?? [];
    let nextSelectedIds: string[];
    if (currentSelectedIds.length === 0 || this.knownColumnIds.size === 0) {
      nextSelectedIds = defaultVisibleIds;
    } else {
      const keptIds = currentSelectedIds.filter((id) => nextColumnIds.has(id));
      if (keptIds.length === 0) {
        nextSelectedIds = defaultVisibleIds;
      } else {
        const selectedSet = new Set(keptIds);
        const newlyAddedBaselineIds = nextTableColumns
          .map((col) => col.id)
          .filter(
            (id) =>
              BASELINE_DEFAULT_VISIBLE_COLUMN_IDS.has(id) &&
              !this.knownColumnIds.has(id) &&
              !selectedSet.has(id),
          );
        nextSelectedIds = [...keptIds, ...newlyAddedBaselineIds];
      }
    }

    this.knownColumnIds = nextColumnIds;
    this.tableColumnsControl.setValue(nextSelectedIds, {emitEvent: false});
    this.applyVisibleColumns(nextSelectedIds, /* notify= */ false);
  }

  updateTableColumns(newValue: string[]) {
    if (newValue.length === 0 || !this.dataTable) return;
    this.applyVisibleColumns(newValue, /* notify= */ true);
  }

  private applyVisibleColumns(selectedIds: string[], notify: boolean) {
    if (!this.dataTable) return;

    const selectedIdSet = new Set(selectedIds);
    const numColumns = this.dataTable.getNumberOfColumns();
    const coreTypeIdx = this.dataTable.getColumnIndex(CORE_TYPE_ID);
    const rankIdx = this.dataTable.getColumnIndex(RANK_ID);

    const visibleColumns = this.tableColumns
      .filter(
        (col) =>
          selectedIdSet.has(col.id) &&
          col.index >= 0 &&
          col.index < numColumns &&
          col.index !== coreTypeIdx,
      )
      .map((col) => col.index);

    this.dataInfoForTable.dataProvider.setVisibleColumns(visibleColumns);

    const sparseCoreVisibleColumns = visibleColumns.filter(
      (index) => index !== rankIdx,
    );
    this.dataInfoForSparseCoreTable.dataProvider.setVisibleColumns(
      sparseCoreVisibleColumns,
    );

    if (notify) {
      this.dataInfoForTable.dataProvider.notifyCharts();
      this.dataInfoForSparseCoreTable.dataProvider.notifyCharts();
    }
  }

  override parseData(data: SimpleDataTable | null) {
    if (!data) {
      this.pendingData = null;
      this.dataTable = undefined;
      this.dataView = undefined;
      this.filters.clear();
      this.tableColumns = [];
      this.knownColumnIds.clear();
      this.tableColumnsControl.setValue([], {emitEvent: false});

      this.pieChartDataProvider.parseData(null);
      this.pieChartDataProvider.notifyCharts();

      this.replicaGroupDataProvider.parseData(null);
      this.communicationOps = new Set<string>();
      this.selectedCommOp = '';
      this.replicaGroupDataProvider.notifyCharts();
      return;
    }
    // Five charts share one DataProvider. In order to prevent DataTable from
    // being created multiple times, it calls DataProvider function directly.
    this.pieChartDataProvider.parseData(data);
    const dataTable = this.pieChartDataProvider.getDataTable();
    if (!dataTable) return;

    const previousDataTable = this.dataTable;
    this.dataTable = dataTable;
    this.pruneFilters(previousDataTable);
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
      this.tensorCoreBaselineData
        ? new CategoryDiffTableDataProcessor(
            this.tensorCoreBaselineData,
            [],
            opCategoryIndex,
            selfTimeIndex,
          )
        : new CategoryTableDataProcessor([], opCategoryIndex, selfTimeIndex);
    this.dataInfoOpChart.customChartDataProcessor =
      this.tensorCoreBaselineData
        ? new CategoryDiffTableDataProcessor(
            this.tensorCoreBaselineData,
            [],
            hloOpNameIndex,
            selfTimeIndex,
          )
        : new CategoryTableDataProcessor([], hloOpNameIndex, selfTimeIndex);
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
    // is truncated to only be the 'replica_groups=...' string.
    this.replicaGroupDataProvider.parseData(data);
    this.communicationOps = this.replicaGroupDataProvider.communicationOps;

    if (this.communicationOps.size > 0) {
      if (!this.communicationOps.has(this.selectedCommOp)) {
        // Set value to the first communication Op in the set.
        this.selectedCommOp = this.communicationOps.values().next().value ?? '';
      }
    } else {
      this.selectedCommOp = '';
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
    // Unsubscribes all pending subscriptions and listeners.
    this.cancelChartLoad?.();
    this.cancelChartLoad = undefined;
    this.disposeSourceInfoClickListeners();
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
