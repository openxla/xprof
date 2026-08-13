import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  inject,
  Input,
  NgModule,
  OnDestroy,
  Output,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {
  type GeneralAnalysis,
  type InputPipelineAnalysis,
  type OverviewPageDataTuple,
  type RunEnvironment,
  type SimpleDataTable,
} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {type Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {
  parseDiagnosticsDataTable,
  setLoadingState,
} from 'org_xprof/frontend/app/common/utils/utils';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';
import {InferenceLatencyChartModule} from 'org_xprof/frontend/app/components/overview_page/inference_latency_chart/inference_latency_chart_module';
import {PerformanceSummaryModule} from 'org_xprof/frontend/app/components/overview_page/performance_summary/performance_summary_module';
import {RunEnvironmentViewModule} from 'org_xprof/frontend/app/components/overview_page/run_environment_view/run_environment_view_module';
import {StepTimeGraphModule} from 'org_xprof/frontend/app/components/overview_page/step_time_graph/step_time_graph_module';
import {SmartSuggestionView} from 'org_xprof/frontend/app/components/smart_suggestion/smart_suggestion_view';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  type DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {BaseDiffService} from 'org_xprof/frontend/app/services/data_service_v2/diff_service';
import {BehaviorSubject, combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

const GENERAL_ANALYSIS_INDEX = 0;
const INPUT_PIPELINE_ANALYSIS_INDEX = 1;
const RUN_ENVIRONMENT_INDEX = 2;
const INFERENCE_LATENCY_CHART_INDEX = 4;
const DIAGNOSTICS_INDEX = 6;
const DISAGGREGATED_SERVING_LATENCY_INDEX = 8;

/** An overview page component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'overview-page',
  templateUrl: './overview_page.ng.html',
  styleUrls: ['./overview_page.scss'],
})
export class OverviewPage implements OnDestroy {
  @Input() darkTheme = false;
  @Output()
  readonly onDataLoaded = new EventEmitter<OverviewPageDataTuple | null>();
  @Output() readonly ready = new EventEmitter<void>();

  diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
  generalAnalysis: GeneralAnalysis | null = null;
  inputPipelineAnalysis: InputPipelineAnalysis | null = null;
  runEnvironment: RunEnvironment | null = null;
  inferenceLatencyData: SimpleDataTable | null = null;
  disaggregatedServingLatencyData: SimpleDataTable | null = null;

  baselineGeneralAnalysis: GeneralAnalysis | null = null;
  baselineInputPipelineAnalysis: InputPipelineAnalysis | null = null;
  baselineInferenceLatencyData: SimpleDataTable | null = null;
  baselineDisaggregatedServingLatencyData: SimpleDataTable | null = null;

  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );
  private readonly diffService = inject(BaseDiffService);
  sessionId = '';
  baseSessionId = '';
  tool = 'overview_page';
  host = '';
  isLoaded = false;
  enableSmartSuggestion = false;
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  private readonly readyChartsCount = new BehaviorSubject<number>(0);

  private readonly route: ActivatedRoute = inject(ActivatedRoute);
  private readonly store: Store = inject(Store);

  constructor() {
    this.readyChartsCount.pipe(takeUntil(this.destroyed)).subscribe(() => {
      this.checkReady();
    });
    combineLatest([this.route.params, this.route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        const oldSessionId = this.sessionId;
        const oldTool = this.tool;
        const oldHost = this.host;
        const oldBaseSessionId = this.baseSessionId;

        this.sessionId = params['sessionId'] || this.sessionId;
        this.processQueryParams(queryParams);

        // Trigger update only if the parameters actually changed.
        const hasChanged =
          this.sessionId !== oldSessionId ||
          this.tool !== oldTool ||
          this.host !== oldHost ||
          this.baseSessionId !== oldBaseSessionId;
        if (hasChanged) {
          this.update();
        }
      });
  }

  get isTrainingString(): string {
    return this.runEnvironment?.p?.['is_training'] || '';
  }

  get isInference(): boolean {
    return this.isTrainingString === 'false';
  }

  get hasInferenceLatencyData(): boolean {
    return this.isInference && !!this.inferenceLatencyData?.rows?.length;
  }

  get hasStepTimeGraphData(): boolean {
    return !this.isInference;
  }

  processQueryParams(params: Params) {
    this.host = params['host'] || this.host || '';
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || 'overview_page';
    if (
      params['base_session_id'] !== undefined ||
      params['baseSessionID'] !== undefined
    ) {
      const paramBaseId =
        params['base_session_id'] || params['baseSessionID'] || '';
      if (paramBaseId !== this.diffService.getBaseSessionId()) {
        this.diffService.setBaseSessionId(paramBaseId);
      }
      this.baseSessionId = paramBaseId;
    } else {
      this.baseSessionId = this.diffService.getBaseSessionId() || '';
    }
    this.enableSmartSuggestion = this.dataService.isSmartSuggestionEnabled();
  }

  update() {
    setLoadingState(true, this.store, 'Loading overview data');
    this.isLoaded = false;

    this.diffService
      .getDiffData<OverviewPageDataTuple>(this.sessionId, this.tool, {
        baselineSessionId:
          this.baseSessionId || this.diffService.getBaseSessionId() || '',
        host: this.host,
      })
      .pipe(takeUntil(this.destroyed))
      .subscribe({
        next: ({active, baseline}) => {
          setLoadingState(false, this.store);
          this.onDataLoaded.emit(active as OverviewPageDataTuple);
          if (active) {
            this.parseOverviewPageData(active as OverviewPageDataTuple);
          }
          if (baseline) {
            this.parseBaselineOverviewPageData(
              baseline as OverviewPageDataTuple,
            );
          } else {
            this.clearBaselineData();
          }
          this.isLoaded = true;
        },
        error: () => {
          setLoadingState(false, this.store);
          this.isLoaded = true;
        },
      });
  }

  onChartReady() {
    this.readyChartsCount.next(this.readyChartsCount.value + 1);
  }

  private checkReady() {
    if (!this.isLoaded) {
      return;
    }
    let expectedCharts = 0;
    if (this.hasStepTimeGraphData) {
      expectedCharts++;
    }
    if (this.hasInferenceLatencyData) {
      expectedCharts++;
    }

    if (this.readyChartsCount.value >= expectedCharts) {
      this.ready.emit();
    }
  }

  parseOverviewPageData(data: OverviewPageDataTuple) {
    this.generalAnalysis = data[GENERAL_ANALYSIS_INDEX];
    this.inputPipelineAnalysis = data[INPUT_PIPELINE_ANALYSIS_INDEX];
    this.runEnvironment = data[RUN_ENVIRONMENT_INDEX];
    if (data.length > INFERENCE_LATENCY_CHART_INDEX + 1) {
      this.inferenceLatencyData = data[INFERENCE_LATENCY_CHART_INDEX];
    }
    if (data.length > DISAGGREGATED_SERVING_LATENCY_INDEX) {
      this.disaggregatedServingLatencyData =
        data[DISAGGREGATED_SERVING_LATENCY_INDEX];
    }
    this.diagnostics = parseDiagnosticsDataTable(data[DIAGNOSTICS_INDEX]);
  }

  parseBaselineOverviewPageData(data: OverviewPageDataTuple) {
    this.baselineGeneralAnalysis = data[GENERAL_ANALYSIS_INDEX];
    this.baselineInputPipelineAnalysis = data[INPUT_PIPELINE_ANALYSIS_INDEX];
    if (data.length > INFERENCE_LATENCY_CHART_INDEX + 1) {
      this.baselineInferenceLatencyData = data[INFERENCE_LATENCY_CHART_INDEX];
    }
    if (data.length > DISAGGREGATED_SERVING_LATENCY_INDEX) {
      this.baselineDisaggregatedServingLatencyData =
        data[DISAGGREGATED_SERVING_LATENCY_INDEX];
    }
  }

  clearBaselineData() {
    this.baselineGeneralAnalysis = null;
    this.baselineInputPipelineAnalysis = null;
    this.baselineInferenceLatencyData = null;
    this.baselineDisaggregatedServingLatencyData = null;
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}

/** An overview page module. */
@NgModule({
  declarations: [OverviewPage],
  imports: [
    CommonModule,
    DiagnosticsViewModule,
    PerformanceSummaryModule,
    RunEnvironmentViewModule,
    StepTimeGraphModule,
    InferenceLatencyChartModule,
    SmartSuggestionView,
  ],
  exports: [OverviewPage],
})
export class OverviewPageModule {}
