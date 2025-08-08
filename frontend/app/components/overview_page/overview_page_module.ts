import {CommonModule} from '@angular/common';
import {Component, EventEmitter, inject, Input, NgModule, OnDestroy, Output} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {type GeneralAnalysis, type InputPipelineAnalysis, type OverviewPageDataTuple, type RunEnvironment, type SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {type Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {parseDiagnosticsDataTable, setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';
import {InferenceLatencyChartModule} from 'org_xprof/frontend/app/components/overview_page/inference_latency_chart/inference_latency_chart_module';
import {PerformanceSummaryModule} from 'org_xprof/frontend/app/components/overview_page/performance_summary/performance_summary_module';
import {RunEnvironmentViewModule} from 'org_xprof/frontend/app/components/overview_page/run_environment_view/run_environment_view_module';
import {StepTimeGraphModule} from 'org_xprof/frontend/app/components/overview_page/step_time_graph/step_time_graph_module';
import {DATA_SERVICE_INTERFACE_TOKEN, type DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';
import {SmartSuggestionView} from 'org_xprof/frontend/app/components/smart_suggestion/smart_suggestion_view';

const GENERAL_ANALYSIS_INDEX = 0;
const INPUT_PIPELINE_ANALYSIS_INDEX = 1;
const RUN_ENVIRONMENT_INDEX = 2;
const INFERENCE_LATENCY_CHART_INDEX = 4;
const DIAGNOSTICS_INDEX = 6;

/** An overview page component. */
@Component({
  standalone: false,
  selector: 'overview-page',
  templateUrl: './overview_page.ng.html',
  styleUrls: ['./overview_page.css']
})
export class OverviewPage implements OnDestroy {
  @Input() darkTheme = false;
  @Output()
  readonly onDataLoaded = new EventEmitter<OverviewPageDataTuple|null>();

  diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
  generalAnalysis: GeneralAnalysis|null = null;
  inputPipelineAnalysis: InputPipelineAnalysis|null = null;
  runEnvironment: RunEnvironment|null = null;
  inferenceLatencyData: SimpleDataTable|null = null;

  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);
  sessionId = '';
  tool = 'overview_page';
  host = '';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  private readonly route: ActivatedRoute = inject(ActivatedRoute);
  private readonly store: Store = inject(Store);

  constructor() {
    this.route.params.pipe(takeUntil(this.destroyed))
        .subscribe((params: Params) => {
          this.processQuery(params);
          this.update();
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

  processQuery(params: Params) {
    this.host = params['host'] || this.host || '';
    this.sessionId = params['run'] || params['sessionId'] || '';
    this.tool = params['tag'] || 'overview_page';
  }

  update() {
    setLoadingState(true, this.store, 'Loading overview data');

    this.dataService.getData(this.sessionId, this.tool, this.host)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          setLoadingState(false, this.store);
          this.onDataLoaded.emit(data as OverviewPageDataTuple);
          data = (data || []) as OverviewPageDataTuple;
          /** Transfer data to Overview Page DataTable type */
          this.parseOverviewPageData(data as OverviewPageDataTuple);
        });
  }

  parseOverviewPageData(data: OverviewPageDataTuple) {
    this.generalAnalysis = data[GENERAL_ANALYSIS_INDEX];
    this.inputPipelineAnalysis = data[INPUT_PIPELINE_ANALYSIS_INDEX];
    this.runEnvironment = data[RUN_ENVIRONMENT_INDEX];
    if (data.length > INFERENCE_LATENCY_CHART_INDEX + 1) {
      this.inferenceLatencyData = data[INFERENCE_LATENCY_CHART_INDEX];
    }
    this.diagnostics = parseDiagnosticsDataTable(data[DIAGNOSTICS_INDEX]);
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
  exports: [OverviewPage]
})
export class OverviewPageModule {
}
