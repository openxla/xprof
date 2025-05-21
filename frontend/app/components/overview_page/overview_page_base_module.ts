import {CommonModule} from '@angular/common';
import {Component, Input, NgModule} from '@angular/core';
import {type GeneralAnalysis, type InputPipelineAnalysis, type RunEnvironment, type SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {type Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';
import {InferenceLatencyChartModule} from 'org_xprof/frontend/app/components/overview_page/inference_latency_chart/inference_latency_chart_module';
import {NormalizedAcceleratorPerformanceViewModule} from 'org_xprof/frontend/app/components/overview_page/normalized_accelerator_performance_view/normalized_accelerator_performance_view_module';
import {PerformanceSummaryModule} from 'org_xprof/frontend/app/components/overview_page/performance_summary/performance_summary_module';
import {RunEnvironmentViewModule} from 'org_xprof/frontend/app/components/overview_page/run_environment_view/run_environment_view_module';
import {StepTimeGraphModule} from 'org_xprof/frontend/app/components/overview_page/step_time_graph/step_time_graph_module';

/** A base overview page component. */
@Component({
  standalone: false,
  selector: 'overview-page-base',
  templateUrl: './overview_page_base.ng.html',
  styleUrls: ['./overview_page.css']
})
export class OverviewPageBase {
  @Input() darkTheme = false;
  @Input() diagnostics: Diagnostics|null = null;
  @Input() generalAnalysis: GeneralAnalysis|null = null;
  @Input() inputPipelineAnalysis: InputPipelineAnalysis|null = null;
  @Input() runEnvironment: RunEnvironment|null = null;
  @Input() inferenceLatencyData: SimpleDataTable|null = null;

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
}

/** An overview page base module. */
@NgModule({
  declarations: [OverviewPageBase],
  imports: [
    CommonModule,
    DiagnosticsViewModule,
    PerformanceSummaryModule,
    RunEnvironmentViewModule,
    StepTimeGraphModule,
    NormalizedAcceleratorPerformanceViewModule,
    InferenceLatencyChartModule,
  ],
  exports: [OverviewPageBase]
})
export class OverviewPageBaseModule {
}
