import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';
import {InferenceLatencyChartModule} from 'org_xprof/frontend/app/components/overview_page/inference_latency_chart/inference_latency_chart_module';
import {PerformanceSummaryModule} from 'org_xprof/frontend/app/components/overview_page/performance_summary/performance_summary_module';
import {RunEnvironmentViewModule} from 'org_xprof/frontend/app/components/overview_page/run_environment_view/run_environment_view_module';
import {StepTimeGraphModule} from 'org_xprof/frontend/app/components/overview_page/step_time_graph/step_time_graph_module';
import {SmartSuggestionView} from 'org_xprof/frontend/app/components/smart_suggestion/smart_suggestion_view';
import {OverviewPage} from './overview_page';

/** An overview page module. */
@NgModule({
  imports: [
    CommonModule,
    DiagnosticsViewModule,
    PerformanceSummaryModule,
    RunEnvironmentViewModule,
    StepTimeGraphModule,
    InferenceLatencyChartModule,
    SmartSuggestionView,
    OverviewPage,
  ],
  exports: [OverviewPage],
})
export class OverviewPageModule {}
export {OverviewPage} from './overview_page';
