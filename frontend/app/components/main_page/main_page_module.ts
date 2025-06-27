import {NgModule} from '@angular/core';
import {Routes, RouterModule} from '@angular/router';
import {EmptyPage} from 'org_xprof/frontend/app/components/empty_page/empty_page';
import {FrameworkOpStatsAdapter} from 'org_xprof/frontend/app/components/framework_op_stats/framework_op_stats_adapter';
import {GraphViewer} from 'org_xprof/frontend/app/components/graph_viewer/graph_viewer';
import {HloStats} from 'org_xprof/frontend/app/components/hlo_stats/hlo_stats';
import {InferenceProfile} from 'org_xprof/frontend/app/components/inference_profile/inference_profile';
import {InputPipeline} from 'org_xprof/frontend/app/components/input_pipeline/input_pipeline';
import {KernelStatsAdapter} from 'org_xprof/frontend/app/components/kernel_stats/kernel_stats_adapter';
import {MegascalePerfetto} from 'org_xprof/frontend/app/components/megascale_perfetto/megascale_perfetto';
import {MegascaleStats} from 'org_xprof/frontend/app/components/megascale_stats/megascale_stats';
import {MemoryProfile} from 'org_xprof/frontend/app/components/memory_profile/memory_profile';
import {MemoryViewer} from 'org_xprof/frontend/app/components/memory_viewer/memory_viewer';
import {OpProfile} from 'org_xprof/frontend/app/components/op_profile/op_profile';
import {OverviewPage} from 'org_xprof/frontend/app/components/overview_page/overview_page_module';
import {PerfCounters} from 'org_xprof/frontend/app/components/perf_counters/perf_counters';
import {PodViewer} from 'org_xprof/frontend/app/components/pod_viewer/pod_viewer';
import {RooflineModel} from 'org_xprof/frontend/app/components/roofline_model/roofline_model';
import {StackTracePage} from 'org_xprof/frontend/app/components/stack_trace_page/stack_trace_page';
import {TraceViewer} from 'org_xprof/frontend/app/components/trace_viewer/trace_viewer';
import {UtilizationViewer} from 'org_xprof/frontend/app/components/utilization_viewer/utilization_viewer';

import {MainPage} from './main_page';
export { MainPage } from './main_page';

/** The list of all routes available in the application. */
export const routes: Routes = [
  {path: 'empty', component: EmptyPage},
  {path: 'overview_page', component: OverviewPage},
  {path: 'input_pipeline_analyzer', component: InputPipeline},
  {path: 'kernel_stats', component: KernelStatsAdapter},
  {path: 'memory_profile', component: MemoryProfile},
  {path: 'memory_viewer', component: MemoryViewer},
  {path: 'op_profile', component: OpProfile},
  {path: 'pod_viewer', component: PodViewer},
  {path: 'framework_op_stats', component: FrameworkOpStatsAdapter},
  {path: 'trace_viewer', component: TraceViewer},
  {path: 'trace_viewer@', component: TraceViewer},
  {path: 'utilization_viewer', component: UtilizationViewer},
  {path: 'graph_viewer', component: GraphViewer},
  {path: 'megascale_perfetto/:sessionId', component: MegascalePerfetto},
  {path: 'megascale_stats', component: MegascaleStats},
  {path: 'perf_counters', component: PerfCounters},
  {path: 'inference_profile', component: InferenceProfile},
  {path: 'hlo_stats', component: HloStats},
  {path: 'roofline_model', component: RooflineModel},
  {path: 'stack_trace_page/:sessionId', component: StackTracePage},
  {path: '**', component: EmptyPage},
];

/** A main page module. */
@NgModule({
  imports: [
    MainPage,
    RouterModule.forRoot(routes),
  ],
  exports: [MainPage]
})
export class MainPageModule {
}
