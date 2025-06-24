import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatSidenavModule} from '@angular/material/sidenav';
import {MatToolbarModule} from '@angular/material/toolbar';
import {RouterModule, Routes} from '@angular/router';
import {DiagnosticsViewModule} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view_module';
import {EmptyPage} from 'org_xprof/frontend/app/components/empty_page/empty_page';
import {EmptyPageModule} from 'org_xprof/frontend/app/components/empty_page/empty_page_module';
import {FrameworkOpStatsAdapter, FrameworkOpStatsAdapterModule} from 'org_xprof/frontend/app/components/framework_op_stats/framework_op_stats_adapter';
import {GraphViewer} from 'org_xprof/frontend/app/components/graph_viewer/graph_viewer';
import {GraphViewerModule} from 'org_xprof/frontend/app/components/graph_viewer/graph_viewer_module';
import {HloStats} from 'org_xprof/frontend/app/components/hlo_stats/hlo_stats';
import {HloStatsModule} from 'org_xprof/frontend/app/components/hlo_stats/hlo_stats_module';
import {InferenceProfile} from 'org_xprof/frontend/app/components/inference_profile/inference_profile';
import {InferenceProfileModule} from 'org_xprof/frontend/app/components/inference_profile/inference_profile_module';
import {InputPipeline} from 'org_xprof/frontend/app/components/input_pipeline/input_pipeline';
import {InputPipelineModule} from 'org_xprof/frontend/app/components/input_pipeline/input_pipeline_module';
import {KernelStatsAdapter, KernelStatsAdapterModule} from 'org_xprof/frontend/app/components/kernel_stats/kernel_stats_adapter';
import {MegascaleStats} from 'org_xprof/frontend/app/components/megascale_stats/megascale_stats';
import {MegascaleStatsModule} from 'org_xprof/frontend/app/components/megascale_stats/megascale_stats_module';
import {MemoryProfile} from 'org_xprof/frontend/app/components/memory_profile/memory_profile';
import {MemoryProfileModule} from 'org_xprof/frontend/app/components/memory_profile/memory_profile_module';
import {MemoryViewer} from 'org_xprof/frontend/app/components/memory_viewer/memory_viewer';
import {MemoryViewerModule} from 'org_xprof/frontend/app/components/memory_viewer/memory_viewer_module';
import {OpProfile} from 'org_xprof/frontend/app/components/op_profile/op_profile';
import {OpProfileModule} from 'org_xprof/frontend/app/components/op_profile/op_profile_module';
import {OverviewPage, OverviewPageModule} from 'org_xprof/frontend/app/components/overview_page/overview_page_module';
import {PodViewer} from 'org_xprof/frontend/app/components/pod_viewer/pod_viewer';
import {PodViewerModule} from 'org_xprof/frontend/app/components/pod_viewer/pod_viewer_module';
import {RooflineModel} from 'org_xprof/frontend/app/components/roofline_model/roofline_model';
import {RooflineModelModule} from 'org_xprof/frontend/app/components/roofline_model/roofline_model_module';
import {SideNavModule} from 'org_xprof/frontend/app/components/sidenav/sidenav_module';
import {StackTracePage} from 'org_xprof/frontend/app/components/stack_trace_page/stack_trace_page';
import {StackTracePageModule} from 'org_xprof/frontend/app/components/stack_trace_page/stack_trace_page_module';
import {TraceViewer} from 'org_xprof/frontend/app/components/trace_viewer/trace_viewer';
import {TraceViewerModule} from 'org_xprof/frontend/app/components/trace_viewer/trace_viewer_module';

import {MainPage} from './main_page';

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
  {path: 'graph_viewer', component: GraphViewer},
  {path: 'megascale_stats', component: MegascaleStats},
  {path: 'inference_profile', component: InferenceProfile},
  {path: 'hlo_stats', component: HloStats},
  {path: 'roofline_model', component: RooflineModel},
  {path: 'stack_trace_page/:sessionId', component: StackTracePage},
  {path: '**', component: EmptyPage},
];

/** A main page module. */
@NgModule({
  declarations: [MainPage],
  imports: [
    CommonModule,
    MatProgressBarModule,
    MatSidenavModule,
    MatToolbarModule,
    MatIconModule,
    EmptyPageModule,
    SideNavModule,
    TraceViewerModule,
    OverviewPageModule,
    InputPipelineModule,
    KernelStatsAdapterModule,
    MemoryProfileModule,
    MemoryViewerModule,
    OpProfileModule,
    PodViewerModule,
    GraphViewerModule,
    FrameworkOpStatsAdapterModule,
    MegascaleStatsModule,
    HloStatsModule,
    RooflineModelModule,
    InferenceProfileModule,
    DiagnosticsViewModule,
    RouterModule.forRoot(routes),
    StackTracePageModule,
  ],
  exports: [MainPage]
})
export class MainPageModule {
}
