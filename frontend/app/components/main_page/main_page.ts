import {CommonModule} from '@angular/common';
import {Component, inject, OnDestroy} from '@angular/core';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatSidenavModule} from '@angular/material/sidenav';
import {MatToolbarModule} from '@angular/material/toolbar';
import {RouterModule} from '@angular/router';
import {DiagnosticsView} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view';
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
import {SideNav} from 'org_xprof/frontend/app/components/sidenav/sidenav';
import {StackTracePage} from 'org_xprof/frontend/app/components/stack_trace_page/stack_trace_page';
import {TraceViewer} from 'org_xprof/frontend/app/components/trace_viewer/trace_viewer';
import {UtilizationViewer} from 'org_xprof/frontend/app/components/utilization_viewer/utilization_viewer';
import {Store} from '@ngrx/store';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {CommunicationService} from 'org_xprof/frontend/app/services/communication_service/communication_service';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {getErrorMessage, getLoadingState, } from 'org_xprof/frontend/app/store/selectors';
import {LoadingState} from 'org_xprof/frontend/app/store/state';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A main page component. */
@Component({
  standalone: true,
  selector: 'main-page',
  templateUrl: 'main_page.ng.html',
  styleUrls: ['main_page.scss'],
  imports: [
    CommonModule,
    DiagnosticsView,
    EmptyPage,
    FrameworkOpStatsAdapter,
    GraphViewer,
    HloStats,
    InferenceProfile,
    InputPipeline,
    KernelStatsAdapter,
    MatIconModule,
    MatProgressBarModule,
    MatSidenavModule,
    MatToolbarModule,
    MegascalePerfetto,
    MegascaleStats,
    MemoryProfile,
    MemoryViewer,
    OpProfile,
    OverviewPage,
    PerfCounters,
    PodViewer,
    RooflineModel,
    RouterModule,
    SideNav,
    StackTracePage,
    TraceViewer,
    UtilizationViewer,
  ],
})
export class MainPage implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  loading = true;
  loadingMessage = '';
  isSideNavOpen = true;
  navigationReady = false;
  errorMessage = '';
  /** The version string of the XProf plugin. */
  pluginVersion = '';

  constructor(
      store: Store<{}>,
      private readonly communicationService: CommunicationService,
  ) {
    window.sessionStorage.setItem(
        'searchParams',
        new URLSearchParams(window.location.search).toString(),
    );
    store.select(getLoadingState)
        .pipe(takeUntil(this.destroyed))
        .subscribe((loadingState: LoadingState) => {
          this.loading = loadingState.loading;
          this.loadingMessage = loadingState.message;
        });
    store.select(getErrorMessage)
        .pipe(takeUntil(this.destroyed))
        .subscribe((errorMessage: string) => {
          this.errorMessage = errorMessage;
        });
    this.communicationService.navigationReady.subscribe(
        (navigationEvent: NavigationEvent) => {
          this.navigationReady = true;
          // TODO(fe-unification): Remove this constraint once the sidepanel
          // content of the 3 tools are moved out from sidenav with consolidated
          // templates.
          const toolsWithSideNav =
              ['op_profile', 'memory_viewer', 'pod_viewer'];
          this.isSideNavOpen =
              (navigationEvent.firstLoad ||
               toolsWithSideNav
                       .filter(tool => navigationEvent?.tag?.startsWith(tool))
                       .length > 0);
        });
    this.dataService.getPluginVersion()
        .pipe(takeUntil(this.destroyed))
        .subscribe((version: string | null) => {
          this.pluginVersion = version || '';
        });
  }

  get diagnostics(): Diagnostics {
    return {
      errors: this.errorMessage ? [this.errorMessage] : [],
      info: [],
      warnings: [],
    };
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
