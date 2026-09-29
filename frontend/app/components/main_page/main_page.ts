import {
  ChangeDetectionStrategy,
  Component,
  inject,
  OnDestroy,
} from '@angular/core';
import {Router} from '@angular/router';
import {Store} from '@ngrx/store';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {RunToolsMap} from 'org_xprof/frontend/app/common/interfaces/tool';
import {CommunicationService} from 'org_xprof/frontend/app/services/communication_service/communication_service';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {
  getCurrentRun,
  getErrorMessage,
  getLoadingState,
  getRunToolsMap,
} from 'org_xprof/frontend/app/store/selectors';
import {LoadingState} from 'org_xprof/frontend/app/store/state';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A main page component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'main-page',
  templateUrl: './main_page.ng.html',
  styleUrls: ['./main_page.scss'],
})
export class MainPage implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );
  private readonly store: Store<{}> = inject(Store);
  private readonly communicationService = inject(CommunicationService);
  private readonly router = inject(Router);

  loading = true;
  loadingMessage = '';
  isSideNavOpen = true;
  navigationReady = false;
  errorMessages: string[] = [];
  /** The version string of the XProf plugin. */
  pluginVersion = '';

  isNewNavEnabled = false;
  currentRun = '';
  currentTag = '';
  runToolsMap: RunToolsMap = {};

  private readonly toolIconMap: {[key: string]: string} = {
    'overview_page': 'home',
    'trace_viewer': 'view_object_track',
    'trace_viewer@': 'view_object_track',
    'graph_viewer': 'graph_2',
    'op_profile': 'bar_chart',
    'hlo_stats': 'query_stats',
    'input_pipeline_analyzer': 'input',
    'kernel_stats': 'memory',
    'memory_profile': 'monitoring',
    'memory_viewer': 'overview_key',
    'roofline_model': 'stacked_line_chart',
    'pod_viewer': 'hive',
    'framework_op_stats': 'pie_chart',
    'inference_profile': 'avg_pace',
    'perf_counters': 'av_timer',
    'utilization_viewer': 'bar_chart_4_bars',
    'megascale_stats': 'monitoring',
  };

  private readonly toolsDisplayMap = new Map<string, string>([
    ['overview_page', 'Overview Page'],
    ['framework_op_stats', 'Framework Op Stats'],
    ['input_pipeline_analyzer', 'Input Pipeline Analysis'],
    ['memory_profile', 'Memory Profile'],
    ['pod_viewer', 'Pod Viewer'],
    ['op_profile', 'HLO Op Profile'],
    ['memory_viewer', 'Memory Viewer'],
    ['graph_viewer', 'Graph Viewer'],
    ['hlo_stats', 'HLO Op Stats'],
    ['inference_profile', 'Inference Profile'],
    ['roofline_model', 'Roofline Model'],
    ['kernel_stats', 'Kernel Stats'],
    ['trace_viewer', 'Trace Viewer'],
    ['megascale_stats', 'Megascale Viewer'],
    ['perf_counters', 'Perf Counters'],
    ['utilization_viewer', 'Utilization Viewer'],
  ]);

  constructor() {
    const searchParams = new URLSearchParams(window.location.search);
    const newNavParam = searchParams.get('new_nav');
    this.isNewNavEnabled = newNavParam === 'true' || newNavParam === '1';

    searchParams.delete('use_pb');
    if (searchParams.toString()) {
      window.sessionStorage.setItem('searchParams', searchParams.toString());
      this.dataService.setSearchParams(searchParams);
    }
    this.store
      .select(getLoadingState)
      .pipe(takeUntil(this.destroyed))
      .subscribe((loadingState: LoadingState) => {
        this.loading = loadingState.loading;
        this.loadingMessage = loadingState.message;
      });
    this.store
      .select(getErrorMessage)
      .pipe(takeUntil(this.destroyed))
      .subscribe((errorMessage: string) => {
        if (!errorMessage || this.errorMessages.includes(errorMessage)) {
          return;
        }
        this.errorMessages.push(errorMessage);
      });
    this.store
      .select(getRunToolsMap)
      .pipe(takeUntil(this.destroyed))
      .subscribe((runTools: RunToolsMap) => {
        this.runToolsMap = runTools || {};
      });
    this.store
      .select(getCurrentRun)
      .pipe(takeUntil(this.destroyed))
      .subscribe((run: string) => {
        this.currentRun = run || '';
      });
    this.communicationService.navigationReady
      .pipe(takeUntil(this.destroyed))
      .subscribe((navigationEvent: NavigationEvent) => {
        this.navigationReady = true;
        this.currentRun = navigationEvent.run || this.currentRun;
        this.currentTag = navigationEvent.tag || '';
        // TODO(fe-unification): Remove this constraint once the sidepanel
        // content of the 3 tools are moved out from sidenav with consolidated
        // templates.
        const toolsWithSideNav = [
          'op_profile',
          'memory_viewer',
          'pod_viewer',
          'megascale_stats',
        ];
        this.isSideNavOpen =
          navigationEvent.firstLoad ||
          toolsWithSideNav.filter((tool) =>
            navigationEvent?.tag?.startsWith(tool),
          ).length > 0;
      });
    this.dataService
      .getPluginVersion()
      .pipe(takeUntil(this.destroyed))
      .subscribe((version: string | null) => {
        this.pluginVersion = version || '';
      });
  }

  get availableTools(): string[] {
    return this.runToolsMap[this.currentRun] || [];
  }

  getToolIcon(tag: string): string {
    return this.toolIconMap[tag] || 'dashboard';
  }

  getToolLabel(tag: string): string {
    const cleanTag = tag && tag.endsWith('@') ? tag.slice(0, -1) : tag || '';
    return this.toolsDisplayMap.get(cleanTag) || cleanTag;
  }

  selectTool(tag: string) {
    this.currentTag = tag;
    this.router.navigate([tag || 'empty'], {
      queryParams: {
        'run': this.currentRun,
        'tag': tag,
      },
      queryParamsHandling: 'merge',
    });
  }

  get diagnostics(): Diagnostics {
    return {
      errors: this.errorMessages,
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
