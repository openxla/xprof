import {
  ChangeDetectionStrategy,
  Component,
  HostListener,
  inject,
  OnDestroy,
  ViewChild,
} from '@angular/core';
import {MatSidenav} from '@angular/material/sidenav';
import {Router} from '@angular/router';
import {Store} from '@ngrx/store';
import {
  CANONICAL_CATEGORY_ORDER,
  NON_UI_TOOLS,
  TOOL_CATEGORIES,
  ToolCategory,
} from 'org_xprof/frontend/app/common/constants/constants';
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

/** Representation of a categorized tool group for the rack navigation. */
export interface CategorizedTools {
  category: ToolCategory;
  tools: string[];
}

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

  @ViewChild(MatSidenav) sidenav?: MatSidenav;
  loading = true;
  loadingMessage = '';
  isSideNavOpen = true;
  isSmallScreen = window.innerWidth < 960;
  navigationReady = false;
  errorMessages: string[] = [];
  /** The version string of the XProf plugin. */
  pluginVersion = '';

  isNewNavEnabled = false;
  currentRun = '';
  currentTag = '';
  runToolsMap: RunToolsMap = {};

  collapsedCategories = new Set<string>();
  pinnedTools = new Set<string>();

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
    this.loadNavPreferences();
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
        if (!errorMessage) {
          this.errorMessages = [];
          return;
        }
        if (this.errorMessages.includes(errorMessage)) {
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
        this.isSmallScreen = window.innerWidth < 960;
        this.currentRun = navigationEvent.run || this.currentRun;
        this.currentTag = navigationEvent.tag || '';
        if (!this.isNewNavEnabled) {
          const hasSideNavControls = this.isToolWithSideNav(
            navigationEvent?.tag,
          );
          if (
            this.isSmallScreen &&
            navigationEvent?.tag &&
            navigationEvent.tag !== 'empty' &&
            !hasSideNavControls
          ) {
            this.isSideNavOpen = false;
            this.sidenav?.close();
          } else {
            this.isSideNavOpen =
              Boolean(navigationEvent.firstLoad) || hasSideNavControls;
          }
        }
      });
    this.dataService
      .getPluginVersion()
      .pipe(takeUntil(this.destroyed))
      .subscribe((version: string | null) => {
        this.pluginVersion = version || '';
      });
  }

  get activeTool(): string {
    return this.currentTag || 'overview_page';
  }

  get availableTools(): string[] {
    return this.runToolsMap[this.currentRun] || [];
  }

  getCategorizedTools(): CategorizedTools[] {
    const activeTools = this.availableTools.filter((tag) => {
      const cleanTag = tag && tag.endsWith('@') ? tag.slice(0, -1) : tag || '';
      return tag !== 'empty' && !NON_UI_TOOLS.has(cleanTag);
    });
    const groupsMap = new Map<ToolCategory, string[]>();
    for (const category of CANONICAL_CATEGORY_ORDER) {
      groupsMap.set(category, []);
    }

    for (const tag of activeTools) {
      const cleanTag = tag && tag.endsWith('@') ? tag.slice(0, -1) : tag || '';
      const category =
        CANONICAL_CATEGORY_ORDER.find((cat) =>
          TOOL_CATEGORIES[cat].includes(cleanTag),
        ) || ToolCategory.OTHER;
      groupsMap.get(category)!.push(tag);
    }

    const result: CategorizedTools[] = [];
    for (const category of CANONICAL_CATEGORY_ORDER) {
      const tools = groupsMap.get(category) || [];
      if (tools.length > 0) {
        const canonicalOrder = TOOL_CATEGORIES[category];
        tools.sort((a, b) => {
          const cleanA = a && a.endsWith('@') ? a.slice(0, -1) : a || '';
          const cleanB = b && b.endsWith('@') ? b.slice(0, -1) : b || '';
          return (
            canonicalOrder.indexOf(cleanA) - canonicalOrder.indexOf(cleanB)
          );
        });
        result.push({category, tools});
      }
    }
    return result;
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
    const queryParams: {[key: string]: string} = {
      'run': this.currentRun,
      'tag': tag,
    };
    if (this.isNewNavEnabled) {
      queryParams['new_nav'] = 'true';
    }
    this.router.navigate([tag || 'empty'], {queryParams});
  }

  private isToolWithSideNav(tag?: string): boolean {
    // TODO(fe-unification): Remove this constraint once the sidepanel
    // content of the tools are moved out from sidenav with consolidated
    // templates.
    const toolsWithSideNav = [
      'op_profile',
      'memory_viewer',
      'pod_viewer',
      'megascale_stats',
    ];
    return toolsWithSideNav.some((tool) => tag?.startsWith(tool));
  }

  @HostListener('window:resize')
  onResize() {
    const wasSmallScreen = this.isSmallScreen;
    this.isSmallScreen = window.innerWidth < 960;
    // Close the overlay side panel only when the window shrinks below the
    // breakpoint, so resizing an already narrow window keeps a panel that the
    // user opened.
    if (
      !wasSmallScreen &&
      this.isSmallScreen &&
      !this.isNewNavEnabled &&
      this.navigationReady &&
      this.currentTag &&
      this.currentTag !== 'empty' &&
      !this.isToolWithSideNav(this.currentTag)
    ) {
      this.isSideNavOpen = false;
      this.sidenav?.close();
    }
  }

  get sideNavMode(): 'over' | 'side' {
    return !this.isNewNavEnabled && this.isSmallScreen ? 'over' : 'side';
  }

  get sideNavWidthPx(): number | null {
    if (!this.isNewNavEnabled) {
      return null;
    }
    return this.isSideNavOpen ? 240 : 64;
  }

  get contentMarginLeftPx(): number {
    if (this.isNewNavEnabled) {
      return this.isSideNavOpen ? 240 : 64;
    }
    return this.sideNavMode === 'side' && this.isSideNavOpen ? 340 : 0;
  }

  onSideNavOpenedChange(opened: boolean) {
    if (!this.isNewNavEnabled) {
      if (!opened) {
        this.isSideNavOpen = false;
      } else if (!this.isSideNavOpen) {
        this.sidenav?.close();
      }
    }
  }

  isCategoryExpanded(category: string): boolean {
    return !this.collapsedCategories.has(category);
  }

  onCategoryExpandedChange(category: string, expanded: boolean) {
    if (expanded) {
      this.collapsedCategories.delete(category);
    } else {
      this.collapsedCategories.add(category);
    }
    this.saveNavPreferences();
  }

  isToolPinned(tag: string): boolean {
    const cleanTag = tag && tag.endsWith('@') ? tag.slice(0, -1) : tag || '';
    return this.pinnedTools.has(cleanTag);
  }

  onToolPinnedChange(tag: string, pinned: boolean) {
    const cleanTag = tag && tag.endsWith('@') ? tag.slice(0, -1) : tag || '';
    if (pinned) {
      this.pinnedTools.add(cleanTag);
    } else {
      this.pinnedTools.delete(cleanTag);
    }
    this.saveNavPreferences();
  }

  private loadNavPreferences() {
    try {
      const collapsedJson = window.localStorage.getItem(
        'xprof_collapsed_categories',
      );
      if (collapsedJson) {
        this.collapsedCategories = new Set(
          (JSON.parse(collapsedJson) as string[]) || [],
        );
      }
      const pinnedJson = window.localStorage.getItem('xprof_pinned_tools');
      if (pinnedJson) {
        this.pinnedTools = new Set((JSON.parse(pinnedJson) as string[]) || []);
      }
    } catch {
      // Ignore JSON parse errors.
    }
  }

  private saveNavPreferences() {
    try {
      window.localStorage.setItem(
        'xprof_collapsed_categories',
        JSON.stringify([...this.collapsedCategories]),
      );
      window.localStorage.setItem(
        'xprof_pinned_tools',
        JSON.stringify([...this.pinnedTools]),
      );
    } catch {
      // Ignore quota/security errors.
    }
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
