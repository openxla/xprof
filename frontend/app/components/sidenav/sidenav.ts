import {
  ChangeDetectionStrategy,
  Component,
  OnDestroy,
  OnInit,
} from '@angular/core';
import {MatCheckboxChange} from '@angular/material/checkbox';
import {ActivatedRouteSnapshot, NavigationEnd, Router} from '@angular/router';
import {Store} from '@ngrx/store';
import {
  DEFAULT_HOST,
  HLO_TOOLS,
} from 'org_xprof/frontend/app/common/constants/constants';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {RunToolsMap} from 'org_xprof/frontend/app/common/interfaces/tool';
import {CommunicationService} from 'org_xprof/frontend/app/services/communication_service/communication_service';
import {DataServiceV2} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2';
import {
  setCurrentRunAction,
  setProfilerConfigAction,
  updateRunToolsMapAction,
} from 'org_xprof/frontend/app/store/actions';
import {
  getCurrentRun,
  getRunToolsMap,
} from 'org_xprof/frontend/app/store/selectors';
import {firstValueFrom, Observable, ReplaySubject} from 'rxjs';
import {defaultIfEmpty, filter, takeUntil} from 'rxjs/operators';

/** Extracts query parameters from window.parent location search. */
export function getParentLocationParams(): Map<string, string> {
  const parentParams = new Map<string, string>();
  try {
    if (window.parent?.location?.search) {
      const urlParams = new URLSearchParams(window.parent.location.search);
      for (const [key, value] of urlParams) {
        parentParams.set(key, value);
      }
      const parentHosts = urlParams.getAll('hosts');
      if (parentHosts.length > 0) {
        parentParams.set('hosts', parentHosts.join(','));
      }
    }
  } catch {
    // In case window.parent access is blocked by cross-origin iframe security
  }
  return parentParams;
}

/** Interface for normalized navigation parameters. */
export interface NavigationParamsRecord {
  run: string;
  tag: string;
  host: string;
  hosts: string;
  sessionPath: string;
  runPath: string;
  baseSessionId: string;
  label: string;
  opName: string;
  moduleName: string;
}

/** Normalizes navigation parameters and aliases. */
export function parseNavigationParams(
  params: ReadonlyMap<string, string>,
): NavigationParamsRecord {
  return {
    run: params.get('run') ?? params.get('sessionId') ?? '',
    tag: params.get('tag') ?? params.get('tool') ?? '',
    host: params.get('host') ?? '',
    hosts: params.get('hosts') ?? '',
    sessionPath: params.get('session_path') ?? params.get('sessionPath') ?? '',
    runPath: params.get('run_path') ?? params.get('runPath') ?? '',
    baseSessionId:
      params.get('base_session_id') ?? params.get('baseSessionId') ?? '',
    label: params.get('label') ?? '',
    opName: params.get('opName') ?? params.get('node_name') ?? '',
    moduleName: params.get('moduleName') ?? params.get('module_name') ?? '',
  };
}

/** Serializes query parameters into a URL query string. */
export function serializeQueryParams(params: {
  [key: string]: string | string[] | boolean | undefined;
}): string {
  const searchParams = new URLSearchParams();
  for (const key of Object.keys(params)) {
    const value = params[key];
    if (value !== undefined && value !== null) {
      if (Array.isArray(value)) {
        searchParams.set(key, value.join(','));
      } else if (typeof value === 'boolean') {
        if (value) {
          searchParams.set(key, 'true');
        }
      } else {
        searchParams.set(key, String(value));
      }
    }
  }
  const queryString = searchParams.toString();
  return queryString ? `?${queryString}` : '';
}

/** A side navigation component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'sidenav',
  templateUrl: './sidenav.ng.html',
  styleUrls: ['./sidenav.scss'],
})
export class SideNav implements OnInit, OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  runToolsMap$: Observable<RunToolsMap>;
  currentRun$: Observable<string>;

  runToolsMap: RunToolsMap = {};
  runs: string[] = [];
  tags: string[] = [];
  hosts: string[] = [];
  moduleList: string[] = [];
  selectedRunInternal = '';
  selectedTagInternal = '';
  selectedHostInternal = '';
  runPathInternal = '';
  sessionPathInternal = '';
  labelInternal = '';
  selectedHostsInternal: string[] = [];
  selectedHostsPending: string[] = [];
  selectedModuleInternal = '';
  navigationParams: {[key: string]: string | boolean} = {};
  multiHostEnabledTools: string[] = ['trace_viewer', 'trace_viewer@'];
  allHostsSelected = false;

  hideCaptureProfileButton = false;
  enableTabNameLabel = false;

  constructor(
    private readonly router: Router,
    // Using DataServiceV2 because methods used in sidenav is not defined in
    // the interface. (b/423713470)
    private readonly dataService: DataServiceV2,
    private readonly communicationService: CommunicationService,
    private readonly store: Store<{}>,
  ) {
    this.runToolsMap$ = this.store
      .select(getRunToolsMap)
      .pipe(takeUntil(this.destroyed));
    this.currentRun$ = this.store
      .select(getCurrentRun)
      .pipe(takeUntil(this.destroyed));
    // TODO(b/241842487): stream is not updated when the state change, should
    // trigger subscribe reactively
    this.runToolsMap$.subscribe((runTools: RunToolsMap) => {
      this.runToolsMap = runTools;
      this.runs = Object.keys(this.runToolsMap);
    });
    this.currentRun$.subscribe((run) => {
      if (run && !this.selectedRunInternal) {
        this.selectedRunInternal = run;
      }
    });
  }

  get is_hlo_tool() {
    return HLO_TOOLS.includes(this.selectedTag);
  }

  get isMultiHostsEnabled() {
    const tag = this.selectedTag || '';
    return this.multiHostEnabledTools.includes(tag);
  }

  // Getter for valid run given url router or user selection.
  get selectedRun() {
    return (
      this.runs.find((validRun) => validRun === this.selectedRunInternal) ||
      this.runs[0] ||
      ''
    );
  }

  // Getter for valid tag given url router or user selection.
  get selectedTag() {
    return (
      this.tags.find((validTag) =>
        validTag.startsWith(this.selectedTagInternal),
      ) ||
      this.tags[0] ||
      ''
    );
  }

  // Getter for valid host given url router or user selection.
  get selectedHost() {
    return (
      this.hosts.find((host) => host === this.selectedHostInternal) ||
      this.hosts[0] ||
      ''
    );
  }

  get selectedModule(): string {
    return this.selectedModuleInternal || this.moduleList[0] || '';
  }

  get selectedHosts() {
    return this.selectedHostsInternal;
  }

  // https://github.com/angular/angular/issues/11023#issuecomment-752228784
  mergeRouteParams(): Map<string, string> {
    const params = new Map<string, string>();
    const rootSnapshot = this.router.routerState?.snapshot?.root;
    const stack: ActivatedRouteSnapshot[] = rootSnapshot ? [rootSnapshot] : [];
    while (stack.length > 0) {
      const route = stack.pop();
      if (!route) continue;
      if (route.queryParams) {
        for (const key of Object.keys(route.queryParams)) {
          params.set(key, route.queryParams[key]);
        }
      }
      if (route.params) {
        for (const key of Object.keys(route.params)) {
          params.set(key, route.params[key]);
        }
      }
      stack.push(...route.children);
    }

    return params;
  }

  navigateWithUrl() {
    const parentParams = getParentLocationParams();
    const routeParams = this.mergeRouteParams();

    const parent = parseNavigationParams(parentParams);
    const route = parseNavigationParams(routeParams);

    const run = route.run || parent.run;
    const tag = route.tag || parent.tag;
    const host = route.host || parent.host;
    const hostsParam = route.hosts || parent.hosts;
    const sessionPath = route.sessionPath || parent.sessionPath;
    const runPath = route.runPath || parent.runPath;
    const baseSessionId = route.baseSessionId || parent.baseSessionId;
    const label = route.label || parent.label;
    const opName = route.opName || parent.opName;
    const moduleName = route.moduleName || parent.moduleName;

    const isHostsEqual = this.isMultiHostsEnabled
      ? this.selectedHostsInternal.join(',') === hostsParam
      : this.selectedHostInternal === host;

    // Guard to prevent infinite navigation loops on identical route params.
    // In Angular 18, navigating to the same route triggers NavigationEnd because
    // getNavigationEvent() returns a new object reference on every call, which
    // is treated as a parameter change by reference comparison.
    //
    // TODO: b/536901902 - Refactor SideNav routing to follow a reactive, one-way
    // data flow (URL as single source of truth) where UI dropdowns only trigger
    // navigations, and an ActivatedRoute queryParams subscription handles all
    // state syncs and data fetches. See xprof_angular_routing_refactor_proposal.md.
    if (
      this.selectedRunInternal === run &&
      this.selectedTagInternal === tag &&
      this.selectedModuleInternal === moduleName &&
      this.runPathInternal === runPath &&
      this.sessionPathInternal === sessionPath &&
      this.labelInternal === label &&
      (this.navigationParams['opName'] || '') === opName &&
      (this.dataService.getBaseSessionId() ?? '') === baseSessionId &&
      isHostsEqual
    ) {
      return;
    }
    this.navigationParams['firstLoad'] = true;
    if (opName) {
      this.navigationParams['opName'] = opName;
    } else {
      delete this.navigationParams['opName'];
    }
    this.selectedRunInternal = run;
    this.selectedTagInternal = tag;
    this.selectedModuleInternal = moduleName;
    this.runPathInternal = runPath;
    this.sessionPathInternal = sessionPath;
    this.labelInternal = label;
    this.dataService.setBaseSessionId(baseSessionId);

    if (this.multiHostEnabledTools.includes(tag)) {
      if (hostsParam) {
        let hostsList = hostsParam.split(',');
        if (hostsList.length > 10) {
          hostsList = hostsList.slice(0, 10);
        }
        this.selectedHostsInternal = hostsList;
      }
      this.selectedHostsPending = [...this.selectedHostsInternal];
      this.updateAllHostsSelectedState();
    } else {
      this.selectedHostInternal = host;
    }
    this.update();
  }

  ngOnInit() {
    this.navigateWithUrl();
    this.fetchProfilerConfig();
    this.router.events
      ?.pipe(
        filter((event) => event instanceof NavigationEnd),
        takeUntil(this.destroyed),
      )
      .subscribe((event) => {
        this.navigateWithUrl();
      });
  }

  async fetchProfilerConfig() {
    const config = await firstValueFrom(
      this.dataService.getConfig().pipe(takeUntil(this.destroyed), defaultIfEmpty(null)),
    );
    if (config) {
      this.store.dispatch(setProfilerConfigAction({config}));
      this.hideCaptureProfileButton = config.hideCaptureProfileButton;
      this.enableTabNameLabel = config.enableTabNameLabel ?? false;
    }
  }

  getNavigationEvent(): NavigationEvent {
    const navigationEvent: NavigationEvent = {
      run: this.selectedRun,
      tag: this.selectedTag,
      ...this.navigationParams,
    };
    if (this.isMultiHostsEnabled) {
      navigationEvent.hosts = this.selectedHostsInternal;
    } else {
      navigationEvent.host = this.selectedHost;
    }
    if (this.is_hlo_tool) {
      if (this.moduleList.length > 0 && !this.selectedModuleInternal) {
        this.selectedModuleInternal = this.moduleList[0];
      }
      navigationEvent.moduleName = this.selectedModule;
    }
    if (this.runPathInternal) {
      navigationEvent.run_path = this.runPathInternal;
    }
    if (this.sessionPathInternal) {
      navigationEvent.session_path = this.sessionPathInternal;
    }
    const baseSessionId = this.dataService.getBaseSessionId();
    if (baseSessionId) {
      navigationEvent.base_session_id = baseSessionId;
    }
    return navigationEvent;
  }

  getDisplayTagName(tag: string): string {
    const tagName =
      tag && tag.length && tag[tag.length - 1] === '@'
        ? tag.slice(0, -1)
        : tag || '';

    const toolsDisplayMap = new Map([
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
    return toolsDisplayMap.get(tagName) || tagName;
  }

  async getToolsForSelectedRun() {
    const tools = await firstValueFrom(
      this.dataService
        .getRunTools(this.selectedRun)
        .pipe(takeUntil(this.destroyed), defaultIfEmpty([])),
    );

    this.store.dispatch(
      updateRunToolsMapAction({
        run: this.selectedRun,
        tools,
      }),
    );
    return tools;
  }

  async getHostsForSelectedTag() {
    if (!this.selectedRun || !this.selectedTag) return [];
    const response = await firstValueFrom(
      this.dataService
        .getHosts(this.selectedRun, this.selectedTag)
        .pipe(takeUntil(this.destroyed), defaultIfEmpty([])),
    );

    let hosts = response.map((host) => host.hostname) || [];
    if (hosts.length === 0) {
      hosts.push('');
    }
    hosts = hosts.map((host) => {
      if (host === null) {
        return '';
      } else if (host === '') {
        return DEFAULT_HOST;
      }
      return host;
    });
    return hosts;
  }

  async getModuleListForSelectedTag() {
    if (!this.selectedRun || !this.selectedTag) return [];
    const response = await firstValueFrom(
      this.dataService
        .getModuleList(this.selectedRun)
        .pipe(takeUntil(this.destroyed), defaultIfEmpty('')),
    );
    return response ? response.split(',') : [];
  }

  onRunSelectionChange(run: string) {
    this.selectedRunInternal = run;
    this.afterUpdateRun();
  }

  afterUpdateRun() {
    this.store.dispatch(
      setCurrentRunAction({
        currentRun: this.selectedRun,
      }),
    );
    this.updateTags();
  }

  async updateTags() {
    this.tags = this.runToolsMap[this.selectedRun] || [];
    if (!this.tags.length) {
      this.tags = ((await this.getToolsForSelectedRun()) || []) as string[];
    }
    this.afterUpdateTag();
  }

  async onTagSelectionChange(tag: string) {
    const previousSelectedTag = this.selectedTagInternal;
    this.selectedTagInternal = tag;

    const isChangingToMultiHost =
      !previousSelectedTag ||
      (this.isMultiHostsEnabled && previousSelectedTag !== this.selectedTag);

    // Reset module and op selection when tool changes
    if (previousSelectedTag !== tag) {
      this.selectedModuleInternal = '';
      delete this.navigationParams['opName'];
      delete this.navigationParams['moduleName'];
    }

    this.selectedHostsInternal = [];
    this.selectedHostsPending = [];
    this.selectedHostInternal = '';

    if (isChangingToMultiHost) {
      this.hosts = await this.getHostsForSelectedTag();
      this.selectedHostsInternal = this.hosts.length > 0 ? [this.hosts[0]] : [];
      this.selectedHostsPending = [...this.selectedHostsInternal];
      this.updateAllHostsSelectedState();

      await this.syncHloModuleList();

      this.navigateTools();
    } else {
      this.afterUpdateTag();
    }
  }

  afterUpdateTag() {
    this.updateHosts();
  }

  /**
   * Synchronizes the HLO module list for the selected tag and ensures a valid
   * module is selected.
   */
  private async syncHloModuleList(): Promise<void> {
    if (this.is_hlo_tool) {
      this.moduleList = await this.getModuleListForSelectedTag();
      if (
        this.moduleList.length > 0 &&
        (!this.selectedModuleInternal ||
          !this.moduleList.includes(this.selectedModuleInternal))
      ) {
        this.selectedModuleInternal = this.moduleList[0];
      }
    }
  }

  // Hosts and ModuleLit used to share the same variable.
  // Keep them under the same update function as initial step of the separation.
  async updateHosts() {
    this.hosts = await this.getHostsForSelectedTag();

    if (this.isMultiHostsEnabled) {
      this.selectedHostsInternal = this.selectedHostsInternal.filter((host) =>
        this.hosts.includes(host),
      );
      if (this.selectedHostsInternal.length === 0 && this.hosts.length > 0) {
        this.selectedHostsInternal = [this.hosts[0]];
      }
      this.selectedHostsPending = [...this.selectedHostsInternal];
      this.updateAllHostsSelectedState();
    } else {
      if (
        !this.hosts.includes(this.selectedHostInternal) &&
        this.hosts.length > 0
      ) {
        this.selectedHostInternal = this.hosts[0];
      }
    }
    await this.syncHloModuleList();

    this.afterUpdateHost();
  }

  onHostSelectionChange(selection: string) {
    this.selectedHostInternal = selection;
    this.navigateTools();
  }

  onHostsSelectionChange(selection: string[]) {
    this.selectedHostsPending = Array.isArray(selection)
      ? selection
      : [selection];
    this.updateAllHostsSelectedState();
  }

  onToggleSelectAllHosts(event: MatCheckboxChange) {
    this.selectedHostsPending = event.checked ? [...this.hosts] : [];
    this.updateAllHostsSelectedState();
  }

  private updateAllHostsSelectedState() {
    this.allHostsSelected =
      this.hosts.length > 0 &&
      this.selectedHostsPending.length === this.hosts.length;
  }

  onSubmitHosts() {
    if (this.selectedHostsPending.length > 10) {
      alert(
        'Warning: Maximum 10 selected hosts allowed. Loading the first 10 hosts in alphabetical order.',
      );
      this.selectedHostsPending = this.selectedHostsPending.slice(0, 10);
      this.updateAllHostsSelectedState();
    }
    this.selectedHostsInternal = [...this.selectedHostsPending];
    this.navigateTools();
  }

  onModuleSelectionChange(module: string) {
    this.selectedModuleInternal = module;
    this.navigateTools();
  }

  afterUpdateHost() {
    this.navigateTools();
  }

  updateUrlHistory(): void {
    try {
      const navigationEvent = this.getNavigationEvent();
      const queryParams: {
        [key: string]: string | string[] | boolean | undefined;
      } = {...navigationEvent};

      if (this.isMultiHostsEnabled) {
        const hosts = queryParams['hosts'];
        if (Array.isArray(hosts)) {
          queryParams['hosts'] = hosts.join(',');
        }
        delete queryParams['host']; // Remove single host param
      } else {
        // For other tools, ensure 'host' is used
        delete queryParams['hosts']; // Remove multi-host param
      }

      // Get current path to avoid changing the base URL
      const pathname = window.parent?.location?.pathname ?? '';

      // Use the custom serialization helper
      const queryString = serializeQueryParams(queryParams);
      const url = pathname + queryString;

      window.parent?.history?.pushState({}, '', url);
    } catch (error) {
      console.error('Failed to update URL history:', error);
    }
  }

  navigateTools() {
    const navigationEvent = this.getNavigationEvent();
    this.communicationService.onNavigateReady(navigationEvent);

    this.updateUrlHistory();
    // This router.navigate call remains, as it's responsible for Angular
    // routing
    // TODO - b/401596855: Deprecate the navigationEvent in route.params as we
    // are subscribing to the queryParams in the components.
    this.router.navigate([this.selectedTag || 'empty'], {
      queryParams: navigationEvent,
    });
    delete this.navigationParams['firstLoad'];
    this.updateTitle();
  }

  updateTitle() {
    if (!this.enableTabNameLabel) {
      return;
    }
    const toolName = this.getDisplayTagName(this.selectedTag);
    let sessionName = '';

    if (this.sessionPathInternal) {
      const parts = this.sessionPathInternal
        .split('/')
        .filter((p) => p.length > 0);
      const pluginsIndex = parts.indexOf('plugins');
      if (pluginsIndex > 0) {
        sessionName = parts[pluginsIndex - 1];
      } else {
        sessionName = parts[parts.length - 1] || '';
      }
    }

    if (!sessionName) {
      sessionName = this.selectedRun;
    }

    let titleLabel = sessionName;
    if (this.labelInternal) {
      titleLabel += ` (${this.labelInternal})`;
    }

    if (titleLabel) {
      document.title = `${titleLabel} | ${toolName} - XProf`;
    } else {
      document.title = `${toolName} - XProf`;
    }
  }

  update() {
    this.afterUpdateRun();
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
