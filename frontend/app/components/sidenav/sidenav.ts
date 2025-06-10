import {Component, OnDestroy, OnInit} from '@angular/core';
import {ActivatedRouteSnapshot, Router} from '@angular/router';
import {Store} from '@ngrx/store';
import {DEFAULT_HOST, HLO_TOOLS} from 'org_xprof/frontend/app/common/constants/constants';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {RunToolsMap} from 'org_xprof/frontend/app/common/interfaces/tool';
import {CommunicationService} from 'org_xprof/frontend/app/services/communication_service/communication_service';
import {DataServiceV2} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2';
import {setCurrentRunAction, updateRunToolsMapAction} from 'org_xprof/frontend/app/store/actions';
import {getCurrentRun, getRunToolsMap} from 'org_xprof/frontend/app/store/selectors';
import {firstValueFrom, Observable, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A side navigation component. */
@Component({
  standalone: false,
  selector: 'sidenav',
  templateUrl: './sidenav.ng.html',
  styleUrls: ['./sidenav.scss']
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
  selectedModuleInternal = '';
  navigationParams: {[key: string]: string|boolean} = {};

  constructor(
      private readonly router: Router,
      // Using DataServiceV2 because methods used in sidenav is not defined in
      // the interface. (b/423713470)
      private readonly dataService: DataServiceV2,
      private readonly communicationService: CommunicationService,
      private readonly store: Store<{}>) {
    this.runToolsMap$ =
        this.store.select(getRunToolsMap).pipe(takeUntil(this.destroyed));
    this.currentRun$ =
        this.store.select(getCurrentRun).pipe(takeUntil(this.destroyed));
    // TODO(b/241842487): stream is not updated when the state change, should
    // trigger subscribe reactively
    this.runToolsMap$.subscribe((runTools: RunToolsMap) => {
      this.runToolsMap = runTools;
      this.runs = Object.keys(this.runToolsMap);
    });
    this.currentRun$.subscribe(run => {
      if (run && !this.selectedRunInternal) {
        this.selectedRunInternal = run;
      }
    });
  }

  get is_hlo_tool() {
    return HLO_TOOLS.includes(this.selectedTag);
  }

  // Getter for valid run given url router or user selection.
  get selectedRun() {
    return this.runs.find(validRun => validRun === this.selectedRunInternal) ||
        this.runs[0] || '';
  }

  // Getter for valid tag given url router or user selection.
  get selectedTag() {
    return this.tags.find(
               validTag => validTag.startsWith(this.selectedTagInternal)) ||
        this.tags[0] || '';
  }

  // Getter for valid host given url router or user selection.
  get selectedHost() {
    return this.hosts.find(host => host === this.selectedHostInternal) ||
        this.hosts[0] || '';
  }

  get selectedModule() {
    return this.moduleList.find(
               module => module === this.selectedModuleInternal) ||
        this.moduleList[0] || '';
  }

  // https://github.com/angular/angular/issues/11023#issuecomment-752228784
  mergeRouteParams(): Map<string, string> {
    const params = new Map<string, string>();
    const stack: ActivatedRouteSnapshot[] =
        [this.router.routerState.snapshot.root];
    while (stack.length > 0) {
      const route = stack.pop();
      if (!route) continue;
      for (const key in route.params) {
        if (route.params.hasOwnProperty(key)) {
          params.set(key, route.params[key]);
        }
      }
      stack.push(...route.children);
    }

    return params;
  }

  navigateWithUrl() {
    let params: Map<string, string>|URLSearchParams;
    if (!!window.parent.location.search) {
      params = new URLSearchParams(window.parent.location.search);
    } else {
      params = this.mergeRouteParams();
    }
    const run = params.get('run') || '';
    const tag = params.get('tool') || params.get('tag') || '';
    const host = params.get('host') || '';
    const opName = params.get('node_name') || params.get('opName') || '';
    const moduleName = params.get('module_name') || '';
    this.navigationParams['firstLoad'] = true;
    if (opName) {
      this.navigationParams['opName'] = opName;
    }
    if (this.selectedRunInternal === run && this.selectedTagInternal === tag &&
        this.selectedHostInternal === host) {
      return;
    }
    this.selectedRunInternal = run;
    this.selectedTagInternal = tag;
    this.selectedHostInternal = host;
    this.selectedModuleInternal = moduleName;
    this.update();
  }

  ngOnInit() {
    this.navigateWithUrl();
  }

  getNavigationEvent(): NavigationEvent {
    const navigationEvent: NavigationEvent = {
      run: this.selectedRun,
      tag: this.selectedTag,
      host: this.selectedHost,
      ...this.navigationParams,
    };
    if (this.is_hlo_tool) {
      navigationEvent.moduleName = this.selectedModule;
    }
    return navigationEvent;
  }

  getDisplayTagName(tag: string): string {
    const tagName = (tag && tag.length && (tag[tag.length - 1] === '@')) ?
        tag.slice(0, -1) :
        tag || '';

    const toolsDisplayMap = new Map([
      ['overview_page', 'Overview Page'],
      ['framework_op_stats', 'Framework Op Stats'],
      ['memory_profile', 'Memory Profile'], ['pod_viewer', 'Pod Viewer'],
      ['op_profile', 'HLO Op Profile'], ['memory_viewer', 'Memory Viewer'],
      ['graph_viewer', 'Graph Viewer'], ['hlo_stats', 'HLO Op Stats'],
      ['inference_profile', 'Inference Profile'],
      ['roofline_model', 'Roofline Model'], ['kernel_stats', 'Kernel Stats'],
      ['trace_viewer', 'Trace Viewer'],
      ['megascale_stats', 'Megascale Stats']
    ]);
    return toolsDisplayMap.get(tagName) || tagName;
  }

  async getToolsForSelectedRun() {
    const tools =
        await firstValueFrom(this.dataService.getRunTools(this.selectedRun)
                                 .pipe(takeUntil(this.destroyed)));

    this.store.dispatch(updateRunToolsMapAction({
      run: this.selectedRun,
      tools,
    }));
    return tools;
  }

  async getHostsForSelectedTag() {
    if (!this.selectedRun || !this.selectedTag) return [];
    const response = await firstValueFrom(
        this.dataService.getHosts(this.selectedRun, this.selectedTag)
            .pipe(takeUntil(this.destroyed)));

    let hosts = response.map(host => host.hostname) || [];
    if (hosts.length === 0) {
      hosts.push('');
    }
    hosts = hosts.map(host => {
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
    const response =
        await firstValueFrom(this.dataService.getModuleList(this.selectedRun)
                                 .pipe(takeUntil(this.destroyed)));
    return response.split(',');
  }

  onRunSelectionChange(run: string) {
    this.selectedRunInternal = run;
    this.afterUpdateRun();
  }

  afterUpdateRun() {
    this.store.dispatch(setCurrentRunAction({
      currentRun: this.selectedRun,
    }));
    this.updateTags();
  }

  async updateTags() {
    this.tags = this.runToolsMap[this.selectedRun] || [];
    if (!this.tags.length) {
      this.tags = (await this.getToolsForSelectedRun() || []) as string[];
    }
    this.afterUpdateTag();
  }

  onTagSelectionChange(tag: string) {
    this.selectedTagInternal = tag;
    this.afterUpdateTag();
  }

  afterUpdateTag() {
    this.updateHosts();
  }

  // Hosts and ModuleLit used to share the same variable.
  // Keep them under the same update function as initial step of the separation.
  async updateHosts() {
    this.hosts = await this.getHostsForSelectedTag();
    if (this.is_hlo_tool) {
      this.moduleList = await this.getModuleListForSelectedTag();
    }

    this.afterUpdateHost();
  }

  onHostSelectionChange(host: string) {
    this.selectedHostInternal = host;
    this.navigateTools();
  }

  onModuleSelectionChange(module: string) {
    this.selectedModuleInternal = module;
    this.navigateTools();
  }

  afterUpdateHost() {
    this.navigateTools();
  }

  updateUrlHistory() {
    // TODO(xprof): change to camel case when constructing url
    const toolQueryParams = Object.keys(this.navigationParams)
                                .map(key => {
                                  return `${key}=${this.navigationParams[key]}`;
                                })
                                .join('&');
    const toolQueryParamsString =
        toolQueryParams.length ? `&${toolQueryParams}` : '';
    const moduleNameQuery =
        this.is_hlo_tool ? `&module_name=${this.selectedModule}` : '';
    const url = `${window.parent.location.origin}?tool=${
        this.selectedTag}&host=${this.selectedHost}&run=${this.selectedRun}${
        toolQueryParamsString}${moduleNameQuery}#profile`;
    window.parent.history.pushState({}, '', url);
  }

  navigateTools() {
    const navigationEvent = this.getNavigationEvent();
    this.communicationService.onNavigateReady(navigationEvent);
    this.router.navigate(
        [
          this.selectedTag || 'empty',
          navigationEvent,
        ],
        // TODO - b/401596855: Clean up query processing in tools component with
        // addition of the query params in navigation.
        {
          queryParams: navigationEvent,
        });
    delete this.navigationParams['firstLoad'];
    this.updateUrlHistory();
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
