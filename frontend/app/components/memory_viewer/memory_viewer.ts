import {
  ChangeDetectionStrategy,
  Component,
  inject,
  OnDestroy,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {MemoryViewerPreprocessResult} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {
  createSessionIdResolver,
  getSessionIdFromParams,
  resolveHostFromParams,
  setLoadingState,
  syncParentUrlParams,
} from 'org_xprof/frontend/app/common/utils/utils';
import {CommunicationService} from 'org_xprof/frontend/app/services/communication_service/communication_service';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, ReplaySubject, Subscription} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A memory viewer component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'memory-viewer',
  templateUrl: './memory_viewer.ng.html',
  styleUrls: ['./memory_viewer.scss'],
})
export class MemoryViewer implements OnDestroy {
  tool = 'memory_viewer';
  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );
  private readonly route = inject(ActivatedRoute);
  private readonly store = inject(Store<{}>);
  private readonly communicationService = inject(CommunicationService);
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private moduleListSubscription?: Subscription;
  private moduleDataSubscription?: Subscription;
  private initialLoadPending = false;
  sessionId = '';
  private loadedSessionId = '';
  host = '';
  loading = false;
  private readonly throbber = new Throbber(this.tool);
  memoryViewerPreprocessResult: MemoryViewerPreprocessResult | null = null;
  moduleList: string[] = [];
  selectedModule = '';
  firstLoadModuleIndex = 0;
  firstLoadMemorySpaceColor = '0';
  /*
   * The number associated with the selected memory space.
   * Is set as a string for frontend compatibility.
   * Is passed as a number to the backend via data service.
   */
  selectedMemorySpaceColor = '0';

  constructor() {
    // TODO - b/552140753: Deprecate matrix params in route.params in favor of route.queryParams.
    const resolveSessionId = createSessionIdResolver();
    combineLatest([this.route.params, this.route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        const prevSessionId = this.sessionId;
        this.sessionId = resolveSessionId(params, queryParams, this.sessionId);
        const merged: Params = {...queryParams};
        for (const [key, value] of Object.entries(params)) {
          if (
            !(key in merged) &&
            key !== 'sessionId' &&
            key !== 'run' &&
            key !== 'session_id'
          ) {
            merged[key] = value;
          }
        }
        this.processQuery(merged, prevSessionId);
        this.load();
      });
    this.store.dispatch(
      setCurrentToolStateAction({currentTool: 'memory_viewer'}),
    );
  }

  processQuery(params: Params, prevSessionId = this.sessionId) {
    this.sessionId = getSessionIdFromParams(undefined, params, this.sessionId);
    this.tool = params['tag'] || params['tool'] || this.tool;
    this.host = resolveHostFromParams(
      params,
      prevSessionId,
      this.sessionId,
      this.host,
    );
    this.selectedModule =
      params['moduleName'] || params['module_name'] || this.selectedModule;
    const memorySpace =
      params['memory_space'] ??
      params['memorySpace'] ??
      params['memorySpaceColor'];
    if (memorySpace !== undefined && memorySpace !== '') {
      this.selectedMemorySpaceColor = String(memorySpace);
    } else {
      this.selectedMemorySpaceColor = this.firstLoadMemorySpaceColor;
    }
  }

  /**
   * Resolves selected module and memory space color, falling back to default
   * values if none are currently selected or if the selected module is invalid.
   */
  private resolveSelectedModuleAndMemorySpace() {
    if (
      !this.selectedModule ||
      !this.moduleList.includes(this.selectedModule)
    ) {
      this.selectedModule = this.moduleList[this.firstLoadModuleIndex];
    }
    if (!this.selectedMemorySpaceColor && this.firstLoadMemorySpaceColor) {
      this.selectedMemorySpaceColor = this.firstLoadMemorySpaceColor;
    }
  }

  private finishInitialLoad(force = false) {
    if (force || this.initialLoadPending) {
      this.initialLoadPending = false;
      this.throbber.stop();
      setLoadingState(false, this.store);
    }
  }

  load() {
    // Note that there could be 1-2 api calls depend on the session id
    // the latency measurement will cover all period
    // as measurement and loading stops only if:
    // 1. getModuleList returned empty results, no need for further operation
    // 2. loadModule is done
    this.moduleListSubscription?.unsubscribe();
    this.moduleDataSubscription?.unsubscribe();
    this.initialLoadPending = true;
    setLoadingState(true, this.store, 'Loading memory viewer data');
    this.throbber.start();
    if (this.sessionId !== this.loadedSessionId) {
      this.moduleList = [];
      this.loadedSessionId = this.sessionId;
    }
    // For xsymbol session, There is only 1 module so there is no need to call
    // getModuleList before calling the analysis code.
    if (this.sessionId === 'xsymbol') {
      // Module name is set to empty, the backend server will automatically
      // choose the only 1 module. Memory space color defaults to 0 (HBM).
      this.loadModule(
        '',
        this.selectedMemorySpaceColor || this.firstLoadMemorySpaceColor,
        true,
      );
    } else if (this.moduleList.length > 0) {
      this.resolveSelectedModuleAndMemorySpace();
      this.loadModule(this.selectedModule, this.selectedMemorySpaceColor, true);
    } else {
      this.moduleListSubscription = this.dataService
        .getModuleList(this.sessionId)
        .pipe(takeUntil(this.destroyed))
        .subscribe({
          next: (moduleList: string) => {
            if (moduleList) {
              // `getModuleList` has more than one server-side implementation and
              // not all of them sort, so order the list before defaulting to [0].
              this.moduleList = moduleList.split(',').sort();
              // No need to regenerate modules.
              this.dataService.disableCacheRegeneration();
              this.resolveSelectedModuleAndMemorySpace();
              this.loadModule(
                this.selectedModule,
                this.selectedMemorySpaceColor,
                true,
              );
            } else {
              this.finishInitialLoad();
              this.loading = false;
              this.moduleList = [];
              this.selectedModule = '';
              this.memoryViewerPreprocessResult = null;
            }
          },
          error: () => {
            this.finishInitialLoad();
            this.loading = false;
            this.moduleList = [];
            this.selectedModule = '';
            this.memoryViewerPreprocessResult = null;
          },
        });
    }
  }

  /**
   * Handles changes emitted by the `changed` emitter of the
   * `memory-viewer-control` component.
   *
   * This is invoked whenever one of the controls in `memory-viewer-control` is
   * updated (providing either a `moduleName` or a `memorySpaceColor` change).
   * Synchronizes the updated module to the parent URL history and reloads the
   * module data.
   */
  update(event: NavigationEvent) {
    const moduleChanged =
      Boolean(event.moduleName) && event.moduleName !== this.selectedModule;
    const memorySpaceChanged =
      Boolean(event.memorySpaceColor) &&
      event.memorySpaceColor !== this.selectedMemorySpaceColor;

    if (!moduleChanged && !memorySpaceChanged) {
      return;
    }

    const urlUpdates: {[key: string]: string} = {};
    if (moduleChanged) {
      this.selectedModule = event.moduleName ?? '';
      urlUpdates['moduleName'] = this.selectedModule;
    }

    if (memorySpaceChanged) {
      this.selectedMemorySpaceColor = event.memorySpaceColor ?? '0';
      urlUpdates['memory_space'] = this.selectedMemorySpaceColor;
    }

    if (Object.keys(urlUpdates).length > 0) {
      syncParentUrlParams(urlUpdates);
      this.communicationService.onToolQueryParamsChange(urlUpdates);
    }

    this.loadModule(this.selectedModule, this.selectedMemorySpaceColor || '0');
  }

  loadModule(module: string, memorySpaceColor: string, initialLoad = false) {
    this.moduleDataSubscription?.unsubscribe();
    this.loading = true;
    this.selectedModule = module;
    this.selectedMemorySpaceColor = memorySpaceColor;
    this.moduleDataSubscription = this.dataService
      .getDataByModuleNameAndMemorySpace(
        'memory_viewer',
        this.sessionId,
        this.host,
        module,
        Number(memorySpaceColor),
      )
      .pipe(takeUntil(this.destroyed))
      .subscribe({
        next: (data) => {
          // Page start latency  = initial load of module list + module data
          this.finishInitialLoad(initialLoad);
          this.loading = false;

          this.memoryViewerPreprocessResult =
            (data as MemoryViewerPreprocessResult | null) ?? null;

          // If the caller of loadModule does not provide the module name (like
          // in xsymbol use case), parse and set selectedModule and moduleList
          // using the data from backend.
          if (module === '') {
            if (this.memoryViewerPreprocessResult) {
              this.selectedModule =
                this.memoryViewerPreprocessResult.moduleName || '';
              this.moduleList = [this.selectedModule];
            } else {
              this.selectedModule = '';
              this.moduleList = [];
            }
          }
        },
        error: () => {
          this.finishInitialLoad(initialLoad);
          this.loading = false;
          this.memoryViewerPreprocessResult = null;
        },
      });
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.initialLoadPending = false;
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
