import {Component, inject, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params, Router} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {MemoryViewerPreprocessResult} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {
  setLoadingState,
  syncParentUrlParams,
} from 'org_xprof/frontend/app/common/utils/utils';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A memory viewer component. */
@Component({
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
  private readonly router = inject(Router);
  private readonly store = inject(Store<{}>);
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  sessionId = '';
  private loadedSessionId = '';
  host = '';
  loading = false;
  private readonly throbber = new Throbber(this.tool);
  memoryViewerPreprocessResult: MemoryViewerPreprocessResult | null = null;
  moduleList: string[] = [];
  selectedModule = '';
  private loadedModule = '';
  private loadedMemorySpaceColor = '';
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
    combineLatest([this.route.params, this.route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        const merged = {...params, ...queryParams};
        this.sessionId = merged['sessionId'] || this.sessionId;
        this.processQuery(merged);
        this.load();
      });
    this.store.dispatch(
      setCurrentToolStateAction({currentTool: 'memory_viewer'}),
    );
  }

  processQuery(params: Params): void {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || params['tool'] || this.tool;
    const host = params['host'] || this.host;
    if (host !== this.host) {
      this.host = host;
      this.loadedModule = '';
      this.loadedMemorySpaceColor = '';
    }
    // Canonical snake_case takes precedence over legacy camelCase
    this.selectedModule =
      params['module_name'] || params['moduleName'] || this.selectedModule;
  }

  /**
   * Resolves selected module and memory space color, falling back to default
   * values if none are currently selected or if the selected module is invalid.
   */
  private resolveSelectedModuleAndMemorySpace(): void {
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

  /**
   * Pins selected module to parent window URL history and internal router state
   * if not already present or if the module has changed.
   */
  private pinSelectedModuleToUrl(initialModule: string): void {
    this.resolveSelectedModuleAndMemorySpace();
    if (
      this.selectedModule &&
      (!initialModule || initialModule !== this.selectedModule)
    ) {
      this.syncUrlParams({'module_name': this.selectedModule});
      this.router.navigate([], {
        queryParams: {'module_name': this.selectedModule, 'moduleName': null},
        queryParamsHandling: 'merge',
        replaceUrl: true,
      });
    }
  }

  load(): void {
    // Note that there could be 1-2 api calls depend on the session id
    // the latency measurement will cover all period
    // as measurement and loading stops only if:
    // 1. getModuleList returned empty results, no need for further operation
    // 2. loadModule is done
    setLoadingState(true, this.store, 'Loading memory viewer data');
    this.throbber.start();
    if (this.sessionId !== this.loadedSessionId) {
      this.moduleList = [];
      this.loadedSessionId = this.sessionId;
      this.loadedModule = '';
      this.loadedMemorySpaceColor = '';
    }
    // For xsymbol session, There is only 1 module so there is no need to call
    // getModuleList before calling the analysis code.
    if (this.sessionId === 'xsymbol') {
      // Module name is set to empty, the backend server will automatically
      // choose the only 1 module. Memory space color is set to 0 (HBM) by
      // default.
      this.loadModule('', this.firstLoadMemorySpaceColor, true);
    } else if (this.moduleList.length > 0) {
      this.pinSelectedModuleToUrl(this.selectedModule);
      this.loadModule(this.selectedModule, this.selectedMemorySpaceColor, true);
    } else {
      this.dataService
        .getModuleList(this.sessionId)
        .pipe(takeUntil(this.destroyed))
        .subscribe((moduleList: string) => {
          if (!moduleList) {
            this.throbber.stop();
            setLoadingState(false, this.store);
            return;
          }
          this.moduleList = moduleList.split(',');
          // No need to regenerate modules.
          this.dataService.disableCacheRegeneration();
          this.pinSelectedModuleToUrl(this.selectedModule);
          this.loadModule(
            this.selectedModule,
            this.selectedMemorySpaceColor,
            true,
          );
        });
    }
  }

  /**
   * Synchronizes navigation query parameters with the parent window URL
   * history, preserving parent window hash and deleting legacy moduleName.
   */
  private syncUrlParams(params: Readonly<Record<string, string>>): void {
    syncParentUrlParams(params, ['moduleName']);
  }

  /**
   * Handles changes emitted by the `changed` emitter of the
   * `memory-viewer-control` component.
   *
   * This is invoked whenever one of the controls in `memory-viewer-control` is
   * updated (providing either a `module_name`/`moduleName` or a
   * `memorySpaceColor` change). Synchronizes the updated module to the parent
   * URL history and in-iframe router state, and reloads the module data.
   */
  update(event: NavigationEvent): void {
    const module = event.module_name ?? event.moduleName ?? '';
    const memorySpaceColor = event.memorySpaceColor ?? '0';
    const moduleChanged = Boolean(module) && module !== this.selectedModule;
    const memorySpaceChanged =
      Boolean(event.memorySpaceColor) &&
      memorySpaceColor !== this.selectedMemorySpaceColor;

    if (!moduleChanged && !memorySpaceChanged) {
      return;
    }

    if (moduleChanged) {
      this.selectedModule = module;
      this.syncUrlParams({'module_name': this.selectedModule});
      this.router.navigate([], {
        queryParams: {'module_name': this.selectedModule},
        queryParamsHandling: 'merge',
        replaceUrl: true,
      });
    }

    if (memorySpaceChanged) {
      this.selectedMemorySpaceColor = memorySpaceColor;
    }

    this.loadModule(this.selectedModule, this.selectedMemorySpaceColor);
  }

  loadModule(
    module: string,
    memorySpaceColor: string,
    initialLoad = false,
  ): void {
    if (
      module === this.loadedModule &&
      memorySpaceColor === this.loadedMemorySpaceColor
    ) {
      if (initialLoad && !this.loading) {
        this.throbber.stop();
        setLoadingState(false, this.store);
      }
      return;
    }
    this.loading = true;
    this.selectedModule = module;
    this.selectedMemorySpaceColor = memorySpaceColor;
    this.dataService
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
          this.loadedModule = module;
          this.loadedMemorySpaceColor = memorySpaceColor;
          this.throbber.stop();
          setLoadingState(false, this.store);
          this.loading = false;

          this.memoryViewerPreprocessResult =
            data as MemoryViewerPreprocessResult;

          // If the caller of loadModule does not provide the module name (like
          // in xsymbol use case), parse and set selectedModule and moduleList
          // using the data from backend.
          if (module === '') {
            if (this.memoryViewerPreprocessResult) {
              this.selectedModule =
                this.memoryViewerPreprocessResult.moduleName || '';
            }
            this.moduleList = [this.selectedModule];
            this.loadedModule = this.selectedModule;
          }
        },
        error: (error: unknown) => {
          this.loadedModule = '';
          this.loadedMemorySpaceColor = '';
          this.throbber.stop();
          setLoadingState(false, this.store);
          this.loading = false;
          console.error('Failed to load memory viewer data:', error);
        },
      });
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
