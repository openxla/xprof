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
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, ReplaySubject} from 'rxjs';
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

  processQuery(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || params['tool'] || this.tool;
    this.host = params['host'] || this.host;
    this.selectedModule =
      params['moduleName'] || params['module_name'] || this.selectedModule;
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

  load() {
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
    }
    // For xsymbol session, There is only 1 module so there is no need to call
    // getModuleList before calling the analysis code.
    if (this.sessionId === 'xsymbol') {
      // Module name is set to empty, the backend server will automatically
      // choose the only 1 module. Memory space color is set to 0 (HBM) by
      // default.
      this.loadModule('', this.firstLoadMemorySpaceColor, true);
    } else if (this.moduleList.length > 0) {
      this.resolveSelectedModuleAndMemorySpace();
      this.loadModule(this.selectedModule, this.selectedMemorySpaceColor, true);
    } else {
      this.dataService
        .getModuleList(this.sessionId)
        .pipe(takeUntil(this.destroyed))
        .subscribe((moduleList: string) => {
          if (moduleList) {
            this.moduleList = moduleList.split(',');
            // No need to regenerate modules.
            this.dataService.disableCacheRegeneration();
            this.resolveSelectedModuleAndMemorySpace();
            this.loadModule(
              this.selectedModule,
              this.selectedMemorySpaceColor,
              true,
            );
          } else {
            this.throbber.stop();
            setLoadingState(false, this.store);
          }
        });
    }
  }

  /**
   * Synchronizes navigation query parameters with the parent window URL
   * history.
   */
  private syncUrlParams(params: {[key: string]: string}) {
    try {
      const searchParams = new URLSearchParams(
        window.parent?.location?.search ?? '',
      );
      for (const [key, value] of Object.entries(params)) {
        if (value) {
          searchParams.set(key, value);
        }
      }
      const pathname = window.parent?.location?.pathname ?? '';
      const queryString = searchParams.toString();
      const url = `${pathname}${queryString ? `?${queryString}` : ''}`;
      window.parent?.history?.pushState({}, '', url);
    } catch (error) {
      // In case window.parent access is blocked by cross-origin iframe security.
      console.error('Failed to sync URL parameters:', error);
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

    if (moduleChanged) {
      this.selectedModule = event.moduleName ?? '';
      this.syncUrlParams({'moduleName': this.selectedModule});
    }

    if (memorySpaceChanged) {
      this.selectedMemorySpaceColor = event.memorySpaceColor ?? '0';
    }

    this.loadModule(this.selectedModule, this.selectedMemorySpaceColor || '0');
  }

  loadModule(module: string, memorySpaceColor: string, initialLoad = false) {
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
      .subscribe((data) => {
        // Page start latency  = initial load of module list + module data
        if (initialLoad) {
          this.throbber.stop();
          setLoadingState(false, this.store);
        }
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
        }
      });
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
