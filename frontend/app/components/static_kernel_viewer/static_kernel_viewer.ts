import {
  ChangeDetectionStrategy,
  Component,
  inject,
  OnDestroy,
  OnInit,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {DEFAULT_HOST} from 'org_xprof/frontend/app/common/constants/constants';
import {
  shutdownTraceViewerV2,
  traceViewerV2Main,
  TraceViewerV2Module,
} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, firstValueFrom, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** Response structure from the Kernel Viewer backend list request. */
declare interface KernelListResponse {
  // tslint:disable-next-line:enforce-name-casing
  readonly hlo_modules?: readonly string[];
  // tslint:disable-next-line:enforce-name-casing
  readonly module_kernels?: Record<string, readonly string[]>;
  readonly kernels?: readonly string[];
}

/** Component for the Static Kernel Viewer tool page. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'static-kernel-viewer',
  templateUrl: './static_kernel_viewer.ng.html',
  styleUrls: ['./static_kernel_viewer.scss'],
})
export class StaticKernelViewer implements OnInit, OnDestroy {
  readonly tool = 'kernel_viewer';
  sessionId = '';
  host = DEFAULT_HOST;

  hloModules: string[] = [];
  moduleKernels: Record<string, readonly string[]> = {};
  selectedHloModule = 'All Modules';
  allKernels: string[] = [];
  filteredKernels: string[] = [];
  selectedKernel = '';
  url = '';
  loadingKernels = false;
  filterQuery = '';

  isRailPinned = true;
  isRailHovered = false;

  traceViewerModule: TraceViewerV2Module | null = null;
  private isInitializing = false;
  private isDestroyed = false;

  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );
  private readonly destroyed = new ReplaySubject<void>(1);

  constructor(
    private readonly route: ActivatedRoute,
    private readonly store: Store<{}>,
  ) {}

  get isRailExpanded(): boolean {
    return this.isRailPinned || this.isRailHovered;
  }

  get kernelList(): string[] {
    return this.allKernels;
  }

  togglePin(): void {
    this.isRailPinned = !this.isRailPinned;
    this.notifyResize();
  }

  onMouseEnter(): void {
    if (!this.isRailHovered) {
      this.isRailHovered = true;
      if (!this.isRailPinned) {
        this.notifyResize();
      }
    }
  }

  onMouseLeave(): void {
    if (this.isRailHovered) {
      this.isRailHovered = false;
      if (!this.isRailPinned) {
        this.notifyResize();
      }
    }
  }

  ngOnInit(): void {
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));

    combineLatest([this.route.params, this.route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]: [Params, Params]) => {
        this.sessionId =
          params['sessionId'] ??
          queryParams['run'] ??
          queryParams['sessionId'] ??
          this.sessionId;
        this.host = queryParams['host'] ?? DEFAULT_HOST;
        const requestedModule = queryParams['hlo_module'];
        if (requestedModule) {
          this.selectedHloModule = requestedModule;
        }
        const requestedKernel =
          queryParams['kernel_name'] ?? queryParams['kernel'];
        if (requestedKernel) {
          this.selectedKernel = requestedKernel;
        }
        void this.loadKernelList();
      });
  }

  ngOnDestroy(): void {
    this.isDestroyed = true;
    this.destroyed.next();
    this.destroyed.complete();
    if (this.traceViewerModule !== null) {
      shutdownTraceViewerV2();
      this.traceViewerModule = null;
    }
  }

  async onInitializeWasm(): Promise<void> {
    if (this.isInitializing || this.traceViewerModule !== null) {
      return;
    }
    this.isInitializing = true;
    try {
      this.traceViewerModule = await traceViewerV2Main();
      if (this.isDestroyed) {
        if (this.traceViewerModule !== null) {
          shutdownTraceViewerV2();
          this.traceViewerModule = null;
        }
      } else if (this.url && this.traceViewerModule?.loadTraceData) {
        void this.traceViewerModule.loadTraceData(this.url);
      }
    } catch (error) {
      console.error('Failed to initialize Trace Viewer V2 WASM module:', error);
    } finally {
      this.isInitializing = false;
    }
  }

  async loadKernelList(): Promise<void> {
    if (!this.sessionId) {
      return;
    }
    this.loadingKernels = true;
    try {
      const response = await firstValueFrom(
        this.dataService.getData(
          this.sessionId,
          this.tool,
          this.host,
          new Map<string, string>([['request_type', 'list']]),
        ),
      );
      const data = response as KernelListResponse;
      if (this.isDestroyed) {
        return;
      }
      this.hloModules = data?.hlo_modules ? [...data.hlo_modules] : [];
      this.moduleKernels = data?.module_kernels ?? {};
      this.allKernels = data?.kernels ? [...data.kernels] : [];

      if (
        !this.selectedHloModule ||
        (this.selectedHloModule !== 'All Modules' &&
          !this.hloModules.includes(this.selectedHloModule))
      ) {
        this.selectedHloModule = 'All Modules';
      }

      this.updateFilteredKernels();
    } catch (error) {
      console.error('Failed to fetch kernel list:', error);
      this.hloModules = [];
      this.moduleKernels = {};
      this.allKernels = [];
      this.filteredKernels = [];
      this.selectedKernel = '';
      this.url = '';
    } finally {
      this.loadingKernels = false;
    }
  }

  onHloModuleChange(moduleName: string): void {
    this.selectedHloModule = moduleName;
    this.updateFilteredKernels();
  }

  updateFilteredKernels(): void {
    let sourceList: readonly string[] = [];
    if (!this.selectedHloModule || this.selectedHloModule === 'All Modules') {
      sourceList = this.allKernels;
    } else {
      sourceList = this.moduleKernels[this.selectedHloModule] ?? [];
    }

    if (!this.filterQuery.trim()) {
      this.filteredKernels = [...sourceList];
    } else {
      const q = this.filterQuery.toLowerCase();
      this.filteredKernels = sourceList.filter((k) =>
        k.toLowerCase().includes(q),
      );
    }

    if (this.filteredKernels.length > 0) {
      if (
        this.selectedKernel &&
        this.filteredKernels.includes(this.selectedKernel)
      ) {
        this.onKernelChange(this.selectedKernel);
      } else {
        this.onKernelChange(this.filteredKernels[0]);
      }
    } else {
      this.selectedKernel = '';
      this.url = '';
    }
  }

  applyFilter(): void {
    this.updateFilteredKernels();
  }

  clearFilter(): void {
    this.filterQuery = '';
    this.updateFilteredKernels();
  }

  onKernelChange(kernel: string): void {
    this.selectedKernel = kernel;
    if (!kernel) {
      this.url = '';
      return;
    }

    const queryParamsMap = new Map<string, string>([
      ['request_type', 'trace'],
      ['kernel_name', kernel],
    ]);
    if (this.selectedHloModule && this.selectedHloModule !== 'All Modules') {
      queryParamsMap.set('hlo_module', this.selectedHloModule);
    }

    this.url = this.dataService.getDataUrl(
      this.sessionId,
      this.tool,
      this.host,
      queryParamsMap,
    );

    if (this.traceViewerModule?.loadTraceData) {
      void this.traceViewerModule.loadTraceData(this.url);
    }
  }

  /**
   * Dispatches resize events immediately and after the CSS drawer transition
   * (200ms) to ensure the canvas recalculates its layout.
   */
  private notifyResize(): void {
    window.dispatchEvent(new Event('resize'));
    setTimeout(() => {
      window.dispatchEvent(new Event('resize'));
    }, 210);
  }
}
