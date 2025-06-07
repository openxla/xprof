import {Component, inject, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {MemoryViewerPreprocessResult} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {ReplaySubject} from 'rxjs';
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
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  sessionId = '';
  host = '';
  loading = false;
  private readonly throbber = new Throbber(this.tool);
  memoryViewerPreprocessResult: MemoryViewerPreprocessResult|null = null;
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

  constructor(
      route: ActivatedRoute,
      private readonly store: Store<{}>,
  ) {
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.processQuery(params);
      this.load();
    });
    this.store.dispatch(
        setCurrentToolStateAction({currentTool: 'memory_viewer'}),
    );
  }

  processQuery(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || this.tool;
    this.host = params['host'] || this.host;
  }

  load() {
    // Note that there could be 1-2 api calls depend on the session id
    // the latency measurement will cover all period
    // as measurement and loading stops only if:
    // 1. getModuleList returned empty results, no need for further operation
    // 2. loadModule is done
    setLoadingState(true, this.store, 'Loading memory viewer data');
    this.throbber.start();
    // For xsymbol session, There is only 1 module so there is no need to call
    // getModuleList before calling the analysis code.
    if (this.sessionId === 'xsymbol') {
      // Module name is set to empty, the backend server will automatically
      // choose the only 1 module. Memory space color is set to 0 (HBM) by
      // default.
      this.loadModule('', this.firstLoadMemorySpaceColor, true);
    } else {
      this.dataService.getModuleList(this.sessionId)
          .pipe(takeUntil(this.destroyed))
          .subscribe((moduleList: string) => {
            if (moduleList) {
              this.moduleList = moduleList.split(',');
              // No need to regenerate modules.
              this.dataService.disableCacheRegeneration();
              // By default, use memory space 0, which is HBM.
              this.loadModule(
                  this.moduleList[this.firstLoadModuleIndex],
                  this.firstLoadMemorySpaceColor,
                  true,
              );
            } else {
              this.throbber.stop();
              setLoadingState(false, this.store);
            }
          });
    }
  }

  update(event: NavigationEvent) {
    if (event.moduleName !== this.selectedModule ||
        event.memorySpaceColor !== this.selectedMemorySpaceColor) {
      this.loadModule(
          event.moduleName!,
          event.memorySpaceColor!,
      );
    }
  }

  loadModule(
      module: string,
      memorySpaceColor: string,
      initialLoad = false,
  ) {
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
