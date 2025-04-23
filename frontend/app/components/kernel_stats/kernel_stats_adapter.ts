import {Component, inject, NgModule, OnDestroy} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {DataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import * as commonDataStoreActions from 'org_xprof/frontend/app/store/common_data_store/actions';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {KernelStatsModule} from './kernel_stats_module';

/** A kernel stats adapter component. */
@Component({
  standalone: false,
  selector: 'kernel-stats-adapter',
  template: '<kernel-stats></kernel-stats>',
})
export class KernelStatsAdapter implements OnDestroy {
  readonly tool = 'kernel_stats';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  sessionId = '';
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  constructor(route: ActivatedRoute, private readonly store: Store<{}>) {
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.sessionId = (params || {})['sessionId'] || '';
      this.update(params as NavigationEvent);
    });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  update(event: NavigationEvent) {
    const params = {
      host: event.host || '',
      sessionId: this.sessionId || event.run || '',
      tool: this.tool,
    };
    this.dataService.getData(params.sessionId, params.tool, params.host)
        .pipe(takeUntil(this.destroyed))
        .subscribe(data => {
          this.parseData(data as DataTable[]);
        });
  }

  parseData(data?: DataTable[]) {
    this.store.dispatch(commonDataStoreActions.setKernelStatsDataAction(
        {kernelStatsData: data ? data[0] : null}));
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}

@NgModule({
  declarations: [KernelStatsAdapter],
  imports: [KernelStatsModule],
  exports: [KernelStatsAdapter]
})
export class KernelStatsAdapterModule {
}
