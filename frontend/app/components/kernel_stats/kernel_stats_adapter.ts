import {Component, NgModule, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {DataRequestType} from 'org_xprof/frontend/app/common/constants/enums';
import {setCurrentToolStateAction, setDataRequestStateAction} from 'org_xprof/frontend/app/store/actions';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {KernelStatsModule} from './kernel_stats_module';

/** A kernel stats adapter component. */
@Component({
  standalone: false,
  selector: 'kernel-stats-adapter',
  template:
      '<kernel-stats [sessionId]="sessionId" [tool]="tool" [host]="host"></kernel-stats>',
})
export class KernelStatsAdapter implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  readonly tool = 'kernel_stats';
  sessionId = '';
  host = '';

  constructor(route: ActivatedRoute, private readonly store: Store<{}>) {
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.processQuery(params);
      this.update();
    });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  processQuery(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.host = params['host'] || this.host;
  }

  update() {
    const params = {
      sessionId: this.sessionId,
      tool: this.tool,
      host: this.host,
    };
    this.store.dispatch(setDataRequestStateAction(
        {dataRequest: {type: DataRequestType.KERNEL_STATS, params}}));
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
