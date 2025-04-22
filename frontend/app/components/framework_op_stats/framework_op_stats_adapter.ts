import {Component, inject, NgModule, OnDestroy} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {DataRequestType} from 'org_xprof/frontend/app/common/constants/enums';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {FrameworkOpStatsModule} from 'org_xprof/frontend/app/components/framework_op_stats/framework_op_stats_module';
import {DATA_SERVICE_INTERFACE_TOKEN, type DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction, setDataRequestStateAction,} from 'org_xprof/frontend/app/store/actions';
import * as actions from 'org_xprof/frontend/app/store/framework_op_stats/actions';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** An overview adapter component. */
@Component({
  standalone: false,
  selector: 'framework-op-stats-adapter',
  template: '<framework-op-stats></framework-op-stats>',
})
export class FrameworkOpStatsAdapter implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  sessionId = '';
  diffBaseSessionId = '';

  constructor(
      route: ActivatedRoute,
      private readonly store: Store<{}>,
  ) {
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.sessionId = (params || {})['sessionId'] || '';
      this.diffBaseSessionId =
          this.dataService.getHttpParams('', 'framework_op_stats')
              .get('diff_base') ||
          '';
      this.store.dispatch(
          actions.setHasDiffAction({hasDiff: Boolean(this.diffBaseSessionId)}),
      );
      this.update(params as NavigationEvent);
    });
    this.store.dispatch(
        setCurrentToolStateAction({currentTool: 'framework_op_stats'}),
    );
    this.store.dispatch(actions.setTitleAction({title: 'Notes'}));
  }

  update(event: NavigationEvent) {
    let params = {
      run: event.run || '',
      tag: event.tag || 'framework_op_stats',
      host: event.host || '',
      tool: 'framework_op_stats',
      sessionId: this.sessionId,
    };
    this.store.dispatch(
        setDataRequestStateAction({
          dataRequest: {type: DataRequestType.TENSORFLOW_STATS, params},
        }),
    );

    if (Boolean(this.diffBaseSessionId)) {
      params = {
        ...params,
        sessionId: this.diffBaseSessionId,
      };
      this.store.dispatch(
          setDataRequestStateAction({
            dataRequest: {type: DataRequestType.TENSORFLOW_STATS_DIFF, params},
          }),
      );
    }
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}

@NgModule({
  declarations: [FrameworkOpStatsAdapter],
  imports: [FrameworkOpStatsModule],
  exports: [FrameworkOpStatsAdapter],
})
export class FrameworkOpStatsAdapterModule {
}
