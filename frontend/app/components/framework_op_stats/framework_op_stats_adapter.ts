import {Component, inject, NgModule, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {DataRequestType} from 'org_xprof/frontend/app/common/constants/enums';
import {FrameworkOpStatsModule} from 'org_xprof/frontend/app/components/framework_op_stats/framework_op_stats_module';
import {DATA_SERVICE_INTERFACE_TOKEN, type DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction, setDataRequestStateAction} from 'org_xprof/frontend/app/store/actions';
import * as actions from 'org_xprof/frontend/app/store/framework_op_stats/actions';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** An overview adapter component. */
@Component({
  standalone: false,
  selector: 'framework-op-stats-adapter',
  template:
      '<framework-op-stats [sessionId]="sessionId" [tool]="tool" [host]="host"></framework-op-stats>',
})
export class FrameworkOpStatsAdapter implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  sessionId = '';
  diffBaseSessionId = '';
  tool = 'framework_op_stats';
  host = '';

  constructor(
      route: ActivatedRoute,
      private readonly store: Store<{}>,
  ) {
    combineLatest([route.params, route.queryParams])
        .pipe(takeUntil(this.destroyed))
        .subscribe(([params, queryParams]) => {
          this.sessionId = params['sessionId'] || this.sessionId;
          this.processQueryParams(queryParams);
          this.store.dispatch(
              actions.setHasDiffAction(
                  {hasDiff: Boolean(this.diffBaseSessionId)}),
          );
          this.update();
        });
    this.store.dispatch(
        setCurrentToolStateAction({currentTool: 'framework_op_stats'}),
    );
    this.store.dispatch(actions.setTitleAction({title: 'Notes'}));
    this.store.dispatch(
        actions.setShowFlopRateChartAction({showFlopRateChart: true}),
    );
    this.store.dispatch(
        actions.setShowModelPropertiesAction({showModelProperties: true}),
    );
  }

  processQueryParams(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.diffBaseSessionId =
        this.dataService.getSearchParams().get('diff_base') || '';
    this.host = params['host'] || this.host;
  }

  update() {
    if (!this.sessionId) {
      return;
    }

    let params = {
      host: this.host,
      tool: this.tool,
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
