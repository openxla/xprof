import {Component, inject, OnDestroy, ChangeDetectionStrategy} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {PodViewerDatabase} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {combineLatest, ReplaySubject, Subscription} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {PodViewerCommon} from './pod_viewer_common';

/** A pod viewer component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'pod-viewer',
  templateUrl: './pod_viewer.ng.html',
  styleUrls: ['./pod_viewer.scss']
})
export class PodViewer extends PodViewerCommon implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private dataSubscription?: Subscription;
  private readonly dataService: DataServiceV2Interface = inject(
      DATA_SERVICE_INTERFACE_TOKEN,
  );

  constructor(
      route: ActivatedRoute,
      override readonly store: Store<{}>,
  ) {
    super(store);
    combineLatest([route.params, route.queryParams])
        .pipe(takeUntil(this.destroyed))
        .subscribe(([params, queryParams]) => {
          this.update({
            run: params['sessionId'] || queryParams['sessionId'] || '',
            ...params,
            ...queryParams,
          } as NavigationEvent);
        });
  }

  update(event: NavigationEvent) {
    const run = event.run || '';
    if (!run) {
      return;
    }
    this.dataSubscription?.unsubscribe();
    setLoadingState(true, this.store, 'Loading data');

    this.dataSubscription = this.dataService
        .getData(run, event.tag || 'pod_viewer', event.host || '')
        .pipe(takeUntil(this.destroyed))
        .subscribe(data => {
          setLoadingState(false, this.store);

          this.parseData(data as PodViewerDatabase | null);
        });
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
