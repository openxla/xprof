import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  inject,
  OnDestroy,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {DataRequestType} from 'org_xprof/frontend/app/common/constants/enums';
import {
  setCurrentToolStateAction,
  setDataRequestStateAction,
} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {KernelStats} from './kernel_stats';

/** A kernel stats adapter component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
  selector: 'kernel-stats-adapter',
  template:
    '<kernel-stats [sessionId]="sessionId" [tool]="tool" [host]="host"></kernel-stats>',
  imports: [KernelStats],
})
export class KernelStatsAdapter implements OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  readonly tool = 'kernel_stats';
  sessionId = '';
  host = '';

  private readonly route: ActivatedRoute = inject(ActivatedRoute);
  private readonly store: Store<{}> = inject(Store);
  private readonly cdr: ChangeDetectorRef = inject(ChangeDetectorRef);

  constructor() {
    combineLatest([this.route.params, this.route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        if (queryParams['tag'] && queryParams['tag'] !== this.tool) {
          return;
        }
        this.processQuery({...params, ...queryParams});
        this.update();
        this.cdr.markForCheck();
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
    this.store.dispatch(
      setDataRequestStateAction({
        dataRequest: {type: DataRequestType.KERNEL_STATS, params},
      }),
    );
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
