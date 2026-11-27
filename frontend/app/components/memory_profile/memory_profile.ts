import {
  ChangeDetectionStrategy,
  Component,
  inject,
  OnDestroy,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {MemoryProfileProto} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {
  createSessionIdResolver,
  getSessionIdFromParams,
  resolveHostFromParams,
} from 'org_xprof/frontend/app/common/utils/utils';
import {MemoryProfileBase} from 'org_xprof/frontend/app/components/memory_profile/memory_profile_base';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, ReplaySubject, Subscription} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A Memory Profile component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'memory-profile',
  templateUrl: './memory_profile.ng.html',
  styleUrls: ['./memory_profile.scss'],
})
export class MemoryProfile extends MemoryProfileBase implements OnDestroy {
  tool = 'memory_profile';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private dataSubscription?: Subscription;
  private readonly throbber = new Throbber(this.tool);
  private readonly dataService = inject(DATA_SERVICE_INTERFACE_TOKEN);

  sessionId = '';
  host = '';
  hostIds: number[] = [];
  selectedHostId = 0;
  hasIncompleteSteps = false;
  loading = false;

  constructor(
    route: ActivatedRoute,
    private readonly store: Store<{}>,
  ) {
    super();
    const resolveSessionId = createSessionIdResolver();
    combineLatest([route.params, route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        const oldSessionId = this.sessionId;
        const oldTool = this.tool;
        const oldHost = this.host;
        const oldHostId = this.selectedHostId;

        this.sessionId = resolveSessionId(params, queryParams, this.sessionId);
        this.processQueryParams(queryParams, oldSessionId);
        // Trigger update only if the parameters actually changed.
        const hasChanged =
          this.sessionId !== oldSessionId ||
          this.tool !== oldTool ||
          this.host !== oldHost ||
          this.selectedHostId !== oldHostId;
        if (hasChanged) {
          this.update();
        }
      });
    this.store.dispatch(
      setCurrentToolStateAction({currentTool: 'memory_profile'}),
    );
  }

  processQueryParams(params: Params, prevSessionId = this.sessionId) {
    this.sessionId = getSessionIdFromParams(undefined, params, this.sessionId);
    this.tool = params['tag'] || this.tool;
    this.host = resolveHostFromParams(
      params,
      prevSessionId,
      this.sessionId,
      this.host,
    );

    const sessionChanged = Boolean(
      prevSessionId && this.sessionId !== prevSessionId,
    );
    let updatedHostId = false;
    const rawHostId = params['host_id'];
    if (rawHostId != null) {
      const trimmedHostId = String(rawHostId).trim();
      if (/^\d+$/.test(trimmedHostId)) {
        const parsed = Number(trimmedHostId);
        const numHosts = !sessionChanged ? this.data?.numHosts : undefined;
        const withinBounds = !numHosts || numHosts <= 0 || parsed < numHosts;
        if (Number.isSafeInteger(parsed) && withinBounds) {
          this.selectedHostId = parsed;
          updatedHostId = true;
        }
      }
    }
    if (!updatedHostId && sessionChanged) {
      this.selectedHostId = 0;
    }
  }

  update() {
    this.dataSubscription?.unsubscribe();
    this.loading = true;
    this.throbber.start();

    this.dataSubscription = this.dataService
      .getData(
        this.sessionId,
        this.tool,
        this.host,
        new Map([['host_id', this.selectedHostId.toString()]]),
      )
      .pipe(takeUntil(this.destroyed))
      .subscribe({
        next: (data) => {
          this.throbber.stop();
          this.loading = false;
          this.parseData(data as MemoryProfileProto | null);
        },
        error: () => {
          this.throbber.stop();
          this.loading = false;
          this.parseData(null);
        },
      });
  }

  override parseData(data: MemoryProfileProto | null) {
    super.parseData(data);

    const hostIds = [];
    if (this.data) {
      const numHosts = this.data.numHosts || 0;
      for (let i = 0; i < numHosts; ++i) {
        hostIds.push(i);
      }
    }
    if (hostIds.length === 0) {
      hostIds.push(0);
    }
    this.hostIds = hostIds;

    this.updateIncompleteSteps();
  }

  updateIncompleteSteps() {
    let hasIncompleteSteps = false;
    const allocatorProfile = this.selectedMemoryId
      ? this.data?.memoryProfilePerAllocator?.[this.selectedMemoryId]
      : undefined;
    const snapshots =
      allocatorProfile?.sampledTimelineSnapshots ??
      allocatorProfile?.memoryProfileSnapshots;
    if (snapshots) {
      hasIncompleteSteps =
        snapshots.length === 0 ||
        Number(snapshots[snapshots.length - 1]?.activityMetadata?.stepId ?? 0) <
          1;
    }
    this.hasIncompleteSteps = hasIncompleteSteps;
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
