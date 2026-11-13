import {
  ChangeDetectionStrategy,
  Component,
  inject,
  OnDestroy,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {
  createSessionIdResolver,
  getSessionIdFromParams,
  resolveHostFromParams,
  setLoadingState,
} from 'org_xprof/frontend/app/common/utils/utils';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {combineLatest, ReplaySubject, Subscription} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A perf counters component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'perf-counters',
  templateUrl: './perf_counters.ng.html',
  styleUrls: ['./perf_counters.scss'],
})
export class PerfCounters extends Dashboard implements OnDestroy {
  tool = 'perf_counters';
  host = '';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private dataSubscription?: Subscription;
  readonly pageSizeOptions = [30, 50, 100, 200];
  private readonly throbber = new Throbber(this.tool);
  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );

  sessionId = '';
  showZeroValues = false;

  deviceType = '';

  constructor(
    route: ActivatedRoute,
    private readonly store: Store<{}>,
  ) {
    super();
    const resolveSessionId = createSessionIdResolver();
    combineLatest([route.params, route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        const prevSessionId = this.sessionId;
        this.sessionId = resolveSessionId(params, queryParams, this.sessionId);
        this.processQueryParams(queryParams, prevSessionId);
        this.update();
      });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
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
  }

  update() {
    this.dataSubscription?.unsubscribe();
    setLoadingState(true, this.store, 'Loading perf counters data');
    this.throbber.start();

    const params = new Map<string, string>([
      ['show_zeros', this.showZeroValues ? '1' : '0'],
    ]);
    this.dataSubscription = this.dataService
      .getData(this.sessionId, this.tool, this.host, params)
      .pipe(takeUntil(this.destroyed))
      .subscribe({
        next: (data) => {
          this.throbber.stop();
          setLoadingState(false, this.store);
          this.parseData(data as SimpleDataTable | null);
        },
        error: () => {
          this.throbber.stop();
          setLoadingState(false, this.store);
          this.parseData(null);
        },
      });
  }

  override parseData(data: SimpleDataTable | null) {
    if (!data) {
      this.pendingData = null;
      this.dataTable = undefined;
      this.dataView = undefined;
      return;
    }
    if (!this.ensureChartPackagesLoaded(data)) return;

    const dataTable = new google.visualization.DataTable(data);

    this.deviceType = dataTable.getTableProperty('device_type');

    const counterColumnIndex = dataTable.getColumnIndex('Counter');
    // Show 'Description' values as tooltips over the 'Counter' values.
    const descriptionColumnIndex = dataTable.getColumnIndex('Description');
    if (descriptionColumnIndex !== -1) {
      const pattern = '<div title="{1}">{0}</div>';
      const formatter = new google.visualization.PatternFormat(pattern);
      formatter.format(
        dataTable,
        /* srcColumnIndices= */ [counterColumnIndex, descriptionColumnIndex],
        /* dstColumnIndex= */ counterColumnIndex,
      );
    }

    // Visible columns
    const valueColumnIndex = dataTable.getColumnIndex('Value (Hex)');
    this.columns = [
      counterColumnIndex,
      {sourceColumn: valueColumnIndex, label: 'Value (Dec)'},
      valueColumnIndex,
    ];

    this.dataTable = dataTable;

    this.updateView();
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.pendingData = null;
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
  updateShowZeroValues(showZeroValuesCheckbox: boolean) {
    this.update();
  }

  getAdditionalParams() {
    const params = new Map<string, string>();
    return params;
  }
}
