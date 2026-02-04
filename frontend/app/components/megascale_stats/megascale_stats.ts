import {Component, inject, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params, Router} from '@angular/router';
import {Store} from '@ngrx/store';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';
import {Dashboard} from 'org_xprof/frontend/app/components/chart/dashboard/dashboard';
import {DataServiceV2Interface, DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {HostMetadata} from 'org_xprof/frontend/app/common/interfaces/hosts';
import {ChartDataInfo} from 'org_xprof/frontend/app/common/interfaces/chart';
import {DefaultDataProvider} from 'org_xprof/frontend/app/components/chart/default_data_provider';
import {TABLE_OPTIONS} from 'org_xprof/frontend/app/components/chart/chart_options';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {getHostsState} from 'org_xprof/frontend/app/store/selectors';
import {setCurrentToolStateAction} from 'org_xprof/frontend/app/store/actions';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {parseDiagnosticsDataTable} from 'org_xprof/frontend/app/common/utils/utils';
import {SimpleDataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Chart} from 'org_xprof/frontend/app/components/chart/chart';
import {CategoryFilter} from 'org_xprof/frontend/app/components/controls/category_filter/category_filter';
import {ExportAsCsv} from 'org_xprof/frontend/app/components/controls/export_as_csv/export_as_csv';
import {DiagnosticsView} from 'org_xprof/frontend/app/components/diagnostics_view/diagnostics_view';
import {CommonModule} from '@angular/common';
import {FormsModule} from '@angular/forms';
import {MatOptionModule} from '@angular/material/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatSelectModule} from '@angular/material/select';

const DIAGNOSTICS_INDEX = 0;
const MEGASCALE_STATS_INDEX = 1;

/** A Megascale Stats page component. */
@Component({
  standalone: true,
  selector: 'megascale-stats',
  templateUrl: 'megascale_stats.ng.html',
  styleUrls: ['megascale_stats.scss'],
  imports: [
    CommonModule,
    Chart,
    CategoryFilter,
    DiagnosticsView,
    ExportAsCsv,
    FormsModule,
    MatFormFieldModule,
    MatSelectModule,
    MatOptionModule,
  ],
})
export class MegascaleStats extends Dashboard implements OnDestroy {
  tool = 'megascale_stats';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly throbber = new Throbber(this.tool);
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  sessionId = '';
  host = '';
  hostList: string[] = [];

  diagnostics: Diagnostics = {info: [], warnings: [], errors: []};

  dataInfo: ChartDataInfo = {
    data: null,
    dataProvider: new DefaultDataProvider(),
    filters: [],
    options: {
      ...TABLE_OPTIONS,
      showRowNumber: false,
    },
  };

  constructor(
      route: ActivatedRoute,
      private readonly store: Store<{}>,
      private readonly router: Router,
  ) {
    super();
    combineLatest([route.params, route.queryParams])
        .pipe(takeUntil(this.destroyed))
        .subscribe(([params, queryParams]) => {
          this.sessionId = params['sessionId'] || this.sessionId;
          this.processQueryParams(queryParams);
          this.update();
        });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
    this.store
      .select(getHostsState)
      .pipe(takeUntil(this.destroyed))
      .subscribe((hostsmetadata: HostMetadata[]) => {
        if (!hostsmetadata || hostsmetadata.length === 0) return;
        this.hostList = hostsmetadata.map((host) => host.hostname);

        if (!this.host && this.hostList.length > 0) {
          this.host =
            hostsmetadata.find(
              (hostMetadata) => hostMetadata.hasDeviceTrace,
            )?.hostname || this.hostList[0];
          this.update();
        }
      });
  }

  processQueryParams(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || this.tool;
    this.host = params['host'] || this.host;
  }

  update() {
    setLoadingState(true, this.store, 'Loading Megascale Stats data');

    this.throbber.start();

    this.dataService.getData(this.sessionId, this.tool, this.host)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          this.throbber.stop();
          setLoadingState(false, this.store);

          if (data) {
            const d = data as SimpleDataTable[] | null;
            if (d) {
              if (d.hasOwnProperty(DIAGNOSTICS_INDEX)) {
                this.diagnostics = parseDiagnosticsDataTable(
                    d[DIAGNOSTICS_INDEX],
                );
              }
              if (d.hasOwnProperty(MEGASCALE_STATS_INDEX)) {
                this.parseData(d[MEGASCALE_STATS_INDEX]);
                this.dataInfo = {
                  ...this.dataInfo,
                  data: d[MEGASCALE_STATS_INDEX],
                };
              }
            }
          }
        });
  }

  override updateView() {
    this.dataInfo = {
      ...this.dataInfo,
      filters: this.getFilters(),
    };
  }

  openPerfetto() {
    const searchParams = this.dataService.getSearchParams();
    const queryParams: Params = {};
    searchParams.forEach((value, key) => {
      queryParams[key] = value;
    });

    if (this.host) {
      queryParams['host'] = this.host;
    }

    this.router.navigate(['/megascale_perfetto', this.sessionId], {queryParams});
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
