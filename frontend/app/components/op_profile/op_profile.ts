import {Component, inject, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {OpProfileProto} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setProfilingDeviceTypeAction} from 'org_xprof/frontend/app/store/actions';
import {ReplaySubject} from 'rxjs';
import {combineLatestWith, takeUntil} from 'rxjs/operators';

/** An op profile component. */
@Component({
  standalone: false,
  selector: 'op-profile',
  templateUrl: './op_profile.ng.html',
  styleUrls: ['./op_profile_common.scss']
})
export class OpProfile implements OnDestroy {
  private tool = 'hlo_op_profile';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly throbber = new Throbber(this.tool);
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);
  private readonly opProfileDataCache = new Map<string, OpProfileProto>();

  sessionId = '';
  host = '';
  moduleList: string[] = [];
  opProfileData: OpProfileProto|null = null;
  groupBy = 'program'; // Default value

  constructor(
      route: ActivatedRoute,
      private readonly store: Store<{}>,
  ) {
    route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.processQuery(params);
      this.update();
    });
  }

  processQuery(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || params['tool'] || this.tool;
    this.host = params['host'] || this.host;
  }

  update() {
    setLoadingState(true, this.store, 'Loading op profile data');
    this.throbber.start();

    const cachedData = this.opProfileDataCache.get(this.groupBy);
    if (cachedData) {
      this.opProfileData = cachedData;
      setLoadingState(false, this.store);
      this.throbber.stop();
      return;
    }

    const params = new Map<string, string>();
    params.set('group_by', this.groupBy);
    const $data =
        this.dataService.getData(this.sessionId, this.tool, this.host, params);
    const $moduleList = this.dataService.getModuleList(
        this.sessionId,
    );

    $data.pipe(combineLatestWith($moduleList), takeUntil(this.destroyed))
        .subscribe(([data, moduleList]) => {
          this.throbber.stop();
          setLoadingState(false, this.store);
          if (data) {
            this.opProfileData = data as OpProfileProto;
            this.opProfileDataCache.set(this.groupBy, this.opProfileData);
            this.store.dispatch(
                setProfilingDeviceTypeAction({
                  deviceType: this.opProfileData.deviceType,
                }),
            );
          }
          if (moduleList) {
            this.moduleList = moduleList.split(',');
          }
        });
  }

  onGroupByChange(newGroupBy: string) {
    this.groupBy = newGroupBy;
    this.update();
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
