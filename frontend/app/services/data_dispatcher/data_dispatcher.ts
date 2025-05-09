import {inject, Injectable} from '@angular/core';
import {Store} from '@ngrx/store';
import {DataRequestType} from 'org_xprof/frontend/app/common/constants/enums';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {DataRequest} from 'org_xprof/frontend/app/store/state';
import {Observable} from 'rxjs';

import {DataDispatcherBase} from './data_dispatcher_base';

/** The data dispatcher class. */
@Injectable()
export class DataDispatcher extends DataDispatcherBase {
  params = {
    run: '',
    tag: '',
    host: '',
  };

  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);

  constructor(store: Store<{}>) {
    super(store);
  }

  override clearData(dataRequest: DataRequest) {
    this.store.dispatch(this.getActions(dataRequest));
  }

  // tslint:disable-next-line:no-any
  override load(dataRequest: DataRequest): Observable<any> {
    const sessionId = dataRequest.params.sessionId || '';
    const tool = dataRequest.params.tool || '';
    const host = dataRequest.params.host || '';

    if (dataRequest.type > DataRequestType.DATA_REQUEST_BEGIN &&
        dataRequest.type < DataRequestType.DATA_REQUEST_END) {
      this.params.run = sessionId;
      this.params.tag = tool;
      this.params.host = host;
    }

    return this.dataService.getData(sessionId, tool, host);
  }

  // tslint:disable-next-line:no-any
  override parseData(dataRequest: DataRequest, data: any) {
    if (dataRequest.type === DataRequestType.KERNEL_STATS) {
      data = (data || [{}])[0];
    }
    this.store.dispatch(this.getActions(dataRequest, data));
  }
}
