import {PlatformLocation} from '@angular/common';
import {HttpClient, HttpErrorResponse, HttpParams} from '@angular/common/http';
import {Injectable} from '@angular/core';
import {Store} from '@ngrx/store';
import {API_PREFIX, DATA_API, HLO_MODULE_LIST_API, LOCAL_URL, PLUGIN_NAME} from 'org_xprof/frontend/app/common/constants/constants';
import {FileExtensionType} from 'org_xprof/frontend/app/common/constants/enums';
import {DataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {OpProfileData, OpProfileSummary} from 'org_xprof/frontend/app/components/op_profile/op_profile_data';
import {DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setErrorMessageStateAction} from 'org_xprof/frontend/app/store/actions';
import {Observable, of} from 'rxjs';
import {catchError} from 'rxjs/operators';

/** The data service class that calls API and return response. */
@Injectable()
export class DataServiceV2 implements DataServiceV2Interface {
  isLocalDevelopment = false;
  pathPrefix = '';
  searchParams?: URLSearchParams;

  constructor(
      private readonly httpClient: HttpClient,
      platformLocation: PlatformLocation,
      private readonly store: Store<{}>,
  ) {
    this.isLocalDevelopment = platformLocation.pathname === LOCAL_URL;
    if (String(platformLocation.pathname).includes(API_PREFIX + PLUGIN_NAME)) {
      this.pathPrefix =
          String(platformLocation.pathname).split(API_PREFIX + PLUGIN_NAME)[0];
    }
    this.searchParams = new URLSearchParams(window.location.search);
  }

  private get<T>(
      url: string,
      // tslint:disable-next-line:no-any
      options: {[key: string]: any} = {},
      notifyError = true,
      ): Observable<T|null> {
    return this.httpClient.get<T>(url, options)
        .pipe(
            catchError((error: HttpErrorResponse) => {
              console.log(error);
              const url = new URL(error.url || '');
              const errorMessage = 'There was an error in the requested URL ' +
                  url.pathname + url.search + '.<br><br>' +
                  '<b>message:</b> ' + error.message + '<br>' +
                  '<b>status:</b> ' + String(error.status) + '<br>' +
                  '<b>statusText:</b> ' + error.statusText + '<br>' +
                  '<b>error:</b> ' + String(error.error);
              if (notifyError) {
                this.store.dispatch(setErrorMessageStateAction({errorMessage}));
              }
              return of(null);
            }),
        );
  }

  getHttpParams(): HttpParams {
    let params = new HttpParams();
    if (this.searchParams) {
      this.searchParams.forEach((value, key) => {
        params = params.set(key, value);
      });
    }
    return params;
  }

  getData(
      sessionId: string, tool: string, host: string,
      parameters: Map<string, string> = new Map()): Observable<DataTable|null> {
    let params = new HttpParams()
                     .set('run', sessionId)
                     .set('tag', tool)
                     .set('host', host);
    parameters.forEach((value, key) => {
      params = params.set(key, value);
    });
    return this.get(this.pathPrefix + DATA_API, {'params': params}) as
        Observable<DataTable>;
  }

  getModuleList(sessionId: string): Observable<string> {
    return this.get(this.pathPrefix + HLO_MODULE_LIST_API, {
      'params': new HttpParams().set('run', sessionId),
      'responseType': 'text',
    }) as Observable<string>;
  }

  // Get graph with program id and op name is not implemented yet.
  getGraphViewerLink(
      sessionId: string, moduleName: string, opName: string, programId = '') {
    if (moduleName && opName) {
      return `${window.parent.location.origin}?tool=graph_viewer&host=${
          moduleName}&opName=${opName}&run=${sessionId}#profile`;
    }
    return '';
  }

  getOpProfileSummary(data: OpProfileData): OpProfileSummary[] {
    return [
      {
        name: 'Hbm',
        value: data?.bandwidthUtilizationPercents
                   ?.[utils.MemBwType.MEM_BW_TYPE_HBM_RW],
        color: data?.bwColors?.[utils.MemBwType.MEM_BW_TYPE_HBM_RW],
      },
    ];
  }

  getCustomCallTextLink(sessionId: string, moduleName: string, opName: string) {
    return '';
  }

  downloadHloProto(
      sessionId: string,
      moduleName: string,
      type: string,
      showMetadata: boolean,
      ): Observable<string|Blob|null> {
    const tool = 'graph_viewer';
    const responseType =
        type === FileExtensionType.PROTO_BINARY ? 'blob' : 'text';
    // Host is not specified for hlo text view now, as we assume metadata the
    // same across all hosts.
    const host = '';
    const params = this.getHttpParams()
                       .set('run', sessionId)
                       .set('tag', tool)
                       .set('host', host)
                       .set('module_name', moduleName)
                       .set('type', type)
                       .set('show_metadata', String(showMetadata));
    return this.get(this.pathPrefix + DATA_API, {
      'params': params,
      'responseType': responseType,
    }) as Observable<string|Blob|null>;
  }

  setSearchParams(params: URLSearchParams) {
    this.searchParams = params;
  }
}
