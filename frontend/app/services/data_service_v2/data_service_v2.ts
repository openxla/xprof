import {PlatformLocation} from '@angular/common';
import {HttpClient, HttpErrorResponse, HttpParams} from '@angular/common/http';
import {Injectable} from '@angular/core';
import {Store} from '@ngrx/store';
import {API_PREFIX, CAPTURE_PROFILE_API, DATA_API, GRAPH_TYPE_DEFAULT, GRAPHVIZ_PAN_ZOOM_CONTROL, HLO_MODULE_LIST_API, HOSTS_API, LOCAL_URL, PLUGIN_NAME, RUN_TOOLS_API, RUNS_API, USE_SAVED_RESULT, CONFIG_API} from 'org_xprof/frontend/app/common/constants/constants';
import {FileExtensionType} from 'org_xprof/frontend/app/common/constants/enums';
import {CaptureProfileOptions, CaptureProfileResponse} from 'org_xprof/frontend/app/common/interfaces/capture_profile';
import {DataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {HostMetadata} from 'org_xprof/frontend/app/common/interfaces/hosts';
import {type SmartSuggestionReport} from 'org_xprof/frontend/app/common/interfaces/smart_suggestion.jsonpb_decls';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {OpProfileData, OpProfileSummary} from 'org_xprof/frontend/app/components/op_profile/op_profile_data';
import {DataServiceV2Interface, ProfilerConfig} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setErrorMessageStateAction} from 'org_xprof/frontend/app/store/actions';
import {Observable, of} from 'rxjs';
import {catchError} from 'rxjs/operators';

/** The data service class that calls API and return response. */
@Injectable()
export class DataServiceV2 implements DataServiceV2Interface {
  isLocalDevelopment = false;
  pathPrefix = '';

  constructor(
      private readonly httpClient: HttpClient,
      platformLocation: PlatformLocation,
      private readonly store: Store<{}>,
  ) {
    // Clear previous searchParams from session storage
    window.sessionStorage.removeItem('searchParams');

    const searchParamsFromUrl = new URLSearchParams(platformLocation.search);
    if (searchParamsFromUrl.toString()) {
      window.sessionStorage.setItem(
          'searchParams', searchParamsFromUrl.toString());
      // Persist the query parameters in the URL.
    }

    this.isLocalDevelopment = platformLocation.pathname === LOCAL_URL;
    if (String(platformLocation.pathname).includes(API_PREFIX + PLUGIN_NAME)) {
      this.pathPrefix =
          String(platformLocation.pathname).split(API_PREFIX + PLUGIN_NAME)[0];
    }
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
              let errorMessage = '';
              if (error.status === 0) {
                errorMessage = 'Request failed : Unable to get the profile data';
              } else {
                const urlObj = new URL(error.url || '');
                const errorString = typeof error.error === 'object' ?
                    String(error.error?.error?.message) :
                    String(error.error);

                errorMessage = 'There was an error in the requested URL ' +
                    urlObj.pathname + urlObj.search + '.<br><br>' +
                    '<b>message:</b> ' + error.message + '<br>' +
                    '<b>status:</b> ' + String(error.status) + '<br>' +
                    '<b>statusText:</b> ' + error.statusText + '<br>' +
                    '<b>error:</b> ' + errorString;
              }

              if (notifyError) {
                this.store.dispatch(setErrorMessageStateAction({errorMessage}));
              }
              return of(null);
            }),
        );
  }

  private getHttpParams(
      sessionId: string|null, tool: string|null, host?: string): HttpParams {
    let params = new HttpParams();
    const searchParams = this.getSearchParams();
    if (searchParams) {
      searchParams.forEach((value, key) => {
        params = params.set(key, value);
      });
    }
    // Ensure the input arguments are populated at last to override the
    // persistent query params in the session storage.
    if (sessionId) {
      params = params.set('run', sessionId);
    }
    if (tool) {
      params = params.set('tag', tool);
    }
    if (host) {
      params = params.set('host', host);
    }
    return params;
  }

  private getHTTPParamsForDataQuery(
      run: string, tag: string, host: string,
      parameters: Map<string, string> = new Map()): HttpParams {
    // Update searchparams with the updated run, tag and host.
    // In a Single Page App, we need to update the searchparams with the updated
    // run, tag and host on tool change for consistency.
    const searchParams = this.getSearchParams();
    searchParams.set('run', run);
    searchParams.set('tag', tag);
    if (host) {
      searchParams.set('host', host);
    }
    this.setSearchParams(searchParams);

    let params = this.getHttpParams(run, tag, host);
    parameters.forEach((value, key) => {
      params = params.set(key, value);
    });

    this.disableCacheRegeneration();
    return params;
  }

  getConfig(): Observable<ProfilerConfig|null> {
    return this.get<ProfilerConfig>(this.pathPrefix + CONFIG_API);
  }

  getData(
      sessionId: string, tool: string, host: string,
      parameters: Map<string, string> = new Map()):
      Observable<DataTable|DataTable[]|null> {
    const params =
        this.getHTTPParamsForDataQuery(sessionId, tool, host, parameters);
    return this.get(this.pathPrefix + DATA_API, {'params': params}) as
        Observable<DataTable>;
  }

  getSmartSuggestions(
      sessionId: string,
      parameters: Map<string, string> = new Map()):
      Observable<SmartSuggestionReport | null> {
    return of(null);
  }

  getModuleList(sessionId: string, graphType = GRAPH_TYPE_DEFAULT):
      Observable<string> {
    const params = this.getHttpParams('', 'graph_viewer')
                       .set('run', sessionId)
                       .set('graph_type', graphType);
    return this.get(this.pathPrefix + HLO_MODULE_LIST_API, {
      'params': params,
      'responseType': 'text',
    }) as Observable<string>;
  }

  // Get graph with program id and op name is not implemented yet.
  getGraphViewerLink(
      sessionId: string, moduleName: string, opName: string, programId = '') {
    if (moduleName && opName) {
      return `${window.parent.location.origin}?tool=graph_viewer&module_name=${
          moduleName}&node_name=${opName}&run=${sessionId}#profile`;
    }
    return '';
  }

  getGraphTypes(sessionId: string) {
    const types = [
      {
        value: GRAPH_TYPE_DEFAULT,
        label: 'Hlo Graph',
      },
    ];
    return of(types);
  }

  // Not implemented.
  getGraphOpStyles(sessionId: string): Observable<string> {
    return of('');
  }

  getGraphVizUri(sessionId: string, params: Map<string, string>): string {
    const searchParams = new URLSearchParams();
    searchParams.set('run', sessionId);
    searchParams.set('tag', 'graph_viewer');
    for (const [key, value] of params.entries()) {
      searchParams.set(key, value.toString());
    }
    searchParams.set('format', 'html');
    searchParams.set('type', 'graph');
    return `${window.origin}/${this.pathPrefix}/${DATA_API}?${
        searchParams.toString()}${GRAPHVIZ_PAN_ZOOM_CONTROL}`;
  }

  // Not implemented.
  getGraphvizUrl(
      sessionId: string,
      opName: string,
      moduleName: string,
      graphWidth: number,
      showMetadata: boolean,
      mergeFusion: boolean,
      graphType: string,
      ): Observable<string> {
    return of('');
  }

  getMeGraphJson(sessionId: string, params: Map<string, string>) {
    const queryPrams = new HttpParams();
    queryPrams.set('run', sessionId);
    queryPrams.set('tag', 'graph_viewer');
    queryPrams.set('module_name', params.get('module_name') || '');
    queryPrams.set('program_id', params.get('program_id') || '');
    queryPrams.set('node_name', params.get('node_name') || '');
    queryPrams.set('graph_width', params.get('graph_width') || '');
    queryPrams.set('type', 'me_graph');
    return this.get(
               this.pathPrefix + DATA_API,
               {'params': queryPrams, 'responseType': 'text'}) as
        Observable<string>;
  }

  // Not implemented.
  getTags(sessionId: string): Observable<string[]> {
    return of([]);
  }

  getHosts(run: string, tool: string): Observable<HostMetadata[]> {
    const params = new HttpParams().set('run', run).set('tag', tool);
    return this.httpClient.get(this.pathPrefix + HOSTS_API, {params}) as
        Observable<HostMetadata[]>;
  }

  getOpProfileData(
      sessionId: string, host: string,
      params: Map<string, string>): Observable<DataTable|null> {
    return this.getData(sessionId, 'op_profile', host, params) as
        Observable<DataTable>;
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

  getCustomCallTextLink(
      sessionId: string, moduleName: string, opName: string, programId = '') {
    if (moduleName && opName) {
      return `${window.parent.location.origin}/${
          DATA_API}?tag=graph_viewer&module_name=${moduleName}&node_name=${
          opName}&run=${sessionId}&type=custom_call#profile`;
    }
    return '';
  }

  getCustomCallText(
      sessionId: string, moduleName: string, opName: string,
      programId: string): Observable<string> {
    return of('');
  }

  // Download by program id is not implemented yet, as the processor is missing
  // hlo proto map by program id.
  downloadHloProto(
      sessionId: string,
      graphType: string,
      moduleName: string,
      type: string,
      showMetadata: boolean,
      programId = '',
      ): Observable<string|Blob|null> {
    const tool = 'graph_viewer';
    const responseType =
        type === FileExtensionType.PROTO_BINARY ? 'blob' : 'text';
    // Host is not specified for hlo text view now, as we assume metadata the
    // same across all hosts.
    const host = '';
    const params = this.getHttpParams('', '')
                       .set('run', sessionId)
                       .set('tag', tool)
                       .set('host', host)
                       .set('graph_type', graphType)
                       .set('module_name', moduleName)
                       .set('type', type)
                       .set('show_metadata', String(showMetadata));
    return this.get(this.pathPrefix + DATA_API, {
      'params': params,
      'responseType': responseType,
    }) as Observable<string|Blob|null>;
  }

  // Not implemented.
  getLloSourceInfo(sessionId: string, opName: string, host = ''):
      Observable<string> {
    return of('');
  }

  getSearchParams(): URLSearchParams {
    return new URLSearchParams(
        window.sessionStorage.getItem('searchParams') || '',
    );
  }
  setSearchParams(searchParams: URLSearchParams) {
    window.sessionStorage.setItem(
        'searchParams',
        new URLSearchParams(searchParams).toString(),
    );
    const newUrl = window.location.pathname + '?' + searchParams.toString();
    window.history.replaceState({}, '', newUrl);
  }

  exportDataAsCSV(sessionId: string, tool: string, host: string) {
    const params = new HttpParams()
                       .set('run', sessionId)
                       .set('tag', tool)
                       .set('host', host)
                       .set('tqx', 'out:csv;');
    window.open( this.pathPrefix + DATA_API + '?' + params.toString(), '_blank');
  }

  getDataByModuleNameAndMemorySpace(
      tool: string,
      sessionId: string,
      host: string,
      moduleName: string,
      memorySpace: number,
      ): Observable<DataTable> {
    return this.getData(sessionId, tool, host, new Map([
                          ['module_name', moduleName],
                          ['memory_space', memorySpace.toString()],
                        ])) as Observable<DataTable>;
  }

  disableCacheRegeneration() {
    const searchParams = this.getSearchParams();
    if (!searchParams.has(USE_SAVED_RESULT)) {
      return;
    }
    searchParams.delete(USE_SAVED_RESULT);
    window.sessionStorage.setItem(
        'searchParams',
        new URLSearchParams(searchParams).toString(),
    );
  }

  /** Methods below are for 3P only */
  getRuns(): Observable<string[]|null> {
    const searchParams = this.getSearchParams();
    const sessionPath = searchParams.get('session_path');
    const runPath = searchParams.get('run_path');
    let params = new HttpParams();
    if (sessionPath) {
      params = params.set('session_path', sessionPath);
    }
    if (runPath) {
      params = params.set('run_path', runPath);
    }
    return this.get<string[]>(this.pathPrefix + RUNS_API, {'params': params});
  }

  getRunTools(run: string): Observable<string[]> {
    const params = new HttpParams().set('run', run);
    return this.get(this.pathPrefix + RUN_TOOLS_API, {'params': params}) as
        Observable<string[]>;
  }

  captureProfile(options: CaptureProfileOptions):
      Observable<CaptureProfileResponse> {
    if (this.isLocalDevelopment) {
      return of({result: 'Done'});
    }
    const params =
        new HttpParams()
            .set('service_addr', options.serviceAddr)
            .set('is_tpu_name', options.isTpuName.toString())
            .set('duration', options.duration.toString())
            .set('num_retry', options.numRetry.toString())
            .set('worker_list', options.workerList)
            .set('host_tracer_level', options.hostTracerLevel.toString())
            .set('device_tracer_level', options.deviceTracerLevel.toString())
            .set('python_tracer_level', options.pythonTracerLevel.toString())
            .set('delay', options.delay.toString());
    return this.httpClient.get<CaptureProfileResponse>(
        this.pathPrefix + CAPTURE_PROFILE_API, {params});
  }

  openUtilizationGraphviz(sessionId: string) {
    return;
  }

  isGraphvizAvailable(): boolean {
    return false;
  }
}
