import {PlatformLocation} from '@angular/common';
import {HttpClient, HttpErrorResponse, HttpParams} from '@angular/common/http';
import {Injectable} from '@angular/core';
import {Store} from '@ngrx/store';
import {API_PREFIX, CAPTURE_PROFILE_API, DATA_API, HOSTS_API, LOCAL_URL, PLUGIN_NAME, RUN_TOOLS_API, RUNS_API} from 'org_xprof/frontend/app/common/constants/constants';
import {CaptureProfileOptions, CaptureProfileResponse} from 'org_xprof/frontend/app/common/interfaces/capture_profile';
import {DataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {HostMetadata} from 'org_xprof/frontend/app/common/interfaces/hosts';
import {setErrorMessageStateAction} from 'org_xprof/frontend/app/store/actions';
import {Observable, of} from 'rxjs';
import {catchError, delay} from 'rxjs/operators';

import * as mockData from './mock_data';

/** Delay time for milisecond for testing */
const DELAY_TIME_MS = 1000;

/** The data service class that calls API and return response. */
@Injectable()
export class DataService {
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
    // Clear error message before making a new request.
    // Currently only 1 error message is displayed at a time, this may cause
    // issue for multi-request pages.
    this.store.dispatch(setErrorMessageStateAction({errorMessage: ''}));
    return this.httpClient.get<T>(url, options)
        .pipe(
            catchError((error: HttpErrorResponse) => {
              console.error(error);
              const url = new URL(error.url || '');
              const errorString = typeof error.error === 'object' ?
                  String(error.error?.error?.message) :
                  String(error.error);
              const errorMessage = 'There was an error in the requested URL ' +
                  url.pathname + url.search + '.<br><br>' +
                  '<b>message:</b> ' + error.message + '<br>' +
                  '<b>status:</b> ' + String(error.status) + '<br>' +
                  '<b>statusText:</b> ' + error.statusText + '<br>' +
                  '<b>error:</b> ' + errorString;
              if (notifyError) {
                this.store.dispatch(setErrorMessageStateAction({errorMessage}));
              }
              return of(null);
            }),
        );
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
    return this.httpClient.get(this.pathPrefix + CAPTURE_PROFILE_API, {params});
  }

  getHosts(run: string, tag: string): Observable<HostMetadata[]> {
    if (this.isLocalDevelopment) {
      return of(mockData.DATA_PLUGIN_PROFILE_HOSTS).pipe(delay(DELAY_TIME_MS));
    }
    const params = new HttpParams().set('run', run).set('tag', tag);
    return this.httpClient.get(this.pathPrefix + HOSTS_API, {params}) as
        Observable<HostMetadata[]>;
  }

  getRuns() {
    if (this.isLocalDevelopment) {
      return of(mockData.DATA_PLUGIN_PROFILE_RUNS);
    }
    return this.get(this.pathPrefix + RUNS_API);
  }

  getRunTools(run: string) {
    if (this.isLocalDevelopment) {
      return of(mockData.DATA_PLUGIN_PROFILE_RUN_TOOLS);
    }
    const params = new HttpParams().set('run', run);
    return this.get(this.pathPrefix + RUN_TOOLS_API, {'params': params});
  }

  getData(
      run: string, tag: string, host: string,
      parameters: Map<string, string> = new Map()): Observable<DataTable|null> {
    if (this.isLocalDevelopment) {
      if (tag.startsWith('overview_page')) {
        return of(mockData.DATA_PLUGIN_PROFILE_OVERVIEW_PAGE_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('input_pipeline_analyzer')) {
        return of(mockData.DATA_PLUGIN_PROFILE_INPUT_PIPELINE_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('tensorflow_stats')) {
        return of(mockData.DATA_PLUGIN_PROFILE_TENSORFLOW_STATS_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('memory_viewer')) {
        return of(mockData.DATA_PLUGIN_PROFILE_MEMORY_VIEWER_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('op_profile')) {
        return of(mockData.DATA_PLUGIN_PROFILE_OP_PROFILE_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('pod_viewer')) {
        return of(mockData.DATA_PLUGIN_PROFILE_POD_VIEWER_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('kernel_stats')) {
        return of(mockData.DATA_PLUGIN_PROFILE_KERNEL_STATS_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else if (tag.startsWith('memory_profile')) {
        return of(mockData.DATA_PLUGIN_PROFILE_MEMORY_PROFILE_DATA)
            .pipe(delay(DELAY_TIME_MS));
      } else {
        return of([]).pipe(delay(DELAY_TIME_MS));
      }
    }
    let params =
        new HttpParams().set('run', run).set('tag', tag).set('host', host);
    parameters.forEach((value, key) => {
      params = params.set(key, value);
    });
    return this.get(this.pathPrefix + DATA_API, {'params': params});
  }

  exportDataAsCSV(run: string, tag: string, host: string) {
    const params = new HttpParams()
                       .set('run', run)
                       .set('tag', tag)
                       .set('host', host)
                       .set('tqx', 'out:csv;');
    window.open(this.pathPrefix + DATA_API + '?' + params.toString(), '_blank');
  }
}
