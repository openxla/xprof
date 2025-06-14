/**
 * @fileoverview Data service interface meant to accommodate different implementations
 */

import {HttpParams} from '@angular/common/http';
import {InjectionToken} from '@angular/core';
import {DataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {GraphTypeObject} from 'org_xprof/frontend/app/common/interfaces/graph_viewer';
import {OpProfileData, OpProfileSummary} from 'org_xprof/frontend/app/components/op_profile/op_profile_data';
import {Observable} from 'rxjs';

/** The data service class that calls API and return response. */
export interface DataServiceV2Interface {
  searchParams?: URLSearchParams;

  getData(
      sessionId: string,
      tool: string,
      host?: string,
      parameters?: Map<string, string>,
      ignoreError?: boolean,
      ): Observable<DataTable|DataTable[]|null>;

  // Returns a string of comma separated module names.
  getModuleList(sessionId: string): Observable<string>;

  getGraphViewerLink(
      sessionId: string,
      moduleName: string,
      opName: string,
      programId: string,
      ): string;

  getGraphTypes(sessionId: string): Observable<GraphTypeObject[]>;
  getGraphOpStyles(sessionId: string): Observable<string>;
  getGraphVizUri(sessionId: string, params: Map<string, string>): string;
  getGraphvizUrl(
      sessionId: string,
      opName: string,
      moduleName: string,
      graphWidth: number,
      showMetadata: boolean,
      mergeFusion: boolean,
      graphType: string,
      ): Observable<string>;
  getMeGraphJson(sessionId: string, params: Map<string, string>):
      Observable<string>;

  getTags(sessionId: string): Observable<string[]>;

  // host is needed to fetch the corresponding <host>.xplane.pb data
  // params: op_profile_limit to control the numbe of op displayed in each layer
  // of the op tree on UI
  getOpProfileData(
      sessionId: string, host: string,
      params: Map<string, string>): Observable<DataTable|null>;

  getOpProfileSummary(data: OpProfileData): OpProfileSummary[];

  getCustomCallTextLink(
      sessionId: string,
      moduleName: string,
      opName: string,
      programId: string,
      ): string;

  downloadHloProto(
      sessionId: string,
      moduleName: string,
      type: string,
      showMetadata: boolean,
      ): Observable<string|Blob|null>;

  setSearchParams(params: URLSearchParams): void;
  getSearchParams(): URLSearchParams;

  exportDataAsCSV(sessionId: string, tool: string, host: string): void;

  getHttpParams(sessionId: string, tool: string): HttpParams;

  getDataByModuleNameAndMemorySpace(
      tool: string,
      sessionId: string,
      host: string,
      moduleName: string,
      memorySpace: number,
      ): Observable<DataTable>;

  disableCacheRegeneration(): void;
}

/** Injection token for the data service interface. */
export const DATA_SERVICE_INTERFACE_TOKEN =
    new InjectionToken<DataServiceV2Interface>(
        'DataServiceV2Interface',
    );
