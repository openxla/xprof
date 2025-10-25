/**
 * @fileoverview Data service interface meant to accommodate different
 * implementations
 */

import {InjectionToken} from '@angular/core';
import {DataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {GraphTypeObject} from 'org_xprof/frontend/app/common/interfaces/graph_viewer';
import {HostMetadata} from 'org_xprof/frontend/app/common/interfaces/hosts';
import {OpProfileData, OpProfileSummary} from 'org_xprof/frontend/app/components/op_profile/op_profile_data';
import {Observable} from 'rxjs';
import {type SmartSuggestionReport} from 'org_xprof/frontend/app/common/interfaces/smart_suggestion.jsonpb_decls';

/** A serializable object with profiler configuration details. */
export interface ProfilerConfig {
  hideCaptureProfileButton: boolean;
}

/** The data service class that calls API and return response. */
export interface DataServiceV2Interface {
  /** Fetches plugin config details from the backend. */
  getConfig(): Observable<ProfilerConfig|null>;

  getData(
      sessionId: string,
      tool: string,
      host?: string,
      parameters?: Map<string, string>,
      ignoreError?: boolean,
      ): Observable<DataTable|DataTable[]|null>;

  getSmartSuggestions(
      sessionId: string,
      parameters?: Map<string, string>): Observable<SmartSuggestionReport | null>;

  // Returns a string of comma separated module names.
  getModuleList(sessionId: string, graphType?: string): Observable<string>;

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
  getHosts(sessionId: string, tool?: string): Observable<HostMetadata[]>;

  // host is needed to fetch the corresponding <host>.xplane.pb data
  // params: op_profile_limit to control the numbe of op displayed in each layer
  // of the op tree on UI
  getOpProfileData(
      sessionId: string, host: string,
      params: Map<string, string>): Observable<DataTable|null>;

  getOpProfileSummary(data: OpProfileData): OpProfileSummary[];
  // TODO(b/429042977): Do not include Custom Call text for provenance nodes.
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
      programId?: string,
      ): Observable<string|Blob|null>;

  getSearchParams(): URLSearchParams;
  setSearchParams(searchParams: URLSearchParams): void;

  exportDataAsCSV(sessionId: string, tool: string, host: string): void;

  getDataByModuleNameAndMemorySpace(
      tool: string,
      sessionId: string,
      host: string,
      moduleName: string,
      memorySpace: number,
      ): Observable<DataTable>;

  disableCacheRegeneration(): void;

  openUtilizationGraphviz(sessionId: string): void;
  isGraphvizAvailable(): boolean;
}

/** Injection token for the data service interface. */
export const DATA_SERVICE_INTERFACE_TOKEN =
    new InjectionToken<DataServiceV2Interface>(
        'DataServiceV2Interface',
    );
