import {PlatformLocation} from '@angular/common';
import {HttpClient} from '@angular/common/http';
import {AfterViewInit, Component, inject, Injector, OnDestroy, OnInit, ViewChild} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {API_PREFIX, DATA_API, PLUGIN_NAME} from 'org_xprof/frontend/app/common/constants/constants';
import {HostMetadata} from 'org_xprof/frontend/app/common/interfaces/hosts';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {EntrySelectedEventDetail, SelectedEvent, SelectedEventProperty, TraceViewerContainer, } from 'org_xprof/frontend/app/components/trace_viewer_container/trace_viewer_container';
import {TraceData as MainTraceData, traceViewerV2Main, TraceViewerV2Module} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {getHostsState} from 'org_xprof/frontend/app/store/selectors';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

interface TraceData {
  traceEvents?: Array<{[key: string]: unknown}>;
  [key: string]: unknown;
}

/**
 * The name of the event selected custom event, dispatched from WASM in Trace
 * Viewer v2.
 */
export const EVENT_SELECTED_EVENT_NAME = 'eventselected';

/** A trace viewer component. */
@Component({
  standalone: false,
  selector: 'trace-viewer',
  templateUrl: './trace_viewer.ng.html',
  styleUrls: ['./trace_viewer.css']
})
export class TraceViewer implements OnInit, AfterViewInit, OnDestroy {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private navigationEvent: NavigationEvent = {};
  private readonly injector = inject(Injector);
  private readonly store = inject(Store<{}>);

  url = '';
  pathPrefix = '';
  sourceCodeServiceIsAvailable = false;
  hostList: string[] = [];
  useTraceViewerV2 = true;
  traceViewerModule: TraceViewerV2Module|null = null;
  selectedEvent: SelectedEvent|null = null;
  selectedEventProperties: SelectedEventProperty[] = [];
  eventDetailColumns: string[] = ['property', 'value'];
  private readonly eventArgsCache = new Map<string, {[key: string]: string}>();
  private queryString = '';
  searching = false;

  @ViewChild(TraceViewerContainer, {static: false})
  container?: TraceViewerContainer;

  constructor(
      private readonly httpClient: HttpClient,
      platformLocation: PlatformLocation,
      route: ActivatedRoute,
  ) {
    if (String(platformLocation.pathname).includes(API_PREFIX + PLUGIN_NAME)) {
      this.pathPrefix =
          String(platformLocation.pathname).split(API_PREFIX + PLUGIN_NAME)[0];
    }
    combineLatest(
        [route.params, route.queryParams, this.store.select(getHostsState)])
        .pipe(takeUntil(this.destroyed))
        .subscribe(([params, queryParams, hostsMetadata]) => {
          if (hostsMetadata && hostsMetadata.length > 0) {
            this.hostList =
                hostsMetadata.map((host: HostMetadata) => host.hostname);
          }
          this.navigationEvent = {...params, ...queryParams};
          this.update(this.navigationEvent);
        });

    // We don't need the source code service to be persistently available.
    // We temporarily use the service to check if it is available and show
    // UI accordingly.
    const sourceCodeService =
        this.injector.get(SOURCE_CODE_SERVICE_INTERFACE_TOKEN, null);
    sourceCodeService?.isAvailable()
        .pipe(takeUntil(this.destroyed))
        .subscribe((isAvailable) => {
          this.sourceCodeServiceIsAvailable = isAvailable;
        });
  }

  ngOnInit() {
    // Event listeners are handled by TraceViewerContainer.
  }

  ngAfterViewInit() {
    if (this.useTraceViewerV2) {
      // Use setTimeout to defer initialization to the next tick to avoid
      // ExpressionChangedAfterItHasBeenCheckedError when setting 'loading'
      // state.
      setTimeout(() => {
        this.initializeWasmApp();
      });
    }
  }

  async initializeWasmApp() {
    this.traceViewerModule = await traceViewerV2Main();
    this.update(this.navigationEvent);
  }

  update(event: NavigationEvent) {
    const isStreaming = (event.tag === 'trace_viewer@');
    const run = event.run || '';
    const tag = event.tag || '';
    const runPath = event.run_path || '';
    const sessionPath = event.session_path || '';
    this.queryString = `run=${run}&tag=${tag}`;

    if (sessionPath) {
      this.queryString += `&session_path=${sessionPath}`;
    } else if (runPath) {
      this.queryString += `&run_path=${runPath}`;
    }

    if (event.hosts && typeof event.hosts === 'string') {
      // Since event.hosts is a comma-separated string, we can use it directly.
      this.queryString += `&hosts=${event.hosts}`;
    } else if (event.host) {
      this.queryString += `&host=${event.host}`;
    } else {
      this.queryString +=
          `&host=${this.hostList.length > 0 ? this.hostList[0] : ''}`;
    }

    const traceDataUrl = `${this.pathPrefix}${DATA_API}?${this.queryString}`;

    if (this.useTraceViewerV2) {
      if (this.traceViewerModule && this.traceViewerModule.loadJsonData) {
        this.traceViewerModule.loadJsonData(traceDataUrl);
      }
    } else {
      this.url = `${this.pathPrefix}${API_PREFIX}${
          PLUGIN_NAME}/trace_viewer_index.html?is_streaming=${
          isStreaming}&is_oss=true&trace_data_url=${
          encodeURIComponent(traceDataUrl)}&source_code_service=${
          this.sourceCodeServiceIsAvailable}`;
    }
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }

  onEventSelected(event: EntrySelectedEventDetail|null) {
    if (!event) {
      this.selectedEvent = null;
      this.selectedEventProperties = [];
      return;
    }

    this.selectedEvent = {
      name: event.name,
      startUsFormatted: event.startUsFormatted,
      durationUsFormatted: event.durationUsFormatted,
    };

    const properties: SelectedEventProperty[] = [];
    properties.push({property: 'Name', value: event.name});
    properties.push({property: 'Start Time', value: event.startUsFormatted});
    properties.push({property: 'Duration', value: event.durationUsFormatted});
    if (event.hloModuleName) {
      properties.push({property: 'HLO Module', value: event.hloModuleName});
    }
    if (event.hloOpName) {
      properties.push({property: 'HLO Op', value: event.hloOpName});
    }
    this.selectedEventProperties = properties;

    if (event.uid) {
      this.maybeFetchEventArgs(
          event.name, event.startUs, event.durationUs, event.uid);
    }
  }

  onSearchEvents(query: string) {
    if (!query) {
      this.traceViewerModule?.setSearchResultsInWasm({traceEvents: []});
      this.container?.updateSearchResultCountText();
      return;
    }
    this.searching = true;
    const searchParams = new URLSearchParams(this.queryString);
    searchParams.set('search_prefix', query);
    this.httpClient
        .get<TraceData>(
            `${this.pathPrefix}${DATA_API}?${searchParams.toString()}`)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          this.searching = false;
          if (this.traceViewerModule && data) {
            this.traceViewerModule.setSearchResultsInWasm({
              ...data,
              traceEvents: data.traceEvents || [],
            } as MainTraceData);
            this.container?.updateSearchResultCountText();
          }
        });
  }

  private maybeFetchEventArgs(
      name: string, startUs: number, durationUs: number, uid: string) {
    const key = `${name}-${startUs}-${durationUs}`;
    if (this.eventArgsCache.has(key)) {
      if (this.selectedEvent) {
        this.addArgsToSelectedEvent(this.eventArgsCache.get(key)!);
      }
      return;
    }

    const params = new URLSearchParams(this.queryString);
    params.set('event_name', name);
    params.set('start_time_ms', (startUs / 1000).toString());
    params.set('duration_ms', (durationUs / 1000).toString());
    params.set('unique_id', Math.floor(Number(uid)).toString());

    this.httpClient
        .get<TraceData>(`${this.pathPrefix}${DATA_API}?${params.toString()}`)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          const traceData = data as TraceData;
          if (!traceData || !traceData.traceEvents ||
              traceData.traceEvents.length === 0) {
            return;
          }
          const lastEvent =
              traceData.traceEvents[traceData.traceEvents.length - 1];
          if (lastEvent['ph'] === 'X' && this.selectedEvent &&
              lastEvent['args']) {
            const args = lastEvent['args'] as {[key: string]: string};
            this.eventArgsCache.set(key, args);
            this.addArgsToSelectedEvent(args);
          }
        });
  }

  private addArgsToSelectedEvent(args: {[key: string]: string}) {
    if (!this.selectedEvent) return;
    const properties = [...this.selectedEventProperties];
    for (const key of Object.keys(args)) {
      properties.push({property: key, value: args[key]});
    }
    this.selectedEventProperties = properties;
  }
}
