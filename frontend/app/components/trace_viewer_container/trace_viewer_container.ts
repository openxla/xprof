import 'org_xprof/frontend/app/common/interfaces/window';
import 'org_xprof/frontend/app/components/trace_viewer_v2/customization_panel';
import 'org_xprof/frontend/app/components/trace_viewer_v2/help_dialog';

import {CommonModule} from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  CUSTOM_ELEMENTS_SCHEMA,
  effect,
  ElementRef,
  EventEmitter,
  inject,
  input,
  NgZone,
  OnChanges,
  OnDestroy,
  OnInit,
  Output,
  SimpleChanges,
  viewChild,
  ViewChild,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatSort, MatSortModule} from '@angular/material/sort';
import {MatTableDataSource, MatTableModule} from '@angular/material/table';
import {MatTabsModule} from '@angular/material/tabs';
import {MatTooltipModule} from '@angular/material/tooltip';
import {ActivatedRoute} from '@angular/router';
import '@material/web/icon/icon.js';
import '@material/web/progress/circular-progress.js';
import '@material/web/progress/linear-progress.js';
import {AngularSplitModule} from 'angular-split';
import {NgxJsonViewerModule} from 'ngx-json-viewer';
import {TimelinePlayer} from 'org_xprof/frontend/app/components/timeline_player/timeline_player';
import {getDefaultFeatureFlag} from 'org_xprof/frontend/app/components/trace_viewer_v2/feature_flags';
import {
  getMouseModeStatusConfig,
  type MouseModeStatusConfig,
} from 'org_xprof/frontend/app/components/trace_viewer_v2/shortcuts';
import {formatHloArgsForJsonTree} from './hlo_pretty_printer';

import {
  isSearchEventsEvent,
  LOADING_STATUS_UPDATE_EVENT_NAME,
  SEARCH_EVENTS_EVENT_NAME,
  SearchEventsEventDetail,
  TraceViewerV2LoadingStatus,
  type TraceViewerV2Module,
} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {SafePipe} from 'org_xprof/frontend/app/pipes/safe_pipe';
import {fromEvent, interval, ReplaySubject, Subject, Subscription} from 'rxjs';
import {debounceTime, distinctUntilChanged, takeUntil} from 'rxjs/operators';

const DEPRECATED_STORAGE_KEYS = ['trace_viewer_timing_prompted'];

/** Default height percentage for the drawer (bottom panel). */
export const DEFAULT_DRAWER_SIZE_PERCENT = 30;

/**
 * Minimum height percentage for the drawer (bottom panel) to ensure the drag
 * handle remains permanently visible and interactive.
 */
export const MIN_DRAWER_SIZE_PERCENT = 10;

function clearDeprecatedStorageKeys(): void {
  for (const key of DEPRECATED_STORAGE_KEYS) {
    window.localStorage.removeItem(key);
  }
}

/**
 * The name of the event selected custom event, dispatched from WASM in Trace
 * Viewer v2.
 */
export const EVENT_SELECTED_EVENT_NAME = 'eventselected';

/**
 * The name of the event hovered custom event, dispatched from WASM in Trace
 * Viewer v2.
 */
export const EVENT_HOVERED_EVENT_NAME = 'eventhovered';

/**
 * The name of the events selected custom event, dispatched from WASM in Trace
 * Viewer v2.
 */
export const EVENTS_SELECTED_EVENT_NAME = 'events_selected';

/**
 * The detail of an 'EventsSelected' custom event. The properties are quoted to
 * prevent renaming during minification.
 */
export declare interface EventsSelectedEventDetail {
  // tslint:disable-next-line:enforce-name-casing
  events_selected_data: string;
}

// Type guard for the 'EventsSelected' custom event.
function isEventsSelectedEvent(
  event: Event,
): event is CustomEvent<EventsSelectedEventDetail> {
  if (!(event instanceof CustomEvent)) return false;
  const detail = event.detail as unknown;
  return (
    typeof detail === 'object' &&
    detail !== null &&
    'events_selected_data' in detail &&
    typeof (detail as EventsSelectedEventDetail).events_selected_data ===
      'string'
  );
}


/**
 * The detail of an 'EntrySelected' custom event. The properties are quoted to
 * prevent renaming during minification.
 */
export declare interface EntrySelectedEventDetail {
  eventIndex: number;
  name: string;
  startUs: number;
  durationUs: number;
  startUsFormatted: string;
  durationUsFormatted: string;
  pid?: number;
  uid?: string;
  hloModuleName?: string;
  hloOpName?: string;
  args?: Record<string, string>;
}

// Type guard for the 'EntrySelected' custom event.
function isEntrySelectedEvent(
  event: Event,
): event is CustomEvent<EntrySelectedEventDetail> {
  if (!(event instanceof CustomEvent)) return false;
  const detail = event.detail as unknown;
  return (
    typeof detail === 'object' &&
    detail !== null &&
    'eventIndex' in detail &&
    (detail as {eventIndex: unknown}).eventIndex !== undefined
  );
}


/**
 * The interface for a selected event.
 */
export interface SelectedEvent {
  eventIndex?: number;
  name: string;
  startUs?: number;
  durationUs?: number;
  startUsFormatted?: string;
  durationUsFormatted?: string;
  stackTraceLinkHtml?: string;
  rooflineModelLinkHtml?: string;
  graphViewerLinkHtml?: string;
  hloModule?: string;
  hloOpName?: string;
  args?: Record<string, unknown>;
  pid?: number;
  uid?: string;
  [key: string]: unknown;
}

/**
 * The interface for selected event property.
 */
export declare interface SelectedEventProperty {
  property?: string;
  value?: string | number;
  [key: string]: string | number | undefined;
}

/**
 * Mouse modes for trace viewer interaction.
 * Must match the values in C++ MouseMode enum.
 */
export enum MouseMode {
  SELECT = 1,
  PAN = 2,
  ZOOM = 3,
  TIMING = 4,
}

/** Event name for mouse mode changes. */
export const MOUSE_MODE_CHANGED_EVENT_NAME = 'mouse_mode_changed';

/** Detail for mouse mode changed event. */
export declare interface MouseModeChangedEventDetail {
  mouseMode: number;
}

/** Type guard for MouseModeChangedEvent. */
export function isMouseModeChangedEvent(
  event: Event,
): event is CustomEvent<MouseModeChangedEventDetail> {
  return !!(
    event instanceof CustomEvent &&
    event.detail &&
    typeof event.detail.mouseMode === 'number'
  );
}

// The tutorials to display while the trace viewer is loading.
const TUTORIALS = Object.freeze([
  'Pan: A/D or Shift+Scroll or Drag',
  'Zoom: W/S or Ctrl+Scroll',
  'Scroll: Up/Down Arrow or Scroll',
]);

// The interval at which to rotate the tutorials.
const TUTORIAL_ROTATION_INTERVAL_MS = 3_000;

/**
 * The detail of a 'LoadingStatusUpdate' custom event.
 */
declare interface LoadingStatusUpdateEventDetail {
  status: TraceViewerV2LoadingStatus;
  message?: string;
}

// Type guard for the 'LoadingStatusUpdate' custom event.
function isLoadingStatusUpdateEvent(
  event: Event,
): event is CustomEvent<LoadingStatusUpdateEventDetail> {
  return (
    event instanceof CustomEvent &&
    event.detail &&
    event.detail.status &&
    Object.values(TraceViewerV2LoadingStatus).includes(event.detail.status)
  );
}

declare interface TrackView extends Element {
  onEndPanScan_(event: Event): void;
  onEndSelection_(event: Event): void;
  onEndZoom_(event: Event): void;
}

declare interface TfTraceViewer {
  _traceViewer?: {trackView?: TrackView | null};
}

/** A trace viewer container component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
  selector: 'trace-viewer-container',
  templateUrl: './trace_viewer_container.ng.html',
  styleUrls: ['./trace_viewer_container.scss'],
  imports: [
    AngularSplitModule,
    CommonModule,
    MatIconModule,
    SafePipe,
    MatButtonModule,
    MatFormFieldModule,
    MatInputModule,
    MatSortModule,
    MatTableModule,
    MatTabsModule,
    MatTooltipModule,
    NgxJsonViewerModule,
  ],
})
export class TraceViewerContainer
  implements OnInit, OnDestroy, AfterViewInit, OnChanges
{
  private readonly el = inject(ElementRef);

  readonly traceViewerModule = input<TraceViewerV2Module | null>(null);
  readonly url = input<string>('');
  readonly useTraceViewerV2 = input<boolean>(true);
  readonly showHelpButton = input<boolean>(false);
  readonly selectedEvent = input<SelectedEvent | null | undefined>(undefined);
  readonly searching = input<boolean>(false);

  /**
   * The selected event rendered as an auto-traversed JSON tree in Trace
   * Viewer v2 (identity, timing and the full args map, with the stack trace
   * already resolved into args). Derived from `selectedEvent` whenever it
   * changes; `undefined` until the event's args are available.
   */
  selectedEventJson?: Record<string, unknown>;

  /** Whether the timeline player applies */
  enableTimelinePlayer = false;

  private readonly handleTimelineRedrawRequest = () => {
    const tvModule = this.traceViewerModule();
    if (!tvModule) return;
    tvModule.application.instance().scheduleForcedRedraw();
  };

  hoveredEvent?: SelectedEvent | null;
  hoveredEventMouseX = 0;
  hoveredEventMouseY = 0;

  isInitialLoading = true;
  readonly eventDetailColumns = input<string[]>([]);
  readonly selectionStartFormat = input<string | undefined>(undefined);
  readonly selectionExtentFormat = input<string | undefined>(undefined);

  private readonly route: ActivatedRoute = inject(ActivatedRoute);
  private readonly cdRef = inject(ChangeDetectorRef);
  private readonly changeDetectorRef = this.cdRef;
  private readonly ngZone = inject(NgZone);
  private sessionId: string | undefined = undefined;

  /** Whether the component is currently in fullscreen mode. */
  isFullscreen = false;

  private readFeatureFlag(flagName: string): boolean {
    try {
      const stored = window.localStorage.getItem(`xprof_ff_${flagName}`);
      if (stored !== null) {
        return stored === 'true';
      }
    } catch {
      // ignore
    }
    return getDefaultFeatureFlag(flagName);
  }

  get enableSourceCodeTooltip(): boolean {
    return this.readFeatureFlag('enable_source_code_tooltip');
  }

  /** Toggles the fullscreen mode for the trace viewer component. */
  toggleFullscreen(): void {
    const element = this.el.nativeElement as HTMLElement;
    if (this.isFullscreen) {
      if (document.exitFullscreen) {
        void document.exitFullscreen();
      }
    } else {
      if (element.requestFullscreen) {
        void element.requestFullscreen();
      }
    }
  }

  isSingleEventTable(): boolean {
    return this.eventDetailColumns().length <= 2;
  }

  getColumnHeader(col: string): string {
    if (this.isSingleEventTable()) {
      return '';
    }
    switch (col) {
      case 'wallDuration':
        return 'Wall Duration';
      case 'selfTime':
        return 'Self Time';
      case 'avgWallDuration':
        return 'Avg Wall Duration';
      case 'occurrences':
        return 'Occurrences';
      case 'counter':
        return 'Counter';
      case 'series':
        return 'Series';
      case 'time':
        return 'Time';
      case 'value':
        return 'Value';
      default:
        return 'Name';
    }
  }

  isPropertyBold(col: string): boolean {
    return this.isSingleEventTable() && col === 'property';
  }

  getCellContent(element: SelectedEventProperty, col: string): string {
    const val = element[col];
    if (val === undefined || val === null) {
      return '';
    }
    if (col === 'property' || col === 'value') {
      return String(val);
    }
    if (col.includes('Time') || col.includes('Duration')) {
      if (typeof val === 'number') {
        return `${val.toFixed(2)}us`;
      }
      return String(val) + 'us';
    }
    return String(val);
  }

  leftSideProperties: SelectedEventProperty[] = [];
  rightSideProperties: SelectedEventProperty[] = [];

  selectedEventPropertiesDataSource =
    new MatTableDataSource<SelectedEventProperty>();
  metricsDataSource = new MatTableDataSource<SelectedEventProperty>();
  countersDataSource = new MatTableDataSource<SelectedEventProperty>();

  metricsColumns = [
    'name',
    'occurrences',
    'wallDuration',
    'selfTime',
    'avgWallDuration',
  ];
  counterColumns = ['counter', 'series', 'time', 'value'];

  readonly selectedEventProperties = input<SelectedEventProperty[]>([]);

  trackByProperty(index: number, prop: SelectedEventProperty): string {
    return `${prop.property ?? ''}:${prop.value ?? ''}`;
  }
  @Output()
  readonly eventSelected = new EventEmitter<EntrySelectedEventDetail | null>();
  @Output()
  readonly eventsSelected =
    new EventEmitter<EventsSelectedEventDetail | null>();
  @Output() readonly searchEvents = new EventEmitter<SearchEventsEventDetail>();
  @Output() readonly initializeWasm = new EventEmitter<void>();

  @Output() readonly requestHoveredEventArgs =
    new EventEmitter<SelectedEvent>();
  @Output() readonly toggleSettings = new EventEmitter<void>();
  readonly hoveredEventArgs = input<Record<string, string> | null>(null);

  getTotal(
    column: string,
    dataSource: MatTableDataSource<SelectedEventProperty> = this
      .selectedEventPropertiesDataSource,
  ): number {
    return dataSource.data
      .map((t) => Number(t[column]))
      .filter((n) => !isNaN(n))
      .reduce((acc, value) => acc + value, 0);
  }

  readonly tvIframe = viewChild<ElementRef<HTMLIFrameElement>>('tvIframe');
  readonly searchContainer =
    viewChild<ElementRef<HTMLElement>>('searchContainer');
  readonly searchBox = viewChild<ElementRef<HTMLInputElement>>('searchBox');
  readonly selectBtn = viewChild<ElementRef<HTMLButtonElement>>('selectBtn');
  readonly panBtn = viewChild<ElementRef<HTMLButtonElement>>('panBtn');
  readonly zoomBtn = viewChild<ElementRef<HTMLButtonElement>>('zoomBtn');
  readonly timingBtn = viewChild<ElementRef<HTMLButtonElement>>('timingBtn');
  readonly timelinePlayer = viewChild(TimelinePlayer);
  readonly sort = viewChild(MatSort);

  /**
   * Whether the JSON "Event details" title is currently stuck to the top of its
   * scroll container. Drives the elevation shadow and divider on the sticky
   * header (see the .is-sticky styles in the stylesheet).
   */
  isJsonTitleStuck = false;

  /** Watches the sticky-header sentinel to toggle {@link isJsonTitleStuck}. */
  private stickyTitleObserver?: IntersectionObserver;

  /**
   * Observes a sentinel at the top of the JSON scroll content to detect when the
   * "Event details" title becomes stuck. The JSON view is rendered behind an
   * *ngIf, so this setter runs whenever the sentinel is added or removed: it
   * (re)creates the observer when the sentinel is present and tears it down
   * otherwise. The observer runs outside the Angular zone and only triggers
   * change detection when the stuck state actually flips, so scrolling never
   * runs app-wide change detection.
   */
  @ViewChild('jsonStickySentinel')
  set jsonStickySentinel(sentinel: ElementRef<HTMLElement> | undefined) {
    this.stickyTitleObserver?.disconnect();
    this.stickyTitleObserver = undefined;
    this.isJsonTitleStuck = false;

    const sentinelEl = sentinel?.nativeElement;
    const scrollRoot = sentinelEl?.closest('.split-area-inner') ?? null;
    if (!sentinelEl || !scrollRoot) return;

    this.ngZone.runOutsideAngular(() => {
      this.stickyTitleObserver = new IntersectionObserver(
        (entries) => {
          const entry = entries[0];
          if (!entry) return;
          const stuck = !entry.isIntersecting;
          if (stuck === this.isJsonTitleStuck) return;
          this.isJsonTitleStuck = stuck;
          this.cdRef.detectChanges();
        },
        {root: scrollRoot, threshold: 0},
      );
      this.stickyTitleObserver.observe(sentinelEl);
    });
  }

  readonly TraceViewerV2LoadingStatus = TraceViewerV2LoadingStatus;
  traceViewerV2LoadingStatus: TraceViewerV2LoadingStatus =
    TraceViewerV2LoadingStatus.IDLE;
  traceViewerV2ErrorMessage?: string;
  readonly MouseMode = MouseMode;
  currentMouseMode = MouseMode.PAN;

  get currentMouseModeConfig(): MouseModeStatusConfig | undefined {
    return getMouseModeStatusConfig(this.currentMouseMode);
  }
  showTimingOnboarding = false;
  private readonly TIMING_PROMPTED_STORAGE_KEY =
    'trace_viewer_timing_prompted_v2';
  searchQuery = '';
  hoveredEventRequest$ = new Subject<SelectedEvent>();
  search$ = new Subject<string>();
  currentSearchQuery = '';
  searchResultCountText = '';
  readonly tutorials = TUTORIALS;
  currentTutorialIndex = 0;
  tutorialSubscription?: Subscription;
  drawerSizePercent = DEFAULT_DRAWER_SIZE_PERCENT;
  readonly minDrawerSizePercent = MIN_DRAWER_SIZE_PERCENT;
  timelineHeightPercent = 100;
  detailHeightPercent = 0;

  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  constructor() {
    effect(() => {
      const data = this.selectedEventProperties();
      this.selectedEventPropertiesDataSource.data = data;

      const metrics = data.filter((prop) => prop.hasOwnProperty('occurrences'));
      const counters = data.filter((prop) => prop.hasOwnProperty('counter'));

      this.metricsDataSource.data = metrics;
      this.countersDataSource.data = counters;

      this.leftSideProperties = data.filter((prop) => {
        const p = prop['property'];
        return p !== 'Operands' && p !== 'Consumers';
      });
      this.rightSideProperties = data.filter((prop) => {
        const p = prop['property'];
        return p === 'Operands' || p === 'Consumers';
      });
    });

    effect(() => {
      const args = this.hoveredEventArgs();
      if (!this.hoveredEvent || !args) {
        return;
      }
      this.hoveredEvent.args = {...this.hoveredEvent.args, ...args};
      this.cdRef.markForCheck();
    });

    effect(() => {
      const matSort = this.sort();
      if (matSort) {
        this.selectedEventPropertiesDataSource.sort = matSort;
      }
    });

    this.search$
      .pipe(
        debounceTime(300),
        distinctUntilChanged(),
        takeUntil(this.destroyed),
      )
      .subscribe((query) => {
        this.currentSearchQuery = query;
        this.searchEvents.emit({events_query: query});
        const tvModule = this.traceViewerModule();
        if (tvModule) {
          tvModule.application.instance().setSearchQuery(query);
          this.updateSearchResultCountText();
        } else if (!query) {
          this.searchResultCountText = '';
        }
      });

    this.hoveredEventRequest$
      .pipe(takeUntil(this.destroyed))
      .subscribe((event) => {
        this.ngZone.run(() => {
          this.requestHoveredEventArgs.emit(event);
        });
      });
  }

  ngOnInit() {
    this.route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.sessionId =
        (params || {})['sessionId'] || (params || {})['run'] || this.sessionId;
    });

    clearDeprecatedStorageKeys();

    window.addEventListener(
      LOADING_STATUS_UPDATE_EVENT_NAME,
      this.loadingStatusUpdateEventListener,
    );
    window.addEventListener(
      EVENT_SELECTED_EVENT_NAME,
      this.eventSelectedEventListener,
    );
    window.addEventListener(
      EVENTS_SELECTED_EVENT_NAME,
      this.eventsSelectedEventListener,
    );
    window.addEventListener(
      SEARCH_EVENTS_EVENT_NAME,
      this.searchEventsEventListener,
    );
    window.addEventListener(
      MOUSE_MODE_CHANGED_EVENT_NAME,
      this.mouseModeChangedEventListener,
    );
    document.addEventListener(
      'fullscreenchange',
      this.fullscreenChangeEventListener,
    );
    window.addEventListener(
      EVENT_HOVERED_EVENT_NAME,
      this.eventHoveredEventListener,
    );
  }

  ngAfterViewInit() {

    window.addEventListener('keydown', this.keyDownEventListener);
    if (this.useTraceViewerV2()) {
      this.initializeWasm.emit();
    } else {
      window.addEventListener('mouseup', this.mouseUpEventListener);
    }
  }

  ngOnDestroy() {
    this.stickyTitleObserver?.disconnect();
    window.removeEventListener(
      'timeline-player-redraw-request',
      this.handleTimelineRedrawRequest,
    );
    window.removeEventListener(
      LOADING_STATUS_UPDATE_EVENT_NAME,
      this.loadingStatusUpdateEventListener,
    );
    window.removeEventListener(
      EVENT_SELECTED_EVENT_NAME,
      this.eventSelectedEventListener,
    );
    window.removeEventListener(
      EVENTS_SELECTED_EVENT_NAME,
      this.eventsSelectedEventListener,
    );
    window.removeEventListener(
      SEARCH_EVENTS_EVENT_NAME,
      this.searchEventsEventListener,
    );
    window.removeEventListener(
      MOUSE_MODE_CHANGED_EVENT_NAME,
      this.mouseModeChangedEventListener,
    );
    document.removeEventListener(
      'fullscreenchange',
      this.fullscreenChangeEventListener,
    );
    window.removeEventListener(
      EVENT_HOVERED_EVENT_NAME,
      this.eventHoveredEventListener,
    );
    window.removeEventListener('keydown', this.keyDownEventListener);
    if (!this.useTraceViewerV2()) {
      window.removeEventListener('mouseup', this.mouseUpEventListener);
    }
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
    this.stopTutorialRotation();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['selectedEvent']) {
      this.updateSplitSizes();
      this.selectedEventJson = this.buildSelectedEventJson();
    }
  }

  /**
   * Builds the object rendered by the JSON tree view in the v2 details panel
   * from the selected event: its identity, timing and, once resolved, the full
   * args map (the stack trace is already resolved into args by the parent
   * component). The object is built as soon as an event is selected, so the
   * JSON tree renders immediately with the identity/timing fields and simply
   * gains an `args` node once args are fetched, avoiding a jarring switch from
   * the flat property rows to the tree view.
   * Returns `undefined` only when there is no selected event.
   */
  private buildSelectedEventJson(): Record<string, unknown> | undefined {
    const event = this.selectedEvent();
    if (!event) {
      return undefined;
    }
    const json: Record<string, unknown> = {
      'name': event.name,
      'startUs': event.startUs,
      'durationUs': event.durationUs,
      'pid': event.pid,
    };
    if (event.args && Object.keys(event.args).length > 0) {
      json['args'] = formatHloArgsForJsonTree(event.args);
    }
    return json;
  }

  private readonly keyDownEventListener = (event: KeyboardEvent) => {
    if (this.useTraceViewerV2()) {
      this.handleV2KeyDown(event);
    } else {
      this.handleV1KeyDown(event);
    }
  };

  private handleV2KeyDown(event: KeyboardEvent): void {
    const el = event.target as HTMLElement;
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') return;

    if (event.key === '/') {
      this.searchBox()?.nativeElement?.focus();
      this.searchBox()?.nativeElement?.select();
      event.preventDefault();
    } else if (event.key === '?') {
      this.openHelpDialog();
      event.preventDefault();
    } else if (
      event.key === ' ' &&
      this.enableTimelinePlayer &&
      this.timelinePlayer()
    ) {
      this.timelinePlayer()?.togglePlay();
      event.preventDefault();
    } else if (event.key === ';') {
      this.toggleSettings.emit();
      event.preventDefault();
    }
  }

  private handleV1KeyDown(event: KeyboardEvent): void {
    // Disable hotkey listening when typing in the input box
    const el = event.target as HTMLInputElement;
    if (el.type === 'text') return;
    switch (event.key) {
      case 'a':
      case 'd':
      case 's':
      case 'w':
        this.tvIframe()?.nativeElement?.contentWindow?.focus();
        break;
      case '1':
        this.setMouseMode(MouseMode.SELECT);
        break;
      case '2':
        this.setMouseMode(MouseMode.PAN);
        break;
      case '3':
        this.setMouseMode(MouseMode.ZOOM);
        break;
      case '4':
        this.setMouseMode(MouseMode.TIMING);
        break;
      default:
        break;
    }
  }

  onPlay() {
    const tvModule = this.traceViewerModule();
    const player = this.timelinePlayer();
    if (!tvModule || !player) return;
    tvModule.SetPlaybackState?.(
      true,
      player.currentTime(),
      player.playbackRate(),
    );
  }

  onPause() {
    const tvModule = this.traceViewerModule();
    const player = this.timelinePlayer();
    if (!tvModule || !player) return;
    tvModule.SetPlaybackState?.(
      false,
      player.currentTime(),
      player.playbackRate(),
    );
  }

  onSeek(time: number) {
    const tvModule = this.traceViewerModule();
    const player = this.timelinePlayer();
    if (!tvModule || !player) return;
    tvModule.SetPlaybackState?.(
      player.isPlaying(),
      time,
      player.playbackRate(),
    );
  }

  onSpeedChange(speed: number) {
    const tvModule = this.traceViewerModule();
    const player = this.timelinePlayer();
    if (!tvModule || !player) return;
    tvModule.SetPlaybackState?.(
      player.isPlaying(),
      player.currentTime(),
      speed,
    );
  }

  private readonly mouseUpEventListener = (event: Event) => {
    const tfViewer =
      this.tvIframe()?.nativeElement?.contentDocument?.querySelector(
        'tf-trace-viewer',
      ) as TfTraceViewer | null;
    const trackView: TrackView | null | undefined =
      tfViewer?._traceViewer?.trackView;
    try {
      trackView?.onEndPanScan_(event);
      trackView?.onEndSelection_(event);
      trackView?.onEndZoom_(event);
    } catch (e) {}
  };

  private readonly loadingStatusUpdateEventListener = (event: Event) => {
    if (!isLoadingStatusUpdateEvent(event)) {
      return;
    }

    this.updateLoadingStatus(event.detail.status);

    if (event.detail.status !== TraceViewerV2LoadingStatus.ERROR) {
      this.traceViewerV2ErrorMessage = undefined;
    } else {
      this.traceViewerV2ErrorMessage = event.detail.message;
    }
    this.changeDetectorRef.markForCheck();
  };

  private readonly mouseModeChangedEventListener = (e: Event) => {
    if (isMouseModeChangedEvent(e)) {
      this.setMouseMode(e.detail.mouseMode);
    }
  };

  private readonly fullscreenChangeEventListener = () => {
    this.isFullscreen = !!document.fullscreenElement;
    this.changeDetectorRef.markForCheck();
  };

  private readonly eventHoveredEventListener = (e: Event) => {
    if (
      e instanceof CustomEvent &&
      e.detail &&
      e.detail.eventIndex !== undefined
    ) {
      if (e.detail.eventIndex === -1) {
        this.hoveredEvent = null;
        this.cdRef.markForCheck();
        return;
      }
      this.hoveredEvent = e.detail as SelectedEvent;
      this.hoveredEventMouseX = e.detail.mouse_x || 0;
      this.hoveredEventMouseY = e.detail.mouse_y || 0;
      this.cdRef.markForCheck();
    }
  };

  private readonly eventSelectedEventListener = (e: Event) => {
    if (!isEntrySelectedEvent(e)) {
      return;
    }
    this.updateSearchResultCountText();
    if (e.detail.eventIndex === -1) {
      this.eventSelected.emit(null);
    } else {
      this.eventSelected.emit(e.detail);
    }
  };

  private readonly eventsSelectedEventListener = (e: Event) => {
    if (isEventsSelectedEvent(e)) {
      this.eventsSelected.emit(e.detail);
    } else {
      console.warn(
        'TraceViewerContainer: Received event but failed type guard',
        e,
      );
    }
  };

  private readonly searchEventsEventListener = (e: Event) => {
    if (!isSearchEventsEvent(e)) {
      return;
    }
    this.searchEvents.emit(e.detail);
  };

  /**
   * Updates the split pane sizes.
   *
   * Sets the height percentages for the timeline and detail views based on
   * whether an event is currently selected.
   *
   * @param drawerSizePercent The new size of the drawer in percent. If
   *     provided, updates the `drawerSizePercent` property. This is undefined
   *     when called from ngOnChanges (i.e. when selectedEvent changes).
   */
  private updateSplitSizes(drawerSizePercent?: number) {
    if (drawerSizePercent !== undefined) {
      this.drawerSizePercent = Math.max(
        drawerSizePercent,
        this.minDrawerSizePercent,
      );
    } else if (this.drawerSizePercent < this.minDrawerSizePercent) {
      this.drawerSizePercent = DEFAULT_DRAWER_SIZE_PERCENT;
    }

    // If an event is selected, the timeline height is reduced to accommodate
    // the detail view (drawer). Otherwise, the timeline takes the full height.
    this.timelineHeightPercent = this.selectedEvent()
      ? 100 - this.drawerSizePercent
      : 100;
    this.detailHeightPercent = this.selectedEvent()
      ? this.drawerSizePercent
      : 0;
  }

  /**
   * Updates the loading status and starts/stops the tutorial rotation
   * accordingly.
   *
   * If the status changes to IDLE or ERROR, the tutorial rotation is stopped.
   * Otherwise (e.g., INITIALIZING, LOADING_DATA), the tutorial rotation is
   * started to provide user feedback.
   */
  private updateLoadingStatus(status: TraceViewerV2LoadingStatus) {
    if (this.traceViewerV2LoadingStatus === status) {
      return;
    }
    this.traceViewerV2LoadingStatus = status;

    if (
      this.traceViewerV2LoadingStatus === TraceViewerV2LoadingStatus.IDLE ||
      this.traceViewerV2LoadingStatus === TraceViewerV2LoadingStatus.ERROR
    ) {
      // Stop the tutorial rotation when loading is finished or failed.
      this.stopTutorialRotation();
      this.isInitialLoading = false;
    } else {
      // Start the tutorial rotation when loading is in progress.
      this.startTutorialRotation();
    }
    this.changeDetectorRef.markForCheck();
  }

  /**
   * Starts the tutorial rotation.
   *
   * This method initializes the `tutorialSubscription` to rotate through
   * tutorials at a set interval. It ensures only one subscription is active at
   * a time. The subscription lifecycle is managed here and will be terminated
   * when `stopTutorialRotation` is called or when the component is destroyed.
   */
  private startTutorialRotation() {
    if (this.tutorialSubscription) return;

    this.tutorialSubscription = interval(TUTORIAL_ROTATION_INTERVAL_MS)
      .pipe(takeUntil(this.destroyed))
      .subscribe(() => {
        this.currentTutorialIndex =
          (this.currentTutorialIndex + 1) % this.tutorials.length;
        this.changeDetectorRef.markForCheck();
      });
  }

  /**
   * Stops the tutorial rotation.
   *
   * This method unsubscribes from the `tutorialSubscription` and clears the
   * reference, stopping the interval timer.
   */
  private stopTutorialRotation() {
    if (this.tutorialSubscription) {
      this.tutorialSubscription.unsubscribe();
      this.tutorialSubscription = undefined;
    }
  }

  onSearchEvent(query: string): void {
    this.searchQuery = query;
    this.search$.next(query);
  }

  clearSearch(event?: Event): void {
    event?.stopPropagation();
    this.searchQuery = '';
    this.currentSearchQuery = '';
    const tvModule = this.traceViewerModule();
    if (tvModule) {
      tvModule.application.instance().setSearchQuery('');
    }
    this.onSearchEvent('');
  }

  dismissTimingOnboarding(): void {
    this.showTimingOnboarding = false;
    window.localStorage.setItem(this.TIMING_PROMPTED_STORAGE_KEY, 'true');
  }

  blurActiveElement(): void {
    const el = document.activeElement;
    if (el instanceof HTMLInputElement) {
      el.blur();
    }
  }

  setMouseMode(mode: MouseMode): void {
    this.currentMouseMode = mode;
    const tvModule = this.traceViewerModule();
    if (tvModule) {
      tvModule.application.instance().setMouseMode(mode);
    }
    if (mode === MouseMode.TIMING) {
      const prompted = window.localStorage.getItem(
        this.TIMING_PROMPTED_STORAGE_KEY,
      );
      if (!prompted) {
        this.showTimingOnboarding = true;
      }
    }
    // Sync focus to the corresponding button
    switch (mode) {
      case MouseMode.SELECT:
        this.selectBtn()?.nativeElement?.focus();
        break;
      case MouseMode.PAN:
        this.panBtn()?.nativeElement?.focus();
        break;
      case MouseMode.ZOOM:
        this.zoomBtn()?.nativeElement?.focus();
        break;
      case MouseMode.TIMING:
        this.timingBtn()?.nativeElement?.focus();
        break;
      default:
        break;
    }
  }

  /**
   * Handles the drag end event from the split pane.
   *
   * @param event The event data containing the new sizes of the split areas.
   *     `event.sizes` is `IOutputAreaSizes` from `angular-split`.
   */
  onDragEnd({sizes}: {sizes: Array<number | '*'>}): void {
    if (this.selectedEvent() && sizes.length > 1) {
      // This assumes the drawer is the second area (index 1). This is safe as
      // long as the template structure remains consistent (Canvas then Drawer).
      const size = sizes[1];

      // '*' represents a wildcard size (null). We ignore it because we need a
      // numeric percentage.
      if (typeof size === 'number') {
        this.updateSplitSizes(Math.max(size, this.minDrawerSizePercent));
      }
    }
  }

  private syncEffectiveSearchQuery(query?: string): void {
    const tvModule = this.traceViewerModule();
    if (!tvModule) return;
    const effectiveQuery = query || this.currentSearchQuery;
    if (effectiveQuery !== this.currentSearchQuery) {
      this.currentSearchQuery = effectiveQuery;
      this.searchQuery = effectiveQuery;
      this.searchEvents.emit({events_query: effectiveQuery});
      tvModule.application.instance().setSearchQuery(effectiveQuery);
    }
  }

  nextSearchResult(query?: string, event?: Event): void {
    event?.stopPropagation();
    const tvModule = this.traceViewerModule();
    if (!tvModule) return;
    this.syncEffectiveSearchQuery(query);
    tvModule.application.instance().navigateToNextSearchResult();
    this.updateSearchResultCountText();
  }

  prevSearchResult(query?: string, event?: Event): void {
    event?.stopPropagation();
    const tvModule = this.traceViewerModule();
    if (!tvModule) return;
    this.syncEffectiveSearchQuery(query);
    tvModule.application.instance().navigateToPrevSearchResult();
    this.updateSearchResultCountText();
  }

  updateSearchResultCountText(): void {
    const tvModule = this.traceViewerModule();
    if (!tvModule || !this.currentSearchQuery) {
      this.searchResultCountText = '';
      this.changeDetectorRef.markForCheck();
      return;
    }
    const instance = tvModule.application.instance();
    const count = instance.getSearchResultsCount();
    const index = instance.getCurrentSearchResultIndex();
    this.searchResultCountText = `${index === -1 ? 0 : index + 1} / ${count}`;
    this.changeDetectorRef.markForCheck();
  }

  openCustomizationPanel(): void {
    const panel = this.el.nativeElement.querySelector(
      'trace-viewer-customization-panel',
    ) as {openDialog?: () => void} | null;
    panel?.openDialog?.();
  }

  openHelpDialog(): void {
    const dialog = this.el.nativeElement.querySelector(
      'trace-viewer-help-dialog',
    ) as (HTMLElement & {openDialog?: () => void; open?: boolean}) | null;
    // Call openDialog() on the upgraded Lit web component instance if available;
    // fallback to setting the `open` property directly if custom element definition
    // upgrade is still pending.
    if (dialog?.openDialog) {
      dialog.openDialog();
    } else if (dialog) {
      dialog.open = true;
    }
  }
}
