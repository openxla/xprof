import 'org_xprof/frontend/app/common/interfaces/window';

import {CommonModule} from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  EventEmitter,
  Input,
  OnChanges,
  OnDestroy,
  OnInit,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatSort, MatSortModule} from '@angular/material/sort';
import {MatTableDataSource, MatTableModule} from '@angular/material/table';
import {AngularSplitModule} from 'angular-split';
import {
  isSearchEventsEvent,
  LOADING_STATUS_UPDATE_EVENT_NAME,
  SEARCH_EVENTS_EVENT_NAME,
  SearchEventsEventDetail,
  TraceViewerV2LoadingStatus,
  type TraceViewerV2Module,
} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';
import {interval, ReplaySubject, Subject, Subscription} from 'rxjs';
import {debounceTime, takeUntil} from 'rxjs/operators';

/**
 * The name of the event selected custom event, dispatched from WASM in Trace
 * Viewer v2.
 */
export const EVENT_SELECTED_EVENT_NAME = 'eventselected';

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
  name: string;
  startUsFormatted?: string;
  durationUsFormatted?: string;
  stackTraceLinkHtml?: string;
  rooflineModelLinkHtml?: string;
  graphViewerLinkHtml?: string;
}

/**
 * The interface for selected event property.
 */
export interface SelectedEventProperty {
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
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: true,
  selector: 'trace-viewer-container',
  templateUrl: './trace_viewer_container.ng.html',
  styleUrls: ['./trace_viewer_container.scss'],
  imports: [
    AngularSplitModule,
    CommonModule,
    MatIconModule,
    MatProgressBarModule,
    PipesModule,
    FormsModule,
    MatButtonModule,
    MatFormFieldModule,
    MatInputModule,
    MatProgressSpinnerModule,
    MatSortModule,
    MatTableModule,
  ],
})
export class TraceViewerContainer
  implements OnInit, OnDestroy, AfterViewInit, OnChanges
{
  @Input() traceViewerModule: TraceViewerV2Module | null = null;
  @Input() url = '';
  @Input() useTraceViewerV2 = true;
  @Input() selectedEvent?: SelectedEvent | null;
  @Input() searching = false;
  isInitialLoading = true;
  @Input() eventDetailColumns: string[] = [];
  @Input() selectionStartFormat?: string;
  @Input() selectionExtentFormat?: string;

  isSingleEventTable(): boolean {
    return this.eventDetailColumns.length <= 2;
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

  selectedEventPropertiesDataSource =
    new MatTableDataSource<SelectedEventProperty>();
  @Input() set selectedEventProperties(data: SelectedEventProperty[]) {
    this.selectedEventPropertiesDataSource.data = data;
  }
  @Output()
  readonly eventSelected = new EventEmitter<EntrySelectedEventDetail | null>();
  @Output()
  readonly eventsSelected =
    new EventEmitter<EventsSelectedEventDetail | null>();
  @Output() readonly searchEvents = new EventEmitter<SearchEventsEventDetail>();
  @Output() readonly initializeWasm = new EventEmitter<void>();

  getTotal(column: string): number {
    return this.selectedEventPropertiesDataSource.data
      .map((t) => Number(t[column]))
      .filter((n) => !isNaN(n))
      .reduce((acc, value) => acc + value, 0);
  }

  @ViewChild('tvIframe') tvIframe?: ElementRef<HTMLIFrameElement>;
  @ViewChild('searchContainer') searchContainer?: ElementRef<HTMLElement>;
  @ViewChild('selectBtn') selectBtn?: ElementRef<HTMLButtonElement>;
  @ViewChild('panBtn') panBtn?: ElementRef<HTMLButtonElement>;
  @ViewChild('zoomBtn') zoomBtn?: ElementRef<HTMLButtonElement>;
  @ViewChild('timingBtn') timingBtn?: ElementRef<HTMLButtonElement>;
  @ViewChild(MatSort) set sort(matSort: MatSort | undefined) {
    if (matSort) {
      this.selectedEventPropertiesDataSource.sort = matSort;
    }
  }

  readonly TraceViewerV2LoadingStatus = TraceViewerV2LoadingStatus;
  traceViewerV2LoadingStatus: TraceViewerV2LoadingStatus =
    TraceViewerV2LoadingStatus.IDLE;
  traceViewerV2ErrorMessage?: string;
  readonly MouseMode = MouseMode;
  currentMouseMode = MouseMode.PAN;
  searchQuery = '';
  search$ = new Subject<string>();
  currentSearchQuery = '';
  searchResultCountText = '';
  readonly tutorials = TUTORIALS;
  currentTutorialIndex = 0;
  tutorialSubscription?: Subscription;
  drawerSizePercent = 30;
  timelineHeightPercent = 100;
  detailHeightPercent = 0;

  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  constructor() {
    this.search$
      .pipe(debounceTime(300), takeUntil(this.destroyed))
      .subscribe((query) => {
        this.currentSearchQuery = query;
        if (this.traceViewerModule) {
          this.traceViewerModule.application.instance().setSearchQuery(query);
          this.updateSearchResultCountText();
        } else if (!query) {
          this.searchResultCountText = '';
        }
      });
  }

  ngOnInit() {
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
  }

  ngAfterViewInit() {
    if (this.useTraceViewerV2) {
      this.initializeWasm.emit();
    } else {
      window.addEventListener('mouseup', this.mouseUpEventListener);
      window.addEventListener('keydown', this.keyDownEventListener);
    }
  }

  ngOnDestroy() {
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
    if (!this.useTraceViewerV2) {
      window.removeEventListener('mouseup', this.mouseUpEventListener);
      window.removeEventListener('keydown', this.keyDownEventListener);
    }
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
    this.stopTutorialRotation();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['selectedEvent']) {
      this.updateSplitSizes();
    }
  }

  private readonly keyDownEventListener = (event: KeyboardEvent) => {
    // Disable hotkey listening when typing in the input box
    const el = event.target as HTMLInputElement;
    if (el.type === 'text') return;
    switch (event.key) {
      case 'a':
      case 'd':
      case 's':
      case 'w':
        this.tvIframe?.nativeElement?.contentWindow?.focus();
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
  };

  private readonly mouseUpEventListener = (event: Event) => {
    const tfViewer =
      this.tvIframe?.nativeElement?.contentDocument?.querySelector(
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
  };

  private readonly mouseModeChangedEventListener = (e: Event) => {
    if (isMouseModeChangedEvent(e)) {
      this.setMouseMode(e.detail.mouseMode);
    }
  };

  private readonly eventSelectedEventListener = (e: Event) => {
    if (!isEntrySelectedEvent(e)) {
      return;
    }
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
      this.drawerSizePercent = drawerSizePercent;
    }

    // If an event is selected, the timeline height is reduced to accommodate
    // the detail view (drawer). Otherwise, the timeline takes the full height.
    this.timelineHeightPercent = this.selectedEvent
      ? 100 - this.drawerSizePercent
      : 100;
    this.detailHeightPercent = this.selectedEvent ? this.drawerSizePercent : 0;
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

  onSearchEvent(query: string) {
    this.search$.next(query);
    this.searchEvents.emit({events_query: query});
  }

  setMouseMode(mode: MouseMode) {
    this.currentMouseMode = mode;
    if (this.traceViewerModule) {
      this.traceViewerModule.application.instance().setMouseMode(mode);
    }
    // Sync focus to the corresponding button
    switch (mode) {
      case MouseMode.SELECT:
        this.selectBtn?.nativeElement?.focus();
        break;
      case MouseMode.PAN:
        this.panBtn?.nativeElement?.focus();
        break;
      case MouseMode.ZOOM:
        this.zoomBtn?.nativeElement?.focus();
        break;
      case MouseMode.TIMING:
        this.timingBtn?.nativeElement?.focus();
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
  onDragEnd({sizes}: {sizes: Array<number | '*'>}) {
    if (this.selectedEvent && sizes.length > 1) {
      // This assumes the drawer is the second area (index 1). This is safe as
      // long as the template structure remains consistent (Canvas then Drawer).
      const size = sizes[1];

      // '*' represents a wildcard size (null). We ignore it because we need a
      // numeric percentage.
      if (typeof size === 'number') {
        this.updateSplitSizes(size);
      }
    }
  }

  nextSearchResult() {
    if (this.traceViewerModule) {
      this.traceViewerModule.application
        .instance()
        .navigateToNextSearchResult();
      this.updateSearchResultCountText();
    }
  }

  prevSearchResult() {
    if (this.traceViewerModule) {
      this.traceViewerModule.application
        .instance()
        .navigateToPrevSearchResult();
      this.updateSearchResultCountText();
    }
  }

  updateSearchResultCountText() {
    if (!this.traceViewerModule || !this.currentSearchQuery) {
      this.searchResultCountText = '';
      return;
    }
    const instance = this.traceViewerModule.application.instance();
    const count = instance.getSearchResultsCount();
    const index = instance.getCurrentSearchResultIndex();
    if (count === 0) {
      this.searchResultCountText = '0 / 0';
      return;
    }
    this.searchResultCountText = `${index === -1 ? 1 : index + 1} / ${count}`;
  }

  /**
   * Handles keydown events in the search container to manage focus flow
   * and trigger actions on Enter for buttons.
   */
  onKeyDown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      // Force Enter key to trigger click on buttons within the search container,
      // as default behavior might be prevented by framework or parent components.
      const currentElement = document.activeElement;
      const container = this.searchContainer?.nativeElement;
      if (
        currentElement instanceof HTMLElement &&
        container &&
        container.contains(currentElement) &&
        currentElement.tagName.toUpperCase() === 'BUTTON'
      ) {
        currentElement.click();
        event.preventDefault();
      }
    }
  }
}
