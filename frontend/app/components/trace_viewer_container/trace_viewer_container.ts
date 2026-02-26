import 'org_xprof/frontend/app/common/interfaces/window';

import {CommonModule} from '@angular/common';
import {AfterViewInit, Component, ElementRef, EventEmitter, Input, OnDestroy, OnInit, Output, ViewChild} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatTableModule} from '@angular/material/table';
import {isSearchEventsEvent, LOADING_STATUS_UPDATE_EVENT_NAME, SEARCH_EVENTS_EVENT_NAME, SearchEventsEventDetail, TraceViewerV2LoadingStatus, type TraceViewerV2Module} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';
import {interval, ReplaySubject, Subject, Subscription} from 'rxjs';
import {debounceTime, takeUntil} from 'rxjs/operators';



/**
 * The name of the event selected custom event, dispatched from WASM in Trace
 * Viewer v2.
 */
export const EVENT_SELECTED_EVENT_NAME = 'eventselected';

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
  return (
      event instanceof CustomEvent && event.detail &&
      typeof event.detail.eventIndex === 'number' &&
      typeof event.detail.name === 'string' &&
      typeof event.detail.startUs === 'number' &&
      typeof event.detail.durationUs === 'number' &&
      typeof event.detail.startUsFormatted === 'string' &&
      typeof event.detail.durationUsFormatted === 'string');
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
  property: string;
  value: string|undefined;
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
      event instanceof CustomEvent && event.detail && event.detail.status &&
      Object.values(TraceViewerV2LoadingStatus).includes(event.detail.status));
}

declare interface TrackView extends Element {
  onEndPanScan_(event: Event): Function;
  onEndSelection_(event: Event): Function;
  onEndZoom_(event: Event): Function;
}

declare interface TfTraceViewer {
  _traceViewer?: {trackView?: TrackView|null;};
}

/** A trace viewer container component. */
@Component({
  standalone: true,
  selector: 'trace-viewer-container',
  templateUrl: './trace_viewer_container.ng.html',
  styleUrls: ['./trace_viewer_container.css'],
  imports: [
    CommonModule,
    MatIconModule,
    MatProgressBarModule,
    PipesModule,
    FormsModule,
    MatButtonModule,
    MatFormFieldModule,
    MatInputModule,
    MatProgressSpinnerModule,
    MatTableModule,
  ],
})
export class TraceViewerContainer implements OnInit, OnDestroy, AfterViewInit {
  @Input() traceViewerModule: TraceViewerV2Module|null = null;
  @Input() url = '';
  @Input() useTraceViewerV2 = true;
  @Input() selectedEvent?: SelectedEvent|null;
  @Input() searching = false;
  isInitialLoading = true;
  @Input() selectedEventProperties: SelectedEventProperty[] = [];
  @Input() eventDetailColumns: string[] = [];
  @Output()
  readonly eventSelected = new EventEmitter<EntrySelectedEventDetail|null>();
  @Output() readonly searchEvents = new EventEmitter<SearchEventsEventDetail>();
  @Output() readonly initializeWasm = new EventEmitter<void>();

  @ViewChild('tvIframe') tvIframe?: ElementRef<HTMLIFrameElement>;

  readonly TraceViewerV2LoadingStatus = TraceViewerV2LoadingStatus;
  traceViewerV2LoadingStatus: TraceViewerV2LoadingStatus =
      TraceViewerV2LoadingStatus.IDLE;
  traceViewerV2ErrorMessage?: string;
  searchQuery = '';
  search$ = new Subject<string>();
  currentSearchQuery = '';
  searchResultCountText = '';
  readonly tutorials = TUTORIALS;
  currentTutorialIndex = 0;
  tutorialSubscription?: Subscription;

  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  constructor() {
    this.search$.pipe(debounceTime(300), takeUntil(this.destroyed))
        .subscribe((query) => {
          this.currentSearchQuery = query;
          if (this.traceViewerModule) {
            this.traceViewerModule.Application.Instance().setSearchQuery(query);
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
        SEARCH_EVENTS_EVENT_NAME,
        this.searchEventsEventListener,
    );
  }

  ngAfterViewInit() {
    if (!this.useTraceViewerV2) {
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
        SEARCH_EVENTS_EVENT_NAME,
        this.searchEventsEventListener,
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

  private readonly keyDownEventListener = (event: KeyboardEvent) => {
    // Disable hotkey listening when typing in the input box
    const el = event.target as HTMLInputElement;
    if (el.type === 'text') return;
    switch (event.key) {
      case 'a':
      case 'd':
      case 's':
      case 'w':
      case '1':
      case '2':
      case '3':
      case '4':
        this.tvIframe?.nativeElement?.contentWindow?.focus();
        break;
      default:
        break;
    }
  };

  private readonly mouseUpEventListener = (event: Event) => {
    const tfViewer =
        this.tvIframe?.nativeElement?.contentDocument?.querySelector(
            'tf-trace-viewer',
            ) as TfTraceViewer |
        null;
    const trackView: TrackView|null|undefined =
        tfViewer?._traceViewer?.trackView;
    try {
      trackView?.onEndPanScan_(event);
      trackView?.onEndSelection_(event);
      trackView?.onEndZoom_(event);
    } catch (e) {
    }
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

  private readonly searchEventsEventListener = (e: Event) => {
    if (!isSearchEventsEvent(e)) {
      return;
    }
    this.searchEvents.emit(e.detail);
  };

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

    if (this.traceViewerV2LoadingStatus === TraceViewerV2LoadingStatus.IDLE ||
        this.traceViewerV2LoadingStatus === TraceViewerV2LoadingStatus.ERROR) {
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

    this.tutorialSubscription =
        interval(TUTORIAL_ROTATION_INTERVAL_MS)
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

  nextSearchResult() {
    if (this.traceViewerModule) {
      this.traceViewerModule.Application.Instance()
          .navigateToNextSearchResult();
      this.updateSearchResultCountText();
    }
  }

  prevSearchResult() {
    if (this.traceViewerModule) {
      this.traceViewerModule.Application.Instance()
          .navigateToPrevSearchResult();
      this.updateSearchResultCountText();
    }
  }

  updateSearchResultCountText() {
    if (!this.traceViewerModule || !this.currentSearchQuery) {
      this.searchResultCountText = '';
      return;
    }
    const instance = this.traceViewerModule.Application.Instance();
    const count = instance.getSearchResultsCount();
    const index = instance.getCurrentSearchResultIndex();
    if (count === 0) {
      this.searchResultCountText = '0 / 0';
      return;
    }
    this.searchResultCountText = `${index === -1 ? 1 : index + 1} / ${count}`;
  }
}
