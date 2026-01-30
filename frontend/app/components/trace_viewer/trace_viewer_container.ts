import {
  Component,
  ElementRef,
  Input,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import {
  LOADING_STATUS_UPDATE_EVENT_NAME,
  TraceViewerV2LoadingStatus,
} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';
import {interval, ReplaySubject, Subscription} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/**
 * The interface for a selected event.
 */
export interface SelectedEvent {
  name: string;
  startUsFormatted?: string;
  durationUsFormatted?: string;
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

/** A trace viewer container component. */
@Component({
  standalone: false,
  selector: 'trace-viewer-container',
  templateUrl: './trace_viewer_container.ng.html',
  styleUrls: ['./trace_viewer_container.css'],
})
export class TraceViewerContainer implements OnInit, OnDestroy {
  @Input() url = '';
  @Input() useTraceViewerV2 = true;
  @Input() selectedEvent?: SelectedEvent|null;

  @ViewChild('tvIframe') tvIframe?: ElementRef<HTMLIFrameElement>;

  protected readonly TraceViewerV2LoadingStatus = TraceViewerV2LoadingStatus;
  traceViewerV2LoadingStatus: TraceViewerV2LoadingStatus =
      TraceViewerV2LoadingStatus.IDLE;
  traceViewerV2ErrorMessage?: string;
  readonly tutorials = TUTORIALS;
  currentTutorialIndex = 0;
  tutorialSubscription?: Subscription;

  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  ngOnInit() {
    window.addEventListener(
        LOADING_STATUS_UPDATE_EVENT_NAME,
        this.loadingStatusUpdateEventListener,
    );
  }

  ngOnDestroy() {
    window.removeEventListener(
        LOADING_STATUS_UPDATE_EVENT_NAME,
        this.loadingStatusUpdateEventListener,
    );
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
    this.stopTutorialRotation();
  }

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
}
