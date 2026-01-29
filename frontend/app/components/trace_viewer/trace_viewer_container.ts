import {Component, ElementRef, Input, ViewChild} from '@angular/core';
import {TraceViewerV2LoadingStatus} from 'org_xprof/frontend/app/components/trace_viewer_v2/main';

/**
 * The interface for a selected event.
 */
export interface SelectedEvent {
  name: string;
  startUsFormatted?: string;
  durationUsFormatted?: string;
}

/** A trace viewer container component. */
@Component({
  standalone: false,
  selector: 'trace-viewer-container',
  templateUrl: './trace_viewer_container.ng.html',
  styleUrls: ['./trace_viewer_container.css'],
})
export class TraceViewerContainer {
  @Input() url = '';
  @Input() useTraceViewerV2 = true;
  @Input()
  traceViewerV2LoadingStatus: TraceViewerV2LoadingStatus =
      TraceViewerV2LoadingStatus.IDLE;
  @Input() traceViewerV2ErrorMessage?: string;
  @Input() tutorials: readonly string[] = [];
  @Input() currentTutorialIndex = 0;
  @Input() selectedEvent?: SelectedEvent|null;

  @ViewChild('tvIframe') tvIframe?: ElementRef<HTMLIFrameElement>;

  protected readonly TraceViewerV2LoadingStatus = TraceViewerV2LoadingStatus;
}
