import {ChangeDetectionStrategy, Component, OnDestroy} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {DATA_API} from 'org_xprof/frontend/app/common/constants/constants';
import {
  getSessionIdFromParams,
  resolveHostFromParams,
  setLoadingState,
} from 'org_xprof/frontend/app/common/utils/utils';
import {
  setCurrentToolStateAction,
  setErrorMessageStateAction,
} from 'org_xprof/frontend/app/store/actions';
import {ReplaySubject, combineLatest} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A megascale perfetto viewer component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'megascale-perfetto',
  templateUrl: './megascale_perfetto.ng.html',
  styleUrls: ['./megascale_perfetto.scss'],
})
export class MegascalePerfetto implements OnDestroy {
  readonly tool = 'megascale_perfetto';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly throbber = new Throbber(this.tool);

  perfettoDataUrl = '';
  protected readonly perfettoOrigin = 'https://ui.perfetto.dev';
  protected perfettoUrl = `${this.perfettoOrigin}?hideSidebar=true`;
  sessionId = '';
  host = '';
  groupTinyEvents = '';
  private initTimer: number | null = null;
  private pingTimer: number | null = null;
  private isLoading = false;
  // tslint:disable-next-line:no-any
  private messageHandler: ((evt: any) => void) | null = null;

  constructor(
    route: ActivatedRoute,
    private readonly store: Store<{}>,
  ) {
    combineLatest([route.params, route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        const prevSessionId = this.sessionId;
        // Prefer the route session, because query params copied from saved
        // search params can carry a stale session. A bare route falls back to
        // the query session.
        this.sessionId =
          getSessionIdFromParams(params) ||
          getSessionIdFromParams(undefined, queryParams, this.sessionId);
        this.host = resolveHostFromParams(
          queryParams,
          prevSessionId,
          this.sessionId,
          this.host,
        );
        this.groupTinyEvents = queryParams['group_tiny_events'] || '';
        this.perfettoDataUrl = this.buildPerfettoDataURL();
        this.perfettoUrl = this.buildMegascalePerfettoUrl();
      });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  buildMegascalePerfettoUrl() {
    return `${this.perfettoOrigin}?hideSidebar=true`;
  }

  getPerfettoWindow() {
    const perfettoIframe = document.getElementById(
      'perfetto-page',
    )! as HTMLIFrameElement;
    return perfettoIframe.contentWindow;
  }

  async initPerfetto() {
    if (this.initTimer !== null) {
      window.clearTimeout(this.initTimer);
      this.initTimer = null;
    }
    // We cannot check perfettoWindow.document.readyState for megascale because of
    // cross-origin restrictions. We rely on the (load) event and the PING/PONG
    // handshake.
    await this.loadPerfetto();
  }

  async loadPerfetto() {
    if (this.isLoading) {
      return;
    }
    this.isLoading = true;
    setLoadingState(true, this.store, 'Loading perfetto data');
    try {
      this.throbber.start();
      await this.getDataAndOpenPerfettoUI();
    } finally {
      this.throbber.stop();
      setLoadingState(false, this.store);
      this.isLoading = false;
    }
  }

  buildPerfettoDataURL() {
    const path = DATA_API;
    const requestURL = new URL(
      `${path}${window.location.search}`,
      window.location.href,
    );
    requestURL.searchParams.set('session_id', this.sessionId);
    requestURL.searchParams.set('run', this.sessionId);
    requestURL.searchParams.delete('sessionId');
    requestURL.searchParams.set('perfetto', 'true');
    requestURL.searchParams.set('tag', 'megascale_stats');
    if (this.host) {
      requestURL.searchParams.set('host', this.host);
    }
    if (this.groupTinyEvents !== '') {
      requestURL.searchParams.set('group_tiny_events', this.groupTinyEvents);
    }
    return requestURL.toString();
  }

  async getDataAndOpenPerfettoUI() {
    if (!this.perfettoDataUrl) return;
    if (!this.host) {
      this.store.dispatch(
        setErrorMessageStateAction({
          errorMessage: 'A host must be specified.',
        }),
      );
      return;
    }
    const response = await fetch(this.perfettoDataUrl);
    if (response.ok) {
      const blob = await response.blob();
      const trace = await blob.arrayBuffer();
      if (trace.byteLength === 0) {
        this.store.dispatch(
          setErrorMessageStateAction({
            errorMessage: 'Trace data too big for perfetto.',
          }),
        );
      } else {
        this.openPerfettoUI(trace);
      }
    } else {
      console.error(`Fail to get perfetto view data: ${await response.text()}`);
    }
  }

  openPerfettoUI(trace: object) {
    const perfettoWindow = this.getPerfettoWindow();
    if (!perfettoWindow) {
      console.error('Perfetto window not identifiable. Try reload the data');
      return;
    }
    this.cleanupPing();
    // tslint:disable:no-any allow any type for the message object
    this.messageHandler = (evt: any) => {
      if (evt.data !== 'PONG') {
        return;
      }
      this.cleanupPing();
      const shareUrl = new URL(
        `/${this.tool}/${this.sessionId}?${window.location.search}`,
        window.location.href,
      );
      const perfettoMessage: any = {
        buffer: trace,
        title: `xprof session ${this.sessionId}`,
        url: shareUrl.toString(),
        keepApiOpen: true,
      };

      perfettoWindow.postMessage(
        {
          'perfetto': perfettoMessage,
        },
        this.perfettoOrigin,
      );
    };
    window.addEventListener('message', this.messageHandler);
    this.pingTimer = window.setInterval(() => {
      perfettoWindow.postMessage('PING', this.perfettoOrigin);
    }, 250);
  }

  private cleanupPing() {
    if (this.pingTimer !== null) {
      window.clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    if (this.messageHandler) {
      window.removeEventListener('message', this.messageHandler);
      this.messageHandler = null;
    }
  }

  ngOnDestroy() {
    if (this.initTimer !== null) {
      window.clearTimeout(this.initTimer);
      this.initTimer = null;
    }
    this.cleanupPing();
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
