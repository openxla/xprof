import {Component, OnDestroy, ChangeDetectionStrategy} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {DATA_API} from 'org_xprof/frontend/app/common/constants/constants';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {
  setCurrentToolStateAction,
  setErrorMessageStateAction,
} from 'org_xprof/frontend/app/store/actions';
import {ReplaySubject, combineLatest} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** A megascale perfetto viewer component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Eager,standalone: false,
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
        // TODO(yinzz) ActivatedRoute's params property is Observable<Params>, so
        // params in this function shouldn't be nullable,
        // https://angular.io/api/router/ActivatedRoute#params, consider changing
        // across codebase to this.sessionId = params['sessionId'] ?? '';
        this.sessionId = (params || {})['sessionId'] || '';
        this.host = (queryParams || {})['host'] || '';
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
      clearTimeout(this.initTimer);
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
    requestURL.searchParams.set('perfetto', 'true');
    if (this.host) {
      requestURL.searchParams.set('host', this.host);
    }
    return requestURL.toString();
  }

  async getDataAndOpenPerfettoUI() {
    if (!this.perfettoDataUrl) return;
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
    this.pingTimer = setInterval(() => {
      perfettoWindow.postMessage('PING', this.perfettoOrigin);
    }, 250);
  }

  private cleanupPing() {
    if (this.pingTimer !== null) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    if (this.messageHandler) {
      window.removeEventListener('message', this.messageHandler);
      this.messageHandler = null;
    }
  }

  ngOnDestroy() {
    if (this.initTimer !== null) {
      clearTimeout(this.initTimer);
      this.initTimer = null;
    }
    this.cleanupPing();
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
