import {ChangeDetectionStrategy, Component, OnDestroy} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {DATA_API} from 'org_xprof/frontend/app/common/constants/constants';
import {
  formatFetchError,
  formatFetchException,
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
  private static readonly PING_INTERVAL_MS = 250;
  private static readonly PING_TIMEOUT_MS = 10_000;

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
  private pingTimer: number | null = null;
  private pingTimeoutTimer: number | null = null;
  private isLoading = false;
  private isIframeLoaded = false;
  private lastLoadedDataUrl = '';
  private inFlightDataUrl = '';
  private cachedTraceBuffer: ArrayBuffer | null = null;
  private fetchAbortController: AbortController | null = null;
  private activeLoadId = 0;
  private messageHandler: ((evt: MessageEvent) => void) | null = null;

  constructor(
    route: ActivatedRoute,
    private readonly store: Store<{}>,
  ) {
    combineLatest([route.params, route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        const prevSessionId = this.sessionId;
        const prevDataUrl = this.perfettoDataUrl;
        const prevHost = this.host;
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
        if (!this.isIframeLoaded) {
          this.perfettoUrl = this.buildMegascalePerfettoUrl();
          return;
        }
        if (!this.host) {
          if (
            prevHost ||
            this.lastLoadedDataUrl ||
            this.inFlightDataUrl ||
            this.perfettoDataUrl !== prevDataUrl
          ) {
            this.clearTrace();
            void this.loadPerfetto();
          }
          return;
        }
        if (this.perfettoUrl === 'about:blank') {
          this.perfettoUrl = this.buildMegascalePerfettoUrl();
        }
        if (this.perfettoDataUrl !== prevDataUrl || !prevHost) {
          void this.loadPerfetto();
        }
      });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  buildMegascalePerfettoUrl() {
    return `${this.perfettoOrigin}?hideSidebar=true`;
  }

  getPerfettoWindow() {
    const perfettoIframe = document.getElementById(
      'perfetto-page',
    ) as HTMLIFrameElement | null;
    return perfettoIframe?.contentWindow ?? null;
  }

  async initPerfetto() {
    this.isIframeLoaded = true;
    // We cannot check perfettoWindow.document.readyState for megascale because of
    // cross-origin restrictions. We rely on the (load) event and the PING/PONG
    // handshake.
    await this.loadPerfetto();
  }

  async loadPerfetto(): Promise<void> {
    const targetUrl = this.perfettoDataUrl;
    if (!this.host) {
      this.resetLoadedState();
      this.store.dispatch(
        setErrorMessageStateAction({
          errorMessage: 'A host must be specified.',
        }),
      );
      return;
    }
    if (this.isLoading && this.inFlightDataUrl === targetUrl) {
      return;
    }
    const loadId = ++this.activeLoadId;
    this.abortInFlightFetch();
    this.cleanupPing();
    if (
      targetUrl &&
      targetUrl === this.lastLoadedDataUrl &&
      this.cachedTraceBuffer !== null
    ) {
      this.throbber.stop();
      setLoadingState(false, this.store);
      this.store.dispatch(setErrorMessageStateAction({errorMessage: ''}));
      this.openPerfettoUI(this.cachedTraceBuffer);
      return;
    }

    const abortController = new AbortController();
    this.fetchAbortController = abortController;
    this.inFlightDataUrl = targetUrl;
    this.isLoading = true;
    this.store.dispatch(setErrorMessageStateAction({errorMessage: ''}));
    setLoadingState(true, this.store, 'Loading perfetto data');
    try {
      this.throbber.start();
      await this.getDataAndOpenPerfettoUI(
        abortController.signal,
        loadId,
        targetUrl,
      );
    } finally {
      if (loadId === this.activeLoadId) {
        this.throbber.stop();
        setLoadingState(false, this.store);
        this.isLoading = false;
        this.inFlightDataUrl = '';
        this.fetchAbortController = null;
      }
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
    } else {
      requestURL.searchParams.delete('host');
    }
    if (this.groupTinyEvents !== '') {
      requestURL.searchParams.set('group_tiny_events', this.groupTinyEvents);
    } else {
      requestURL.searchParams.delete('group_tiny_events');
    }
    return requestURL.toString();
  }

  async getDataAndOpenPerfettoUI(
    signal?: AbortSignal,
    loadId?: number,
    dataUrl = this.perfettoDataUrl,
  ): Promise<void> {
    if (!dataUrl) return;
    if (!this.host) {
      this.store.dispatch(
        setErrorMessageStateAction({
          errorMessage: 'A host must be specified.',
        }),
      );
      return;
    }
    const isStale = () =>
      Boolean(signal?.aborted) ||
      (loadId !== undefined && loadId !== this.activeLoadId);
    try {
      const response = signal
        ? await fetch(dataUrl, {signal})
        : await fetch(dataUrl);
      if (isStale()) return;
      if (!response.ok) {
        const errorMessage = `Failed to load Perfetto data: ${await formatFetchError(response)}`;
        if (isStale()) return;
        this.store.dispatch(setErrorMessageStateAction({errorMessage}));
        return;
      }
      const blob = await response.blob();
      if (isStale()) return;
      const trace = await blob.arrayBuffer();
      if (isStale()) return;
      if (trace.byteLength === 0) {
        this.store.dispatch(
          setErrorMessageStateAction({
            errorMessage: 'Trace data too big for perfetto.',
          }),
        );
        return;
      }
      this.lastLoadedDataUrl = dataUrl;
      this.cachedTraceBuffer = trace;
      this.openPerfettoUI(trace);
    } catch (e) {
      if (isStale() || (e instanceof DOMException && e.name === 'AbortError')) {
        return;
      }
      this.store.dispatch(
        setErrorMessageStateAction({
          errorMessage: `Failed to load Perfetto data: ${formatFetchException(e)}`,
        }),
      );
    }
  }

  openPerfettoUI(trace: object) {
    const perfettoWindow = this.getPerfettoWindow();
    if (!perfettoWindow) {
      console.error('Perfetto window not identifiable. Try reload the data');
      return;
    }
    this.cleanupPing();
    this.messageHandler = (evt: MessageEvent) => {
      if (
        evt.origin !== this.perfettoOrigin ||
        evt.source !== perfettoWindow ||
        evt.data !== 'PONG'
      ) {
        return;
      }
      this.cleanupPing();
      const shareUrl = new URL(
        `${this.tool}/${encodeURIComponent(this.sessionId)}${window.location.search}`,
        document.baseURI,
      );
      const perfettoMessage: Record<string, unknown> = {
        'buffer': trace,
        'title': `xprof session ${this.sessionId}`,
        'url': shareUrl.toString(),
        'keepApiOpen': true,
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
    }, MegascalePerfetto.PING_INTERVAL_MS);
    this.pingTimeoutTimer = window.setTimeout(() => {
      this.cleanupPing();
      this.store.dispatch(
        setErrorMessageStateAction({
          errorMessage:
            'Timed out waiting for Perfetto UI to respond. Please reload the page.',
        }),
      );
    }, MegascalePerfetto.PING_TIMEOUT_MS);
  }

  private clearTrace() {
    this.resetLoadedState();
    this.perfettoUrl = 'about:blank';
  }

  private resetLoadedState() {
    this.activeLoadId++;
    this.abortInFlightFetch();
    this.cleanupPing();
    this.throbber.stop();
    setLoadingState(false, this.store);
    this.lastLoadedDataUrl = '';
    this.cachedTraceBuffer = null;
  }

  private abortInFlightFetch() {
    if (this.fetchAbortController) {
      this.fetchAbortController.abort();
      this.fetchAbortController = null;
    }
    this.inFlightDataUrl = '';
    this.isLoading = false;
  }

  private cleanupPing() {
    if (this.pingTimer !== null) {
      window.clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    if (this.pingTimeoutTimer !== null) {
      window.clearTimeout(this.pingTimeoutTimer);
      this.pingTimeoutTimer = null;
    }
    if (this.messageHandler) {
      window.removeEventListener('message', this.messageHandler);
      this.messageHandler = null;
    }
  }

  ngOnDestroy() {
    this.activeLoadId++;
    this.abortInFlightFetch();
    this.cleanupPing();
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
