/**
 * @fileoverview Base service for diffing tool data.
 */

import {inject, Injectable} from '@angular/core';
import {ALL_HOSTS} from 'org_xprof/frontend/app/common/constants/constants';
import {DataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {forkJoin, Observable, of} from 'rxjs';
import {catchError, map, switchMap} from 'rxjs/operators';

/** Options for fetching diff data. */
interface DiffOptions {
  /**
   * The baseline session ID to compare against. If not provided, the baseline
   * session ID from the data service will be used.
   */
  baselineSessionId?: string;
  host?: string;
  parameters?: Map<string, string | boolean>;
}

/**
 * Base service for diffing tool data.
 */
@Injectable({providedIn: 'root'})
export class BaseDiffService {
  protected readonly dataService = inject(DATA_SERVICE_INTERFACE_TOKEN);

  getBaseSessionId(): string {
    return this.dataService.getBaseSessionId() || '';
  }

  setBaseSessionId(sessionId: string) {
    this.dataService.setBaseSessionId(sessionId);
  }

  /**
   * Fetches data for active and baseline sessions.
   */
  getDiffData<T = DataTable | DataTable[]>(
    activeSessionId: string,
    tool: string,
    diffOptions: DiffOptions = {},
  ): Observable<{active: T | null; baseline: T | null}> {
    const {
      baselineSessionId = this.getBaseSessionId() || '',
      host = '',
      parameters = new Map(),
    } = diffOptions;
    const active$ = this.dataService
      .getData(activeSessionId, tool, host, parameters)
      .pipe(
        map((data) => data as T | null),
        catchError((error) => {
          console.error(
            `Error fetching active session data for ${tool}:`,
            error,
          );
          return of(null);
        }),
      );

    if (!baselineSessionId) {
      return active$.pipe(map((active) => ({active, baseline: null})));
    }

    const baseline$ = this.getBaselineHost(baselineSessionId, tool, host).pipe(
      switchMap((baselineHost) =>
        this.dataService.getData(
          baselineSessionId,
          tool,
          baselineHost,
          parameters,
          {updateSearchParams: false},
        ),
      ),
      map((data) => data as T | null),
      catchError((error) => {
        console.error(
          `Error fetching baseline session data for ${tool}:`,
          error,
        );
        return of(null);
      }),
    );

    return forkJoin({
      active: active$,
      baseline: baseline$,
    });
  }

  /**
   * Returns the host to request for the baseline session.
   *
   * The baseline session can record different host names than the active
   * session, and the backend rejects a host that a session lacks. This keeps
   * `activeHost` when the baseline lists it and otherwise uses the baseline's
   * first host, which is the default host the sidebar selects. It also keeps
   * `activeHost` when the lookup fails or lists no hosts.
   */
  private getBaselineHost(
    baselineSessionId: string,
    tool: string,
    activeHost: string,
  ): Observable<string> {
    if (!activeHost || activeHost === ALL_HOSTS) {
      return of(activeHost);
    }
    return this.dataService.getHosts(baselineSessionId, tool).pipe(
      map((hosts) => {
        // Some data service implementations emit null when the lookup fails.
        const hostnames = (hosts ?? [])
          .map((host) => host.hostname)
          .filter(Boolean);
        return hostnames.includes(activeHost)
          ? activeHost
          : (hostnames[0] ?? activeHost);
      }),
      catchError((error) => {
        console.error(`Error fetching baseline hosts for ${tool}:`, error);
        return of(activeHost);
      }),
    );
  }
}
