/**
 * @fileoverview Base service for diffing tool data.
 */

import {inject, Injectable} from '@angular/core';
import {DataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {forkJoin, Observable, of} from 'rxjs';
import {catchError, map} from 'rxjs/operators';

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

  get baseSessionId(): string {
    return this.dataService.getBaseSessionId() || '';
  }

  getBaseSessionId(): string {
    return this.baseSessionId;
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
      baselineSessionId = this.baseSessionId,
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

    const baseline$ = this.dataService
      .getData(baselineSessionId, tool, host, parameters, {
        updateSearchParams: false,
      })
      .pipe(
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
}
