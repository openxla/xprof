/**
 * @fileoverview Base service for diffing tool data.
 */

import {inject, Injectable} from '@angular/core';
import {DataTable} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {forkJoin, Observable, of} from 'rxjs';
import {catchError, map} from 'rxjs/operators';

/**
 * Base service for diffing tool data.
 */
@Injectable()
export class BaseDiffService {
  protected readonly dataService = inject(DATA_SERVICE_INTERFACE_TOKEN);
  baseSessionId: string = this.dataService.getBaseSessionId() || '';

  getBaseSessionId(): string {
    return this.baseSessionId;
  }

  setBaseSessionId(sessionId: string) {
    this.baseSessionId = sessionId;
    this.dataService.setBaseSessionId(sessionId);
  }

  /**
   * Fetches data for active and baseline sessions.
   */
  getDiffData(
    activeSessionId: string,
    baselineSessionId: string = this.baseSessionId,
    tool: string,
    host = '',
    parameters: Map<string, string | boolean> = new Map(),
  ): Observable<{active: DataTable | null; baseline: DataTable | null}> {
    const active$ = this.dataService
      .getData(activeSessionId, tool, host, parameters)
      .pipe(
        catchError((error) => {
          console.error(
            `Error fetching active session data for ${tool}:`,
            error,
          );
          return of(null);
        }),
      );

    if (!baselineSessionId) {
      return active$.pipe(
        map((active) => ({active: active as DataTable, baseline: null})),
      );
    }

    const baseline$ = this.dataService
      .getData(baselineSessionId, tool, host, parameters, {
        updateSearchParams: false,
      })
      .pipe(
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
    }) as Observable<{active: DataTable | null; baseline: DataTable | null}>;
  }
}
