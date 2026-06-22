import {HttpClient} from '@angular/common/http';
import {inject, Injectable} from '@angular/core';
import {Observable, of} from 'rxjs';
import {catchError, shareReplay, tap} from 'rxjs/operators';
import {CuratedTool} from './curated_tool';

/**
 * Service responsible for fetching XProf Labs curated tools configuration
 * from the backend API.
 */
@Injectable({providedIn: 'root'})
export class CuratedToolsService {
  private readonly http = inject(HttpClient);

  private configCache$: Observable<readonly CuratedTool[]> | null = null;

  /**
   * Retrieves the catalog of curated tools, caching the stream in memory
   * across component initializations.
   */
  getCuratedTools(): Observable<readonly CuratedTool[]> {
    if (this.configCache$) {
      return this.configCache$;
    }
    this.configCache$ = this.http
      .get<readonly CuratedTool[]>('/api/curated_tools')
      .pipe(
        shareReplay(1),
        tap({
          error: (error: unknown) => {
            console.error('Failed to fetch curated tools config:', error);
          },
        }),
        catchError(() => {
          this.configCache$ = null;
          return of([]);
        }),
      );
    return this.configCache$;
  }
}
