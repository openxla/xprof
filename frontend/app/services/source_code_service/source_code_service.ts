import {Injectable} from '@angular/core';
import {Store} from '@ngrx/store';
import {ProfilerConfig} from 'org_xprof/frontend/app/common/interfaces/capture_profile';
import {getProfilerConfig} from 'org_xprof/frontend/app/store/selectors';
import {EMPTY, Observable, of, throwError} from 'rxjs';
import {filter, map, switchMap, take} from 'rxjs/operators';

import {
  Address,
  Content,
  SourceCodeServiceInterface,
} from './source_code_service_interface';

const CODE_SEARCH_HOSTS: ReadonlySet<string> = new Set([
  'cs.opensource.google',
]);

/**
 * Formats a source link URL from an HTTP(S) prefix, file path, and optional
 * line number, normalizing slashes only at the join boundary. Returns
 * `undefined` for empty or non-HTTP(S) prefixes.
 */
function buildSourceUrl(
  pathPrefix: string,
  fileName: string,
  lineNumber: number,
): string | undefined {
  const trimmedPrefix = pathPrefix.trim();
  if (!trimmedPrefix) {
    return undefined;
  }
  let parsedUrl: URL;
  try {
    parsedUrl = new URL(trimmedPrefix);
  } catch {
    return undefined;
  }
  if (parsedUrl.protocol !== 'http:' && parsedUrl.protocol !== 'https:') {
    return undefined;
  }
  const base = trimmedPrefix.replace(/\/+$/, '');
  const cleanFile = fileName.replace(/^\/+/, '');
  if (!cleanFile) {
    return undefined;
  }
  let url = `${base}/${cleanFile}`;
  if (lineNumber > 0 && Number.isFinite(lineNumber)) {
    const line = Math.floor(lineNumber);
    const hostname = parsedUrl.hostname.toLowerCase();
    url += CODE_SEARCH_HOSTS.has(hostname) ? `;l=${line}` : `#L${line}`;
  }
  return url;
}

/**
 * A service for loading source code for Xprof.
 *
 * This service is responsible for loading source code around a stack frame.
 */
@Injectable()
export class SourceCodeService implements SourceCodeServiceInterface {
  constructor(private readonly store: Store) {}

  loadContent(sessionId: string, addr: Address): Observable<Content> {
    return throwError(() => new Error('Not implemented'));
  }

  codeSearchLink(
    sessionId: string,
    fileName: string,
    lineNumber: number,
    pathPrefix = '',
  ): Observable<string> {
    if (!fileName) {
      return throwError(() => new Error('File name is empty'));
    }
    if (pathPrefix.trim()) {
      const url = buildSourceUrl(pathPrefix, fileName, lineNumber);
      return url ? of(url) : EMPTY;
    }
    return this.store.select(getProfilerConfig).pipe(
      filter(
        (config): config is ProfilerConfig =>
          config !== null && config !== undefined,
      ),
      take(1),
      switchMap((config) => {
        const url = buildSourceUrl(
          config.srcPathPrefix ?? '',
          fileName,
          lineNumber,
        );
        return url ? of(url) : EMPTY;
      }),
    );
  }

  isAvailable(): Observable<boolean> {
    return this.store.select(getProfilerConfig)
        .pipe(
            map((config: ProfilerConfig) => !!config?.srcPathPrefix),
        );
  }

  isCodeFetchEnabled(): boolean {
    return false;
  }
}
