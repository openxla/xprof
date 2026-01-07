import {Injectable} from '@angular/core';
import {Store} from '@ngrx/store';
import {ProfilerConfig} from 'org_xprof/frontend/app/common/interfaces/capture_profile';
import {getProfilerConfig} from 'org_xprof/frontend/app/store/selectors';
import {Observable, of, throwError} from 'rxjs';

import {Address, Content, SourceCodeServiceInterface} from './source_code_service_interface';

/**
 * A service for loading source code for Xprof.
 *
 * This service is responsible for loading source code around a stack frame.
 */
@Injectable()
export class SourceCodeService implements SourceCodeServiceInterface {
  private srcPathPrefix = '';

  constructor(private readonly store: Store) {
    this.store.select(getProfilerConfig).subscribe((config: ProfilerConfig) => {
      if (config?.srcPathPrefix) {
        this.srcPathPrefix = config.srcPathPrefix;
      }
    });
  }

  loadContent(sessionId: string, addr: Address): Observable<Content> {
    return throwError(() => new Error('Not implemented'));
  }

  codeSearchLink(
      sessionId: string, fileName: string, lineNumber: number,
      pathPrefix = ''): Observable<string> {
    if (!fileName) {
      return throwError(() => new Error('File name is empty'));
    }
    if (!pathPrefix) {
      return throwError(() => new Error('Path prefix is empty'));
    }
    return of(`${pathPrefix}/${fileName}`);
  }

  isAvailable(): boolean {
    return !!this.srcPathPrefix;
  }

  isCodeFetchEnabled(): boolean {
    return false;
  }
}
