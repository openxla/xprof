import {Injectable} from '@angular/core';
import {Observable, throwError} from 'rxjs';

import {Address, Content, SourceCodeServiceInterface} from './source_code_service_interface';

/**
 * A service for loading source code for Xprof.
 *
 * This service is responsible for loading source code around a stack frame.
 */
@Injectable()
export class SourceCodeService implements SourceCodeServiceInterface {
  loadContent(sessionId: string, addr: Address): Observable<Content> {
    return throwError(() => new Error('Not implemented'));
  }

  codeSearchLink(sessionId: string, fileName: string, lineNumber: number):
      Observable<string> {
    return throwError(() => new Error('Not implemented'));
  }

  isAvailable(): boolean {
    return false;
  }
}
