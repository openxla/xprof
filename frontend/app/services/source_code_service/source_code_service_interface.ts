/**
 * @fileoverview Interfaces for the source code service.
 */

import {InjectionToken} from '@angular/core';
import {Observable} from 'rxjs';

/**
 * Address of a source code line and a few lines around it.
 *
 * Users use this to request a specific code snippet around a stack frame.
 */
export class Address {
  /**
   * @throws {Error} If the input parameters are invalid.
   */
  constructor(
      readonly fileName: string, readonly lineNumber: number,
      readonly linesBefore: number, readonly linesAfter: number) {
    if (lineNumber <= 0) {
      throw new Error(`lineNumber (${lineNumber}) must be > 0`);
    }
    if (linesBefore < 0) {
      throw new Error(`linesBefore (${linesBefore}) must be >= 0`);
    }
    if (linesAfter < 0) {
      throw new Error(`linesAfter (${linesAfter}) must be >= 0`);
    }
  }
}

/**
 * Content of a source code around a stack frame.
 *
 * @param address An address of the source code.
 * @param firstLine The first line of the source code. In general, this could
 *     be very different from `address.lineNumber - address.linesBefore`.
 * @param lines The lines of the source code snippet. There must be at least
 *     `address.lineNumber - firstLine + 1` lines in the array.
 *
 * @throws {Error} If the input `lines` does not contain the line requested in
 *     the address.
 */
export class Content {
  constructor(
      readonly address: Address, readonly firstLine: number,
      readonly lines: readonly string[]) {
    if (firstLine > address.lineNumber) {
      throw new Error(
          `The first line number ${firstLine} must be less than or equal to ` +
          `the requested line number ${address.lineNumber}`);
    }
    if (lines.length <= address.lineNumber - firstLine) {
      throw new Error(
          `The input 'lines' has only ${lines.length} lines. Since the first ` +
          `line number is ${firstLine}, the input lines does not contain the ` +
          `requested line number in the address ${address.lineNumber}.`);
    }
  }
}

/**
 * Interface for the source code service.
 */
export interface SourceCodeServiceInterface {
  /**
   * Loads content of a source code around a stack frame.
   *
   * Implementation is expected to use cache to (temporarily) store loaded
   * contents. UI uses this interface many times to load source code around
   * different stack frames.
   *
   * @param sessionId XProf session ID.
   * @param address An address of the source code.
   * @return The content of the source code.
   */
  loadContent(sessionId: string, address: Address): Observable<Content>;

  /**
   * Returns a link to the code search for the given address.
   *
   * The return value is a promise because the implementation might need to
   * send a request to the server to get the change list number.
   *
   * @param sessionId XProf session ID.
   * @param fileName The file name.
   * @param lineNumber The line number.
   * @return A link to the code search.
   */
  codeSearchLink(sessionId: string, fileName: string, lineNumber: number):
      Observable<string>;

  /**
   * Returns true if the source code service is available.
   *
   * A service might have different implementations for different environments
   * and not all implementations become available at the same time. Yet, the UI
   * looks differently when they assume the service is available. This method
   * allows the UI to check if the service is available and show a proper
   * content accordingly.
   */
  isAvailable(): boolean;
}

/** Injection token for the source code service interface. */
export const SOURCE_CODE_SERVICE_INTERFACE_TOKEN =
    new InjectionToken<SourceCodeServiceInterface>(
        'SourceCodeServiceInterface',
    );
