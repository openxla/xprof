import {Component, inject, Injector, OnDestroy} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {combineLatest, ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/**
 * A stack trace page component.
 *
 * We use this component to show snippets of source code around the stack trace
 * in a new page. This is particularly useful for Trace Viewer, since it is not
 * written in Angular, but `StackTraceSnippet` is written in Angular. This
 * component provides a bridge between the two.
 *
 * Note: we keep this component named as StackTracePage, even though it now
 * contains source viewer info including both stack trace source code and
 * various IR text mapping.
 */
@Component({
  standalone: false,
  selector: 'stack-trace-page',
  templateUrl: './stack_trace_page.ng.html',
  styleUrls: ['./stack_trace_page.css'],
})
export class StackTracePage implements OnDestroy {
  private readonly injector = inject(Injector);
  private readonly route = inject(ActivatedRoute);
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly hloModuleKey = 'hlo_module';
  private readonly hloOpKey = 'hlo_op';
  private readonly sourceKey = 'source';
  private readonly stackTraceKey = 'stack_trace';
  private readonly sessionIdKey = 'session_id';
  private readonly opCategoryKey = 'op_category';

  hloModule = '';
  hloOp = '';
  sourceFileAndLineNumber = '';
  stackTrace = '';
  sourceCodeServiceIsAvailable = false;
  sessionId = '';
  programId = '';
  opCategory = '';

  constructor() {
    // We don't need the source code service to be persistently available.
    // We temporarily use the service to check if it is available and show
    // UI accordingly.
    const sourceCodeService = this.injector.get(
        SOURCE_CODE_SERVICE_INTERFACE_TOKEN,
        null,
    );
    this.sourceCodeServiceIsAvailable =
        sourceCodeService?.isAvailable() === true;
    combineLatest([this.route.params, this.route.queryParams])
        .pipe(takeUntil(this.destroyed))
        .subscribe(([params, queryParams]) => {
          this.sessionId = params['sessionId'] || this.sessionId;
          this.processQueryParams(queryParams);
        });
  }

  processQueryParams(params: Params) {
    this.hloModule = params[this.hloModuleKey] || '';
    this.hloOp = params[this.hloOpKey] || '';
    this.sourceFileAndLineNumber = params[this.sourceKey] || '';
    this.stackTrace = params[this.stackTraceKey] || '';
    const regex = /\((.*?)\)/;
    const programIdMatch = this.hloModule.match(regex);
    this.programId = programIdMatch ? programIdMatch[1] : '';
    this.sessionId = params[this.sessionIdKey] || params['run'] || '';
    this.opCategory = params[this.opCategoryKey] || '';
  }

  ngOnDestroy(): void {
    this.destroyed.next();
    this.destroyed.complete();
  }
}
