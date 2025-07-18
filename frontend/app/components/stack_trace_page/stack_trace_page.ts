import {Component, inject, Injector, OnDestroy} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/**
 * A stack trace page component.
 *
 * We use this component to show snippets of source code around the stack trace
 * in a new page. This is particularly useful for Trace Viewer, since it is not
 * written in Angular, but `StackTraceSnippet` is written in Angular. This
 * component provides a bridge between the two.
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
  // LINT.IfChange(keys)
  private readonly hloModuleKey = 'hlo_module';
  private readonly hloOpKey = 'hlo_op';
  private readonly stackTraceKey = 'stack_trace';
  // LINT.ThenChange(
  //   //depot/google3/perftools/accelerators/xprof/frontend/app/common/constants/constants.ts:stack_trace_page_keys,
  //   //depot/org_xprof/plugin/trace_viewer/tf_trace_viewer/tf-trace-viewer.html:stack_trace_page_keys)

  hloModule = '';
  hloOp = '';
  stackTrace = '';
  sourceCodeServiceIsAvailable = false;

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

    this.route.queryParams.pipe(takeUntil(this.destroyed))
        .subscribe((params) => {
          this.hloModule = params[this.hloModuleKey] || '';
          this.hloOp = params[this.hloOpKey] || '';
          this.stackTrace = params[this.stackTraceKey] || '';
        });
  }

  ngOnDestroy(): void {
    this.destroyed.next();
    this.destroyed.complete();
  }
}
