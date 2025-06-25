import {Component, inject, Injector, OnDestroy, OnInit} from '@angular/core';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';

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
export class StackTracePage implements OnInit, OnDestroy {
  private readonly injector = inject(Injector);
  // LINT.IfChange(stack_trace_key)
  private readonly stackTraceKey = 'stack_trace';
  // LINT.ThenChange(//depot/org_xprof/plugin/trace_viewer/tf_trace_viewer/tf-trace-viewer.html:stack_trace_key)

  stackTrace = '';
  sourceCodeServiceIsAvailable = false;

  ngOnInit() {
    // We don't need the source code service to be persistently available.
    // We temporarily use the service to check if it is available and show
    // UI accordingly.
    const sourceCodeService = this.injector.get(
        SOURCE_CODE_SERVICE_INTERFACE_TOKEN,
        null,
    );
    this.sourceCodeServiceIsAvailable =
        sourceCodeService?.isAvailable() === true;

    this.stackTrace = window.sessionStorage.getItem(this.stackTraceKey) || '';
  }

  ngOnDestroy(): void {
    window.sessionStorage.removeItem(this.stackTraceKey);
  }
}
