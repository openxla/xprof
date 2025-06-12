import {Component, inject, Input, OnChanges, OnDestroy, OnInit, SimpleChanges} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Address, Content, SOURCE_CODE_SERVICE_INTERFACE_TOKEN, SourceCodeServiceInterface} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {Subject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/**
 * A component to display a snippet of source code corresponding to a given
 * stack frame address.
 */
@Component({
  standalone: false,
  selector: 'stack-frame-snippet',
  templateUrl: './stack_frame_snippet.ng.html',
  styleUrls: ['./stack_frame_snippet.scss'],
})
export class StackFrameSnippet implements OnChanges, OnInit, OnDestroy {
  private readonly route: ActivatedRoute = inject(ActivatedRoute);
  private readonly sourceCodeService: SourceCodeServiceInterface =
      inject(SOURCE_CODE_SERVICE_INTERFACE_TOKEN);
  private readonly destroy$ = new Subject<void>();
  private sessionId: string|undefined = undefined;
  frame: Content|undefined = undefined;
  failure: string|undefined = undefined;
  codeSearchLink: string|undefined = undefined;
  codeSearchLinkTooltip: string|undefined = undefined;
  @Input() address: Address|undefined = undefined;

  ngOnInit() {
    this.route.params.pipe(takeUntil(this.destroy$)).subscribe((params) => {
      this.sessionId = (params || {})['sessionId'];
      this.reload();
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['address']) {
      this.reload();
    }
  }

  loaded() {
    return this.frame !== undefined || this.failure !== undefined;
  }

  private reload() {
    this.frame = undefined;
    this.failure = undefined;
    this.codeSearchLink = undefined;
    if (!this.sessionId || !this.address) {
      return;
    }
    this.sourceCodeService.loadContent(this.sessionId, this.address)
        .then((frame) => {
          this.frame = frame;
          this.codeSearchLinkTooltip = 'Open in Code Search';
        })
        .catch((err) => {
          this.codeSearchLinkTooltip =
              'Try Opening in Code Search (might fail)';
          if (err === null) {
            this.failure = 'Unknown Error';
          } else if ('error' in err && typeof err.error === 'string') {
            this.failure = err.error;
          } else if ('message' in err && typeof err.message === 'string') {
            this.failure = err.message;
          } else {
            this.failure = 'Unknown Error';
          }
        });
    this.sourceCodeService
        .codeSearchLink(
            this.sessionId, this.address.fileName, this.address.lineNumber)
        .then((link) => {
          this.codeSearchLink = link;
        })
        .catch((err) => {
          console.log('Failed to get code search link', err);
        });
  }
}
