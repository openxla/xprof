import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  effect,
  inject,
  input,
  OnDestroy,
  ViewChild,
} from '@angular/core';
import {ActivatedRoute} from '@angular/router';
import {Metric} from 'org_xprof/frontend/app/common/interfaces/source_stats';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {
  Address,
  Content,
  SOURCE_CODE_SERVICE_INTERFACE_TOKEN,
  SourceCodeServiceInterface,
} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {Subject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {NgIf} from '@angular/common';
import {
  MatExpansionPanel,
  MatExpansionPanelHeader,
  MatExpansionPanelTitle,
} from '@angular/material/expansion';
import {MatIcon} from '@angular/material/icon';
import {MatProgressBar} from '@angular/material/progress-bar';
import {MatTooltip} from '@angular/material/tooltip';
import {Message} from './message';
/**
 * A component to display a snippet of source code corresponding to a given
 * stack frame address.
 */
@Component({
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'stack-frame-snippet',
  templateUrl: './stack_frame_snippet.ng.html',
  styleUrls: ['./stack_frame_snippet.scss'],
  imports: [
    MatExpansionPanel,
    MatExpansionPanelHeader,
    MatExpansionPanelTitle,
    MatIcon,
    MatProgressBar,
    MatTooltip,
    Message,
    NgIf,
  ],
})
export class StackFrameSnippet implements OnDestroy {
  readonly sourceCodeSnippetAddress = input<Address>();
  readonly topOfStack = input<boolean>();
  readonly usingSourceFileAndLineNumber = input<boolean>();
  readonly srcPathPrefix = input('');
  private readonly route: ActivatedRoute = inject(ActivatedRoute);
  private readonly sourceCodeService: SourceCodeServiceInterface = inject(
    SOURCE_CODE_SERVICE_INTERFACE_TOKEN,
  );
  private readonly cdr = inject(ChangeDetectorRef);
  private readonly destroy$ = new Subject<void>();
  private sessionId: string | undefined = undefined;
  frame: Content | undefined = undefined;
  failure: string | undefined = undefined;
  codeSearchLink: string | undefined = undefined;
  codeSearchLinkTooltip: string | undefined = undefined;
  lineNumberToMetricMap: Map<number, Metric> | undefined = undefined;
  isCodeFetchEnabled = false;

  constructor() {
    this.route.params.pipe(takeUntil(this.destroy$)).subscribe((params) => {
      this.sessionId =
        (params || {})['sessionId'] || (params || {})['run'] || this.sessionId;
      this.reload();
    });
    this.isCodeFetchEnabled = this.sourceCodeService.isCodeFetchEnabled();

    effect(() => {
      this.sourceCodeSnippetAddress();
      this.srcPathPrefix();
      this.reload();
    });
  }

  ngOnDestroy(): void {
    this.destroy$.next();
    this.destroy$.complete();
  }

  trackByIndex(index: number, item: string): number {
    return index;
  }

  lineMetric(lineNumber: number): Metric | undefined {
    return this.lineNumberToMetricMap?.get(lineNumber);
  }

  get loaded() {
    return (
      this.frame !== undefined ||
      this.failure !== undefined ||
      !this.isCodeFetchEnabled
    );
  }

  get isAtTopOfStack(): boolean {
    // We intentionally treat `undefined` as `false` here. That is if we don't
    // know whether we are at the top of the stack, we assume that we are not.
    return this.topOfStack() ?? false;
  }

  private reload() {
    this.frame = undefined;
    this.failure = undefined;
    this.codeSearchLink = undefined;
    this.lineNumberToMetricMap = undefined;
    const address = this.sourceCodeSnippetAddress();
    if (!this.sessionId || !address) {
      return;
    }
    this.sourceCodeService
      .loadContent(this.sessionId, address)
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (frame) => {
          this.frame = frame;
          this.codeSearchLinkTooltip = 'Open in Code Search';
          this.lineNumberToMetricMap = new Map(
            frame.metrics.map((lineMetric) => [
              lineMetric.lineNumber,
              lineMetric.metric,
            ]),
          );
          this.cdr.markForCheck();
        },
        error: (err) => {
          this.codeSearchLinkTooltip =
            'Try Opening in Code Search (might fail)';
          if (typeof err === 'object' && typeof err?.error === 'string') {
            this.failure = err.error;
          } else if (
            typeof err === 'object' &&
            typeof err?.message === 'string'
          ) {
            this.failure = err.message;
          } else {
            this.failure = 'Unknown Error';
          }
          this.cdr.markForCheck();
        },
      });
    this.sourceCodeService
      .codeSearchLink(
        this.sessionId,
        address.fileName,
        address.lineNumber,
        this.srcPathPrefix(),
      )
      .pipe(takeUntil(this.destroy$))
      .subscribe({
        next: (link) => {
          this.codeSearchLink = link;
          this.cdr.markForCheck();
        },
        error: (err) => {
          console.error('Failed to get code search link', err);
        },
      });
  }

  percent = utils.percent;
  formatDurationPs = utils.formatDurationPs;
}
