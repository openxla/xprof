import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  EventEmitter,
  inject,
  Input,
  OnDestroy,
  Output,
} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatSelectModule} from '@angular/material/select';
import {parseFiniteNumber} from 'org_xprof/frontend/app/common/utils/utils';
import {
  DEFAULT_SAMPLING_COUNTER_SIZE_BITS,
  DEFAULT_SAMPLING_INTERVAL_US,
  DEFAULT_SAMPLING_SCALING,
  type PeriodicCounterSamplingOptions,
} from '../data/data_sampling_info';
import type {TpuGeneration} from '../data/data_tpu_generations';

const MAX_INT32 = 2147483647;

/** Component for previewing the execution command for kernel analysis.
 *  This is the last step in the kernel analysis workflow.
 */
@Component({
  selector: 'app-execution-preview',
  standalone: true,
  imports: [
    CommonModule,
    MatButtonModule,
    MatIconModule,
    MatSelectModule,
    MatFormFieldModule,
    MatInputModule,
  ],
  templateUrl: './execution_preview.component.html',
  styleUrls: ['./execution_preview.component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ExecutionPreviewComponent implements OnDestroy {
  private readonly cdr = inject(ChangeDetectorRef);
  private copyResetTimer?: ReturnType<typeof setTimeout>;
  private isDestroyed = false;

  @Input() estimatedPasses = 1;
  @Input() path = '';
  @Input() deviceName: TpuGeneration | null = null;

  @Input() tcSampling: PeriodicCounterSamplingOptions | null = null;
  @Input() scsSampling: PeriodicCounterSamplingOptions | null = null;
  @Input() sctcSampling: PeriodicCounterSamplingOptions | null = null;
  @Input() sctdSampling: PeriodicCounterSamplingOptions | null = null;
  @Input() cmnSampling: PeriodicCounterSamplingOptions | null = null;
  @Input() icrSampling: PeriodicCounterSamplingOptions | null = null;

  @Output() readonly pathChange = new EventEmitter<string>();
  @Output() readonly expandedToggle = new EventEmitter<void>();

  copyState: 'Copy' | 'Copied!' | 'Copy failed' = 'Copy';

  get copyIcon(): string {
    if (this.copyState === 'Copied!') {
      return 'check';
    }
    if (this.copyState === 'Copy failed') {
      return 'error_outline';
    }
    return 'content_copy';
  }

  private getValidIndices(
    options: PeriodicCounterSamplingOptions | null | undefined,
  ): number[] {
    if (!options?.indices) {
      return [];
    }
    return options.indices.filter(
      (index) =>
        typeof index === 'number' && Number.isInteger(index) && index >= 0,
    );
  }

  private isSamplingInUse(
    options: PeriodicCounterSamplingOptions | null | undefined,
  ): boolean {
    return this.getValidIndices(options).length > 0;
  }

  private formatSamplingOptions(
    options: PeriodicCounterSamplingOptions | null | undefined,
  ): string {
    if (!options) return '';
    const protoParts: string[] = [];
    if (options.is_external_trigger) {
      protoParts.push('is_external_trigger:true');
    } else {
      const intervalUs = Math.min(
        MAX_INT32,
        Math.max(
          0,
          Math.trunc(
            parseFiniteNumber(
              options.interval_us,
              DEFAULT_SAMPLING_INTERVAL_US,
            ),
          ),
        ),
      );
      protoParts.push(`interval_us:${intervalUs}`);
    }
    const scaling = Math.min(
      63,
      Math.max(
        0,
        Math.trunc(
          parseFiniteNumber(options.scaling, DEFAULT_SAMPLING_SCALING),
        ),
      ),
    );
    const counterSizeBits =
      options.counter_size_bits ?? DEFAULT_SAMPLING_COUNTER_SIZE_BITS;
    protoParts.push(`scaling:${scaling}`);
    protoParts.push(`counter_size_bits:${counterSizeBits}`);
    const validIndices = this.getValidIndices(options);
    if (validIndices.length > 0) {
      protoParts.push(
        `${validIndices.map((index) => `indices:${index}`).join(' ')}`,
      );
    }
    return protoParts.join(' ');
  }

  private hasAnySampling(): boolean {
    return [
      this.tcSampling,
      this.scsSampling,
      this.sctcSampling,
      this.sctdSampling,
      this.cmnSampling,
      this.icrSampling,
    ].some((s) => this.isSamplingInUse(s));
  }

  private generateOptions(): string {
    const samplingConfig: Array<
      [string, PeriodicCounterSamplingOptions | null | undefined]
    > = [
      ['tpu_tc_perf_counter_sampling_options', this.tcSampling],
      ['tpu_scs_perf_counter_sampling_options', this.scsSampling],
      ['tpu_sctc_perf_counter_sampling_options', this.sctcSampling],
      ['tpu_sctd_perf_counter_sampling_options', this.sctdSampling],
      ['tpu_cmn_perf_counter_sampling_options', this.cmnSampling],
      ['tpu_icr_perf_counter_sampling_options', this.icrSampling],
    ];

    const configParts: string[] = [];
    if (this.hasAnySampling()) {
      configParts.push(`    "tpu_enable_periodic_counter_sampling" : True`);
    }

    for (const [key, sampling] of samplingConfig) {
      if (this.isSamplingInUse(sampling)) {
        const formatted = this.formatSamplingOptions(sampling);
        configParts.push(`    "${key}" : (\n        '${formatted}'\n    )`);
      }
    }

    const configStr =
      configParts.length > 0 ? `{\n${configParts.join(',\n')},\n}` : '{}';

    return `options = jax.profiler.ProfileOptions()
options.advanced_configuration = ${configStr}`;
  }

  get generatedCommand(): string {
    return this.generateOptions();
  }

  async copyCommand() {
    const command = this.generatedCommand;
    if (!command) return;

    let copied = false;
    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(command);
        copied = true;
      } catch (err) {
        console.error('Failed to copy command via Clipboard API:', err);
      }
    }

    if (this.isDestroyed) {
      return;
    }

    if (!copied) {
      copied = this.copyCommandFallback(command);
    }

    this.setTransientCopyState(copied ? 'Copied!' : 'Copy failed');
  }

  private copyCommandFallback(command: string): boolean {
    const textarea = document.createElement('textarea');
    textarea.value = command;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.top = '-9999px';
    textarea.style.left = '-9999px';
    textarea.style.opacity = '0';
    const activeElement = document.activeElement;
    document.body.appendChild(textarea);
    try {
      textarea.focus();
      textarea.select();
      textarea.setSelectionRange(0, command.length);
      // execCommand('copy') is deprecated, but is the only fallback when
      // navigator.clipboard is unavailable (plain HTTP or restricted iframes).
      // tslint:disable-next-line:deprecation
      return document.execCommand('copy');
    } catch (err) {
      console.error('Failed to copy command via execCommand:', err);
      return false;
    } finally {
      document.body.removeChild(textarea);
      if (activeElement instanceof HTMLElement) {
        activeElement.focus();
      }
    }
  }

  private setTransientCopyState(state: 'Copied!' | 'Copy failed') {
    this.copyState = state;
    this.cdr.markForCheck();
    if (this.copyResetTimer !== undefined) {
      clearTimeout(this.copyResetTimer);
    }
    this.copyResetTimer = setTimeout(() => {
      this.copyState = 'Copy';
      this.copyResetTimer = undefined;
      this.cdr.markForCheck();
    }, 2000);
  }

  ngOnDestroy() {
    this.isDestroyed = true;
    if (this.copyResetTimer !== undefined) {
      clearTimeout(this.copyResetTimer);
      this.copyResetTimer = undefined;
    }
  }
}
