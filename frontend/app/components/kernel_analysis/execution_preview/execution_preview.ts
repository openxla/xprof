import {ChangeDetectionStrategy, Component, input, output} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatSelectModule} from '@angular/material/select';
import {
  DEFAULT_SAMPLING_COUNTER_SIZE_BITS,
  DEFAULT_SAMPLING_INTERVAL_US,
  DEFAULT_SAMPLING_SCALING,
  type PeriodicCounterSamplingOptions,
} from '../data/data_sampling_info';
import type {TpuGeneration} from '../data/data_tpu_generations';

/** Component for previewing the execution command for kernel analysis.
 *  This is the last step in the kernel analysis workflow.
 */
@Component({
  selector: 'app-execution-preview',
  imports: [
    MatButtonModule,
    MatIconModule,
    MatSelectModule,
    MatFormFieldModule,
    MatInputModule,
  ],
  templateUrl: './execution_preview.ng.html',
  styleUrls: ['./execution_preview.scss'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ExecutionPreviewComponent {
  readonly estimatedPasses = input<number>(1);
  readonly path = input<string>('');
  readonly deviceName = input<TpuGeneration | null>(null);

  readonly tcSampling = input<PeriodicCounterSamplingOptions | null>(null);
  readonly scsSampling = input<PeriodicCounterSamplingOptions | null>(null);
  readonly sctcSampling = input<PeriodicCounterSamplingOptions | null>(null);
  readonly sctdSampling = input<PeriodicCounterSamplingOptions | null>(null);
  readonly cmnSampling = input<PeriodicCounterSamplingOptions | null>(null);
  readonly icrSampling = input<PeriodicCounterSamplingOptions | null>(null);

  readonly pathChange = output<string>();
  readonly expandedToggle = output<void>();

  copyState: 'Copy' | 'Copied!' = 'Copy';

  private isSamplingInUse(
    options: PeriodicCounterSamplingOptions | null | undefined,
  ): boolean {
    return !!(options && options.indices && options.indices.length > 0);
  }

  private formatSamplingOptions(
    options: PeriodicCounterSamplingOptions | null | undefined,
  ): string {
    if (!options) return '';
    const protoParts: string[] = [];
    if (options.is_external_trigger) {
      protoParts.push('is_external_trigger:true');
    } else {
      const intervalUs = options.interval_us ?? DEFAULT_SAMPLING_INTERVAL_US;
      protoParts.push(`interval_us:${intervalUs}`);
    }
    const scaling = options.scaling ?? DEFAULT_SAMPLING_SCALING;
    const counterSizeBits =
      options.counter_size_bits ?? DEFAULT_SAMPLING_COUNTER_SIZE_BITS;
    protoParts.push(`scaling:${scaling}`);
    protoParts.push(`counter_size_bits:${counterSizeBits}`);
    if (options.indices && options.indices.length > 0) {
      protoParts.push(
        `${options.indices.map((index) => `indices:${index}`).join(' ')}`,
      );
    }
    return protoParts.join(' ');
  }

  private hasAnySampling(): boolean {
    return [
      this.tcSampling(),
      this.scsSampling(),
      this.sctcSampling(),
      this.sctdSampling(),
      this.cmnSampling(),
      this.icrSampling(),
    ].some((s) => this.isSamplingInUse(s));
  }

  private generateOptions(): string {
    const samplingConfig: Array<
      [string, PeriodicCounterSamplingOptions | null | undefined]
    > = [
      ['tpu_tc_perf_counter_sampling_options', this.tcSampling()],
      ['tpu_scs_perf_counter_sampling_options', this.scsSampling()],
      ['tpu_sctc_perf_counter_sampling_options', this.sctcSampling()],
      ['tpu_sctd_perf_counter_sampling_options', this.sctdSampling()],
      ['tpu_cmn_perf_counter_sampling_options', this.cmnSampling()],
      ['tpu_icr_perf_counter_sampling_options', this.icrSampling()],
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
    try {
      await navigator.clipboard.writeText(command);
      this.copyState = 'Copied!';
      setTimeout(() => {
        this.copyState = 'Copy';
      }, 2000);
    } catch (err) {
      console.error('Failed to copy command:', err);
    }
  }
}
