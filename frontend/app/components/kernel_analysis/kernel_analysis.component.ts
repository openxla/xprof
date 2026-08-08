import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  DestroyRef,
  inject,
  ViewEncapsulation,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {
  FormBuilder,
  FormGroup,
  ReactiveFormsModule,
  Validators,
} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatStepperModule} from '@angular/material/stepper';

import {
  COMPONENT_TO_GROUP_NAME,
  ComponentType,
} from './data/data_component_keywords';
import {
  DEFAULT_SAMPLING_COUNTER_SIZE_BITS,
  DEFAULT_SAMPLING_INTERVAL_US,
  DEFAULT_SAMPLING_IS_EXTERNAL_TRIGGER,
  DEFAULT_SAMPLING_SCALING,
  PeriodicCounterSamplingOptions,
} from './data/data_sampling_info';

/**
 * Interface for the kernel form value. This is typing representing the form for each sampling component.
 */
export declare interface KernelFormValue {
  path: string;
  device_name: TpuGeneration | null;
  tc_sampling: PeriodicCounterSamplingOptions;
  scs_sampling: PeriodicCounterSamplingOptions;
  sctc_sampling: PeriodicCounterSamplingOptions;
  sctd_sampling: PeriodicCounterSamplingOptions;
  cmn_sampling: PeriodicCounterSamplingOptions;
  icr_sampling: PeriodicCounterSamplingOptions;
}

import {TPU_GENERATIONS, TpuGeneration} from './data/data_tpu_generations';
import {ExecutionPreviewComponent} from './execution_preview/execution_preview.component';
import {ProfilerOptionsComponent} from './profiler_options/profiler_options.component';

function createSamplingGroup(fb: FormBuilder) {
  return fb.group({
    'interval_us': [DEFAULT_SAMPLING_INTERVAL_US],
    'is_external_trigger': [DEFAULT_SAMPLING_IS_EXTERNAL_TRIGGER],
    'scaling': [
      DEFAULT_SAMPLING_SCALING,
      [Validators.min(0), Validators.max(63)],
    ],
    'counter_size_bits': [DEFAULT_SAMPLING_COUNTER_SIZE_BITS],
    'indices': fb.control<number[]>([]),
  });
}

/**
 * The main component for kernel analysis.
 *
 * This component encapsulates the "Capture Kernel" button. It ties together the profiler options and execution preview
 * components, allowing the user to configure profiling settings and preview
 * the generated command before execution.
 */
@Component({
  selector: 'app-kernel-analysis',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatButtonModule,
    MatStepperModule,
    ProfilerOptionsComponent,
    ExecutionPreviewComponent,
  ],
  templateUrl: './kernel_analysis.component.html',
  styleUrls: ['./kernel_analysis.component.css'],
  encapsulation: ViewEncapsulation.None,
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class KernelAnalysisComponent {
  kernelForm!: FormGroup;
  kernelPasses = 1;

  get formValue(): KernelFormValue {
    return this.kernelForm.value;
  }

  private readonly fb = inject(FormBuilder);
  private readonly cdr = inject(ChangeDetectorRef);

  private readonly destroyRef = inject(DestroyRef);

  constructor() {
    this.kernelForm = this.fb.nonNullable.group({
      'path': [''],
      'device_name': this.fb.control<TpuGeneration | null>(
        TPU_GENERATIONS[0],
        Validators.required,
      ),

      [COMPONENT_TO_GROUP_NAME[ComponentType.TC]]: createSamplingGroup(this.fb),
      [COMPONENT_TO_GROUP_NAME[ComponentType.SCS]]: createSamplingGroup(
        this.fb,
      ),
      [COMPONENT_TO_GROUP_NAME[ComponentType.SCTC]]: createSamplingGroup(
        this.fb,
      ),
      [COMPONENT_TO_GROUP_NAME[ComponentType.SCTD]]: createSamplingGroup(
        this.fb,
      ),
      [COMPONENT_TO_GROUP_NAME[ComponentType.CMN]]: createSamplingGroup(
        this.fb,
      ),
      [COMPONENT_TO_GROUP_NAME[ComponentType.ICR]]: createSamplingGroup(
        this.fb,
      ),
    });

    const samplingGroups = Object.values(COMPONENT_TO_GROUP_NAME);

    // Disable/Enable interval_us based on is_external_trigger
    for (const groupName of samplingGroups) {
      const group = this.kernelForm.get(groupName) as FormGroup;
      group.get('is_external_trigger')?.valueChanges.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((isExternal) => {
        const intervalCtrl = group.get('interval_us');
        if (isExternal) {
          intervalCtrl?.disable();
        } else {
          intervalCtrl?.enable();
        }
      });
    }

    this.kernelForm.get('device_name')?.valueChanges.pipe(takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      for (const groupName of samplingGroups) {
        const group = this.kernelForm.get(groupName) as FormGroup;
        group.get('indices')?.setValue([]);
      }
    });

    this.kernelForm.valueChanges.pipe(takeUntilDestroyed(this.destroyRef)).subscribe(() => {
      this.cdr.markForCheck();
    });
  }
}
