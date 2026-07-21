import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  DestroyRef,
  Input,
  OnInit,
  inject,
} from '@angular/core';
import {takeUntilDestroyed} from '@angular/core/rxjs-interop';
import {FormGroup, ReactiveFormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatButtonToggleModule} from '@angular/material/button-toggle';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatDialog, MatDialogModule} from '@angular/material/dialog';
import {MatDividerModule} from '@angular/material/divider';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatRadioModule} from '@angular/material/radio';
import {MatSelectModule} from '@angular/material/select';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  type DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {
  COMPONENT_TO_GROUP_NAME,
  COMPONENT_TO_SEARCH_KEYWORD,
  ComponentType,
} from '../data/data_component_keywords';
import {TPU_GENERATIONS} from '../data/data_tpu_generations';
import {CounterSelectionComponent} from '../shared/components/counter_selection/counter_selection.component';
import {
  Counter,
  CounterSelectionConfig,
} from '../shared/components/counter_selection/types';

/**
 * Component representing the "Profiler Options" step of a kernel analysis sequence.
 * Now customized for Periodic Counter Sampling.
 */
@Component({
  selector: 'app-profiler-options',
  standalone: true,
  imports: [
    CommonModule,
    ReactiveFormsModule,
    MatDialogModule,
    MatRadioModule,
    MatInputModule,
    MatCheckboxModule,
    MatIconModule,
    MatButtonModule,
    MatButtonToggleModule,
    MatSelectModule,
    MatDividerModule,
  ],
  templateUrl: './profiler_options.component.html',
  styleUrls: ['./profiler_options.component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ProfilerOptionsComponent implements OnInit {
  @Input() formGroup?: FormGroup;

  selectedComponent: 'tc' | 'scs' | 'sctc' | 'sctd' | 'cmn' | 'icr' = 'tc';
  tpuGenerations = TPU_GENERATIONS;

  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );
  private readonly dialog = inject(MatDialog);
  private readonly cdr = inject(ChangeDetectorRef);
  allCounters: Counter[] = [];

  private readonly destroyRef = inject(DestroyRef);

  ngOnInit() {
    if (!this.formGroup) {
      throw new Error('formGroup is required');
    }
    this.formGroup
      .get('device_name')
      ?.valueChanges.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((gen: unknown) => {
        const tpuGen = gen as {id: string; name: string} | null;
        if (tpuGen) {
          this.fetchCounters(tpuGen.id);
        }
      });

    const initialGen = this.formGroup.get('device_name')?.value;
    if (initialGen) {
      this.fetchCounters(initialGen.id);
    }
  }

  fetchCounters(deviceType: string) {
    this.dataService
      .getData(
        '',
        'perf_counters',
        '',
        new Map([
          ['names_only', '1'],
          ['device_type', deviceType],
        ]),
      )
      .pipe(takeUntilDestroyed(this.destroyRef))
      .subscribe((data: unknown) => {
        if (Array.isArray(data)) {
          this.allCounters = data as Counter[];
          this.cdr.markForCheck();
        }
      });
  }

  convertValuesToIndices(
    counters: Counter[],
  ): Array<{name: string; index: number}> {
    if (!counters || counters.length === 0) return [];

    // Check if we have 'val'
    if (counters.some((c) => c.val !== undefined)) {
      const vals = counters
        .map((c) => c.val)
        .filter((v) => typeof v === 'number' && !isNaN(v));
      if (vals.length === 0) return [];
      const minVal = Math.min(...vals);
      return counters
        .map((c) => {
          const name = c.name || '';
          const parts = name.split('unprivileged_');
          const shortName = parts[parts.length - 1];
          const val = c.val;
          const index = Math.floor((val - minVal) / 8);
          return {name: shortName, index};
        })
        .sort((a, b) => a.index - b.index);
    }
    return [];
  }

  getCounters(groupName: string): Array<{name: string; index: number}> {
    const component = Object.values(ComponentType).find(
      (c) => COMPONENT_TO_GROUP_NAME[c as ComponentType] === groupName,
    ) as ComponentType | undefined;

    if (component) {
      const keyword = COMPONENT_TO_SEARCH_KEYWORD[component];
      const counters = this.allCounters.filter((c) => c.name.includes(keyword));
      return this.convertValuesToIndices(counters);
    }
    return [];
  }
  isComponentInUse(groupName: string): boolean {
    if (!this.formGroup) {
      return false;
    }
    const group = this.formGroup.get(groupName);
    const indices = group?.get('indices')?.value;
    return Array.isArray(indices) && indices.length > 0;
  }

  getSelectedCountersCount(groupName: string): number {
    if (!this.formGroup) {
      return 0;
    }
    const group = this.formGroup.get(groupName);
    const indices = group?.get('indices')?.value;
    return Array.isArray(indices) ? indices.length : 0;
  }

  openCustomizeDialog(groupName: string) {
    if (!this.formGroup) {
      return;
    }
    const counters = this.getCounters(groupName);
    const config: CounterSelectionConfig = {
      exactMatchForPureNumbers: true,
      groups: [
        {
          name: 'Available Counters',
          counters: counters.map((c) => ({
            id: String(c.index),
            label: c.name,
          })),
          expandByDefault: true,
        },
      ],
    };

    const group = this.formGroup.get(groupName);
    const currentSelections = group?.get('indices')?.value || [];
    const selectedIds = currentSelections.map((v: number) => String(v));

    const dialogRef = this.dialog.open(CounterSelectionComponent, {
      width: '1150px',
      maxWidth: 'none',
      height: '650px',
      data: {
        config,
        selectedIds,
      },
    });

    dialogRef.afterClosed().pipe(takeUntilDestroyed(this.destroyRef)).subscribe((result: string[]) => {
      if (result) {
        const numericIndices = result
          .map((id) => Number(id))
          .sort((a, b) => a - b);
        group?.get('indices')?.setValue(numericIndices);
        this.cdr.markForCheck();
      }
    });
  }
}
