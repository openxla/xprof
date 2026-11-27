import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  ChangeDetectorRef,
  Component,
  Input,
  OnDestroy,
  OnInit,
  inject,
} from '@angular/core';
import {FormGroup, ReactiveFormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatButtonToggleModule} from '@angular/material/button-toggle';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MatDialog, MatDialogModule} from '@angular/material/dialog';
import {MatDividerModule} from '@angular/material/divider';
import {MatIconModule} from '@angular/material/icon';
import {type MatInput, MatInputModule} from '@angular/material/input';
import {MatRadioModule} from '@angular/material/radio';
import {MatSelectModule} from '@angular/material/select';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  type DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {EMPTY, Observable, ReplaySubject} from 'rxjs';
import {
  catchError,
  filter,
  startWith,
  switchMap,
  takeUntil,
} from 'rxjs/operators';
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

function hasDeviceId(gen: unknown): gen is {id: string; name?: string} {
  return (
    typeof gen === 'object' &&
    gen !== null &&
    'id' in gen &&
    typeof gen.id === 'string' &&
    gen.id.length > 0
  );
}

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
export class ProfilerOptionsComponent implements OnInit, OnDestroy {
  @Input() formGroup?: FormGroup;

  readonly componentTabs: ReadonlyArray<{
    readonly id: 'tc' | 'scs' | 'sctc' | 'sctd' | 'cmn' | 'icr';
    readonly groupName: string;
    readonly label: string;
  }> = [
    {id: 'tc', groupName: 'tc_sampling', label: 'TC'},
    {id: 'scs', groupName: 'scs_sampling', label: 'SCS'},
    {id: 'sctc', groupName: 'sctc_sampling', label: 'SCTC'},
    {id: 'sctd', groupName: 'sctd_sampling', label: 'SCTD'},
    {id: 'cmn', groupName: 'cmn_sampling', label: 'CMN'},
    {id: 'icr', groupName: 'icr_sampling', label: 'ICR'},
  ];

  readonly invalidErrorStateMatcher: MatInput['errorStateMatcher'] = {
    isErrorState: (control) => Boolean(control?.invalid),
  };

  selectedComponent: 'tc' | 'scs' | 'sctc' | 'sctd' | 'cmn' | 'icr' = 'tc';
  tpuGenerations = TPU_GENERATIONS;
  allCounters: Counter[] = [];

  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );
  private readonly dialog = inject(MatDialog);
  private readonly cdr = inject(ChangeDetectorRef);
  private readonly destroyed = new ReplaySubject<void>(1);

  ngOnInit() {
    if (!this.formGroup) {
      throw new Error('formGroup is required');
    }
    this.formGroup.statusChanges
      .pipe(takeUntil(this.destroyed))
      .subscribe(() => {
        this.cdr.markForCheck();
      });

    const deviceNameControl = this.formGroup.get('device_name');
    if (deviceNameControl) {
      deviceNameControl.valueChanges
        .pipe(
          startWith(deviceNameControl.value),
          filter(hasDeviceId),
          switchMap((tpuGen) => this.fetchCounters(tpuGen.id)),
          takeUntil(this.destroyed),
        )
        .subscribe((data: unknown) => {
          if (Array.isArray(data)) {
            this.allCounters = data;
            this.cdr.markForCheck();
          }
        });
    }
  }

  ngOnDestroy() {
    this.destroyed.next();
    this.destroyed.complete();
  }

  fetchCounters(deviceType: string): Observable<unknown> {
    this.allCounters = [];
    this.cdr.markForCheck();
    return this.dataService
      .getData(
        '',
        'perf_counters',
        '',
        new Map([
          ['names_only', '1'],
          ['device_type', deviceType],
        ]),
        {updateSearchParams: false},
      )
      .pipe(catchError(() => EMPTY));
  }

  convertValuesToIndices(
    counters: Counter[],
  ): Array<{name: string; index: number}> {
    if (!counters || counters.length === 0) return [];

    const validCounters = counters.filter(
      (c) => c != null && typeof c.val === 'number' && Number.isFinite(c.val),
    );
    if (validCounters.length === 0) return [];

    let minVal = Infinity;
    for (const c of validCounters) {
      if (c.val < minVal) {
        minVal = c.val;
      }
    }

    return validCounters
      .map((c) => {
        const name = typeof c.name === 'string' ? c.name : '';
        const parts = name.split('unprivileged_');
        const shortName = parts[parts.length - 1];
        const index = Math.floor((c.val - minVal) / 8);
        return {name: shortName, index};
      })
      .filter((c) => Number.isInteger(c.index) && c.index >= 0)
      .sort((a, b) => a.index - b.index);
  }

  getCounters(groupName: string): Array<{name: string; index: number}> {
    const component = Object.values(ComponentType).find(
      (c) => COMPONENT_TO_GROUP_NAME[c as ComponentType] === groupName,
    ) as ComponentType | undefined;

    if (component) {
      const keyword = COMPONENT_TO_SEARCH_KEYWORD[component];
      const counters = this.allCounters.filter(
        (c) => typeof c?.name === 'string' && c.name.includes(keyword),
      );
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

  isComponentInvalid(groupName: string): boolean {
    return Boolean(this.formGroup?.get(groupName)?.invalid);
  }

  getIntervalError(groupName: string): string {
    const control = this.formGroup?.get(`${groupName}.interval_us`);
    if (control?.hasError('min') || control?.hasError('max')) {
      return 'Must be between 0 and 2147483647';
    }
    if (control?.hasError('integer')) {
      return 'Must be a whole number';
    }
    return '';
  }

  getScalingError(groupName: string): string {
    const control = this.formGroup?.get(`${groupName}.scaling`);
    if (control?.hasError('min') || control?.hasError('max')) {
      return 'Must be between 0 and 63';
    }
    if (control?.hasError('integer')) {
      return 'Must be a whole number';
    }
    return '';
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
    const rawSelections: unknown = group?.get('indices')?.value;
    const currentSelections = Array.isArray(rawSelections) ? rawSelections : [];
    const selectedIds = currentSelections
      .filter(
        (v): v is number =>
          typeof v === 'number' && Number.isInteger(v) && v >= 0,
      )
      .map((v) => String(v));

    const dialogRef = this.dialog.open(CounterSelectionComponent, {
      width: '1150px',
      maxWidth: 'none',
      height: '650px',
      data: {
        config,
        selectedIds,
      },
    });

    dialogRef
      .afterClosed()
      .pipe(takeUntil(this.destroyed))
      .subscribe((result: string[] | undefined) => {
        if (result) {
          const numericIndices = result
            .map((id) => (id.trim() === '' ? Number.NaN : Number(id)))
            .filter((index) => Number.isInteger(index) && index >= 0)
            .sort((a, b) => a - b);
          group?.get('indices')?.setValue(numericIndices);
          this.cdr.markForCheck();
        }
      });
  }
}
