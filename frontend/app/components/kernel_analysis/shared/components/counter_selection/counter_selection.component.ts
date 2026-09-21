import {
  Component,
  OnInit,
  computed,
  inject,
  input,
  output,
  signal,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatCheckboxModule} from '@angular/material/checkbox';
import {MAT_DIALOG_DATA, MatDialogRef} from '@angular/material/dialog';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatListModule} from '@angular/material/list';
import {MatTooltipModule} from '@angular/material/tooltip';

import type {
  CounterGroup,
  CounterSelectionConfig,
  CounterSelectionDialogData,
} from './types';

/** Component for selecting counters from a categorized list. */
@Component({
  selector: 'app-counter-selection',
  imports: [
    FormsModule,
    MatExpansionModule,
    MatCheckboxModule,
    MatFormFieldModule,
    MatInputModule,
    MatIconModule,
    MatButtonModule,
    MatListModule,
    MatTooltipModule,
  ],
  templateUrl: './counter_selection.component.html',
  styleUrls: ['./counter_selection.component.scss'],
})
export class CounterSelectionComponent implements OnInit {
  data = inject<CounterSelectionDialogData | null>(MAT_DIALOG_DATA, {
    optional: true,
  });
  dialogRef = inject<MatDialogRef<CounterSelectionComponent> | null>(
    MatDialogRef,
    {optional: true},
  );

  readonly config = input<CounterSelectionConfig>(
    this.data?.config ?? {groups: []},
  );
  readonly selectedIds = input<string[]>(this.data?.selectedIds ?? []);
  readonly layout = input<'columns' | 'list'>(this.data?.layout ?? 'columns');
  readonly density = input<'comfortable' | 'compact'>(
    this.data?.density ?? 'compact',
  );

  readonly applied = output<string[]>();
  readonly cancelled = output<void>();

  readonly searchText = signal('');
  readonly currentSelections = signal<Set<string>>(new Set());

  ngOnInit() {
    this.currentSelections.set(new Set(this.selectedIds()));
  }

  filteredGroups = computed(() => {
    const filter = this.searchText().trim().toLowerCase();
    const config = this.config();
    if (!filter) return config.groups;

    const isPureNumber = /^\d+$/.test(filter);
    const useExactMatch = config.exactMatchForPureNumbers && isPureNumber;

    return config.groups
      .map((group) => {
        const filteredMetrics = group.counters.filter((metric) => {
          if (useExactMatch) {
            return metric.id === filter;
          }
          return (
            metric.label.toLowerCase().includes(filter) ||
            metric.id.toLowerCase().includes(filter) ||
            (metric.description &&
              metric.description.toLowerCase().includes(filter))
          );
        });

        if (
          filteredMetrics.length > 0 ||
          group.name.toLowerCase().includes(filter)
        ) {
          return {
            ...group,
            counters:
              filteredMetrics.length > 0 ? filteredMetrics : group.counters,
            expandByDefault: true, // Force expand if matched
          };
        }
        return null;
      })
      .filter((group) => group !== null) as CounterGroup[];
  });

  selectedCounters = computed(() => {
    const selections = this.currentSelections();
    const allMetrics = this.config().groups.flatMap((g) => g.counters);
    return allMetrics.filter((m) => selections.has(m.id));
  });

  isSelected(id: string): boolean {
    return this.currentSelections().has(id);
  }

  toggleSelection(id: string) {
    const newSelections = new Set(this.currentSelections());
    if (newSelections.has(id)) {
      newSelections.delete(id);
    } else {
      newSelections.add(id);
    }
    this.currentSelections.set(newSelections);
  }

  isAllSelected(group: CounterGroup): boolean {
    return group.counters.every((metric) =>
      this.currentSelections().has(metric.id),
    );
  }

  isSomeSelected(group: CounterGroup): boolean {
    const selectedCount = group.counters.filter((metric) =>
      this.currentSelections().has(metric.id),
    ).length;
    return selectedCount > 0 && selectedCount < group.counters.length;
  }

  toggleGroupSelection(group: CounterGroup) {
    const newSelections = new Set(this.currentSelections());
    const allSelected = this.isAllSelected(group);

    group.counters.forEach((metric) => {
      if (allSelected) {
        newSelections.delete(metric.id);
      } else {
        newSelections.add(metric.id);
      }
    });
    this.currentSelections.set(newSelections);
  }

  clearAll() {
    this.currentSelections.set(new Set());
  }

  onApply() {
    const selections = Array.from(this.currentSelections());
    this.applied.emit(selections);
    if (this.dialogRef) {
      this.dialogRef.close(selections);
    }
  }

  onCancel() {
    this.cancelled.emit();
    if (this.dialogRef) {
      this.dialogRef.close();
    }
  }
}
