import {CommonModule} from '@angular/common';
import {
  Component,
  EventEmitter,
  Inject,
  Input,
  OnInit,
  Optional,
  Output,
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
  CounterOption,
  CounterSelectionConfig,
} from './types';

/** Component for selecting counters from a categorized list. */
@Component({
  selector: 'app-counter-selection',
  standalone: true,
  imports: [
    CommonModule,
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
  @Input() config: CounterSelectionConfig = {groups: []};
  @Input() selectedIds: string[] = [];
  @Input() layout: 'columns' | 'list' = 'columns';
  @Input() density: 'comfortable' | 'compact' = 'compact';

  @Output() readonly applied = new EventEmitter<string[]>();
  @Output() readonly cancelled = new EventEmitter<void>();

  searchText = '';
  currentSelections = new Set<string>();

  constructor(
    @Optional()
    @Inject(MAT_DIALOG_DATA)
    public data: {
      config: CounterSelectionConfig;
      selectedIds: string[];
      layout: 'columns' | 'list';
      density: 'comfortable' | 'compact';
    } | null,
    @Optional()
    public dialogRef: MatDialogRef<CounterSelectionComponent> | null,
  ) {
    if (data) {
      this.config = data.config || this.config;
      this.selectedIds = data.selectedIds || this.selectedIds;
      this.layout = data.layout || this.layout;
      this.density = data.density || this.density;
    }
  }

  ngOnInit() {
    this.currentSelections = new Set(this.selectedIds);
  }

  get filteredGroups(): CounterGroup[] {
    const filter = this.searchText.trim().toLowerCase();
    if (!filter) return this.config.groups;

    const isPureNumber = /^\d+$/.test(filter);
    const useExactMatch = this.config.exactMatchForPureNumbers && isPureNumber;

    return this.config.groups
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
  }

  get selectedCounters(): CounterOption[] {
    const selections = this.currentSelections;
    const allMetrics = this.config.groups
      .map((g) => g.counters)
      .reduce((acc, val) => acc.concat(val), []);
    return allMetrics.filter((m) => selections.has(m.id));
  }

  isSelected(id: string): boolean {
    return this.currentSelections.has(id);
  }

  toggleSelection(id: string) {
    const newSelections = new Set(this.currentSelections);
    if (newSelections.has(id)) {
      newSelections.delete(id);
    } else {
      newSelections.add(id);
    }
    this.currentSelections = newSelections;
  }

  isAllSelected(group: CounterGroup): boolean {
    return group.counters.every((metric) =>
      this.currentSelections.has(metric.id),
    );
  }

  isSomeSelected(group: CounterGroup): boolean {
    const selectedCount = group.counters.filter((metric) =>
      this.currentSelections.has(metric.id),
    ).length;
    return selectedCount > 0 && selectedCount < group.counters.length;
  }

  toggleGroupSelection(group: CounterGroup) {
    const newSelections = new Set(this.currentSelections);
    const allSelected = this.isAllSelected(group);

    group.counters.forEach((metric) => {
      if (allSelected) {
        newSelections.delete(metric.id);
      } else {
        newSelections.add(metric.id);
      }
    });
    this.currentSelections = newSelections;
  }

  clearAll() {
    this.currentSelections = new Set();
  }

  onApply() {
    const selections = Array.from(this.currentSelections);
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
