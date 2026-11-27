import {CommonModule} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  EventEmitter,
  Input,
  OnChanges,
  OnDestroy,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {MatOptionModule} from '@angular/material/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatSelectModule} from '@angular/material/select';

function arraysEqual(a: readonly string[], b: readonly string[]): boolean {
  if (a === b) return true;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/**
 * A reusable standalone component for a searchable dropdown.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: true,
  selector: 'app-searchable-dropdown',
  templateUrl: './searchable_dropdown.ng.html',
  styleUrls: ['./searchable_dropdown.scss'],
  imports: [
    CommonModule,
    MatSelectModule,
    MatFormFieldModule,
    MatIconModule,
    MatOptionModule,
  ],
})
export class SearchableDropdown implements OnChanges, OnDestroy {
  @Input() itemList: string[] = [];
  @Input() selectedItem = '';
  @Input() label = '';
  @Output() readonly selectionChange = new EventEmitter<string>();
  @ViewChild('searchInput') searchInput?: ElementRef<HTMLInputElement>;

  filterText = '';
  private focusTimerId?: ReturnType<typeof setTimeout>;
  private isPanelOpen = false;

  get hasMatchingItems(): boolean {
    if (!this.itemList?.length) {
      return false;
    }
    const filter = this.filterText.trim().toLowerCase();
    if (!filter) {
      return true;
    }
    return this.itemList.some((item) => item.toLowerCase().includes(filter));
  }

  matchesFilter(item: string): boolean {
    const filter = this.filterText.trim().toLowerCase();
    if (!filter) {
      return true;
    }
    return item.toLowerCase().includes(filter);
  }

  ngOnChanges(changes: SimpleChanges): void {
    const itemListChange = changes['itemList'];
    if (itemListChange && !itemListChange.isFirstChange()) {
      const prev: readonly string[] = itemListChange.previousValue ?? [];
      const curr: readonly string[] = itemListChange.currentValue ?? [];
      if (!arraysEqual(prev, curr)) {
        this.filterText = '';
      }
    }
  }

  onOpenedChange(opened: boolean): void {
    this.isPanelOpen = opened;
    this.clearFocusTimer();
    if (opened) {
      this.focusTimerId = setTimeout(() => {
        this.focusTimerId = undefined;
        if (this.isPanelOpen) {
          this.searchInput?.nativeElement.focus();
        }
      }, 0);
    } else {
      this.filterText = '';
    }
  }

  onFilterInput(event: Event): void {
    const target = event.target;
    if (target instanceof HTMLInputElement) {
      this.filterText = target.value;
    }
  }

  onSearchKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape' || event.key === 'Tab') {
      return;
    }
    if (
      event.key === 'Enter' ||
      event.key === 'ArrowUp' ||
      event.key === 'ArrowDown'
    ) {
      event.preventDefault();
    }
    event.stopPropagation();
  }

  emitSelectionChange(): void {
    this.selectionChange.emit(this.selectedItem);
  }

  ngOnDestroy(): void {
    this.isPanelOpen = false;
    this.clearFocusTimer();
  }

  private clearFocusTimer(): void {
    if (this.focusTimerId !== undefined) {
      clearTimeout(this.focusTimerId);
      this.focusTimerId = undefined;
    }
  }
}
