import {CommonModule} from '@angular/common';
import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
  ViewChild,
} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatOptionModule} from '@angular/material/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatInputModule} from '@angular/material/input';
import {MatSelectModule} from '@angular/material/select';

/**
 * A reusable standalone component for a searchable dropdown.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
  selector: 'app-searchable-dropdown',
  templateUrl: './searchable_dropdown.ng.html',
  styleUrls: ['./searchable_dropdown.scss'],
  imports: [
    CommonModule,
    FormsModule,
    MatSelectModule,
    MatFormFieldModule,
    MatInputModule,
    MatIconModule,
    MatOptionModule,
  ],
})
export class SearchableDropdown implements OnChanges, AfterViewInit {
  @Input() itemList: string[] = [];
  @Input() selectedItem = '';
  @Input() label = '';
  @Output() readonly selectionChange = new EventEmitter<string>();
  @ViewChild('searchInput') searchInput!: ElementRef<HTMLInputElement>;

  filterText = '';
  filteredItemList: string[] = [];

  ngOnChanges(changes: SimpleChanges) {
    if (changes['itemList']) {
      this.updateFilteredItemList();
    }
  }

  updateFilteredItemList() {
    const items = this.itemList ?? [];
    if (!items.length) {
      this.filteredItemList = [];
      return;
    }
    const filter = this.filterText.trim().toLowerCase();
    if (!filter) {
      this.filteredItemList = items;
      return;
    }
    this.filteredItemList = items.filter((item) =>
      item.toLowerCase().includes(filter),
    );
  }

  ngAfterViewInit() {
    setTimeout(() => {
      if (this.searchInput) {
        this.searchInput.nativeElement.focus();
      }
    }, 0);
  }

  trackByItem(index: number, item: string): string {
    return item;
  }

  emitSelectionChange() {
    this.selectionChange.emit(this.selectedItem);
  }
}
