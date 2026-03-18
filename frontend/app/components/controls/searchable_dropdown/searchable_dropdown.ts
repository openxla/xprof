import {CommonModule} from '@angular/common';
import {AfterViewInit, Component, ElementRef, EventEmitter, Input, Output, ViewChild, ChangeDetectionStrategy} from '@angular/core';
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
  changeDetection: ChangeDetectionStrategy.Eager,standalone: true,
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
export class SearchableDropdown implements AfterViewInit {
  @Input() itemList: string[] = [];
  @Input() selectedItem = '';
  @Input() label = '';
  @Output() readonly selectionChange = new EventEmitter<string>();
  @ViewChild('searchInput') searchInput!: ElementRef<HTMLInputElement>;

  filterText = '';

  get filteredItemList(): string[] {
    if (!this.itemList) {
      return [];
    }
    if (!this.filterText) {
      return this.itemList;
    }
    const filter = this.filterText.trim().toLowerCase();
    return this.itemList.filter(item => item.toLowerCase().includes(filter));
  }

  ngAfterViewInit() {
    setTimeout(() => {
      if (this.searchInput) {
        this.searchInput.nativeElement.focus();
      }
    }, 0);
  }

  emitSelectionChange() {
    this.selectionChange.emit(this.selectedItem);
  }
}
