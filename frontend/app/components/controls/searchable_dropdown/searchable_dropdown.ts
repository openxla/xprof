import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  input,
  output,
  viewChild,
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
  selector: 'app-searchable-dropdown',
  templateUrl: './searchable_dropdown.ng.html',
  styleUrls: ['./searchable_dropdown.scss'],
  imports: [
    FormsModule,
    MatSelectModule,
    MatFormFieldModule,
    MatInputModule,
    MatIconModule,
    MatOptionModule,
  ],
})
export class SearchableDropdown implements AfterViewInit {
  readonly itemList = input<string[]>([]);
  readonly selectedItem = input('');
  readonly label = input('');
  readonly selectionChange = output<string>();
  readonly searchInput = viewChild<ElementRef<HTMLInputElement>>('searchInput');

  filterText = '';

  get filteredItemList(): string[] {
    const list = this.itemList();
    if (!list) {
      return [];
    }
    if (!this.filterText) {
      return list;
    }
    const filter = this.filterText.trim().toLowerCase();
    return list.filter((item) => item.toLowerCase().includes(filter));
  }

  ngAfterViewInit() {
    setTimeout(() => {
      this.searchInput()?.nativeElement.focus();
    }, 0);
  }

  emitSelectionChange(value: string) {
    this.selectionChange.emit(value);
  }
}
