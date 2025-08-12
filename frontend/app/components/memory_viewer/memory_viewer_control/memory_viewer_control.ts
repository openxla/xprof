import {Component, ElementRef, EventEmitter, Input, OnChanges, Output, SimpleChanges, ViewChild} from '@angular/core';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';

/** A side navigation component. */
@Component({
  standalone: false,
  selector: 'memory-viewer-control',
  templateUrl: './memory_viewer_control.ng.html',
  styleUrls: ['./memory_viewer_control.scss'],
})
export class MemoryViewerControl implements OnChanges {
  /** The hlo module list. */
  @Input() moduleList: string[] = [];
  @Input() firstLoadSelectedModule = '';
  @Input() firstLoadSelectedMemorySpaceColor = '';

  /** The event when the controls are changed. */
  @Output() readonly changed = new EventEmitter<NavigationEvent>();

  @ViewChild('searchInput') searchInput!: ElementRef<HTMLInputElement>;

  selectedModule = '';
  selectedMemorySpaceColor = '';
  filterText = '';
  filteredModuleList: string[] = [];

  ngOnChanges(changes: SimpleChanges) {
    if (changes['firstLoadSelectedModule']?.currentValue !==
            changes['firstLoadSelectedModule']?.previousValue &&
        this.selectedModule === '') {
      this.selectedModule = changes['firstLoadSelectedModule'].currentValue;
    }
    if (changes['firstLoadSelectedMemorySpaceColor']?.currentValue !==
            changes['firstLoadSelectedMemorySpaceColor']?.previousValue &&
        this.selectedMemorySpaceColor === '') {
      this.selectedMemorySpaceColor =
          changes['firstLoadSelectedMemorySpaceColor'].currentValue;
    }
    if (changes['moduleList']?.currentValue !==
        changes['moduleList']?.previousValue) {
      this.filterModules();
    }
  }

  ngAfterViewInit() {
    this.searchInput.nativeElement.focus();
  }

  filterModules() {
    if (!this.moduleList) {
      this.filteredModuleList = [];
      return;
    }
    if (!this.filterText) {
      this.filteredModuleList = this.moduleList;
      return;
    }
    const filterText = this.filterText.toLowerCase();
    this.filteredModuleList = this.moduleList.filter(
        module => module.toLowerCase().includes(filterText));
  }

  emitUpdateEvent() {
    this.changed.emit({
      moduleName: this.selectedModule,
      memorySpaceColor: this.selectedMemorySpaceColor,
    });
  }
}
