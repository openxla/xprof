import {Component, EventEmitter, Input, OnChanges, Output, SimpleChanges} from '@angular/core';
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

  selectedModule = '';
  selectedMemorySpaceColor = '';

  ngOnChanges(changes: SimpleChanges) {
    if (changes['firstLoadSelectedModule']) {
      this.selectedModule = this.firstLoadSelectedModule;
    }
    if (changes['firstLoadSelectedMemorySpaceColor']) {
      this.selectedMemorySpaceColor = this.firstLoadSelectedMemorySpaceColor;
    }
  }

  emitUpdateEvent() {
    this.changed.emit({
      moduleName: this.selectedModule,
      memorySpaceColor: this.selectedMemorySpaceColor,
    });
  }
}
