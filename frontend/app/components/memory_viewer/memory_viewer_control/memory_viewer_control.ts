import {Component, EventEmitter, Input, Output} from '@angular/core';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';

/** Control panel for the Memory Viewer component. */
@Component({
  standalone: false,
  selector: 'memory-viewer-control',
  templateUrl: './memory_viewer_control.ng.html',
  styleUrls: ['./memory_viewer_control.scss'],
})
export class MemoryViewerControl {
  private moduleListInternal: string[] = [];

  /** The hlo module list. */
  @Input()
  set moduleList(value: string[]) {
    this.moduleListInternal = value || [];
  }
  get moduleList(): string[] {
    return this.moduleListInternal;
  }

  /** The initially selected module. */
  @Input()
  set firstLoadSelectedModule(value: string) {
    this.selectedModule = value;
  }

  /** The initially selected memory space color. */
  @Input()
  set firstLoadSelectedMemorySpaceColor(value: string) {
    this.selectedMemorySpaceColor = value;
  }

  /** The event when the controls are changed. */
  @Output() readonly changed = new EventEmitter<NavigationEvent>();

  selectedModule = '';
  selectedMemorySpaceColor = '';

  emitUpdateEvent(): void {
    this.changed.emit({
      module_name: this.selectedModule,
      moduleName: this.selectedModule,
      memorySpaceColor: this.selectedMemorySpaceColor,
    });
  }
}
