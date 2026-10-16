import {NgClass} from '@angular/common';
import {MatOption} from '@angular/material/core';
import {MatFormField, MatLabel} from '@angular/material/form-field';
import {MatSelect} from '@angular/material/select';
import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Input,
  Output,
} from '@angular/core';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';

/** A side navigation component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
  imports: [
    NgClass,
    MatFormField,
    MatLabel,
    MatSelect,
    MatOption,
  ],
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

  emitUpdateEvent() {
    this.changed.emit({
      moduleName: this.selectedModule,
      memorySpaceColor: this.selectedMemorySpaceColor,
    });
  }
}
