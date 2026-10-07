import {CommonModule} from '@angular/common';
import {MatOptionModule} from '@angular/material/core';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatSelectModule} from '@angular/material/select';
import {
  ChangeDetectionStrategy,
  Component,
  effect,
  input,
  output,
} from '@angular/core';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';

/** A side navigation component. */
@Component({
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    CommonModule,
    MatFormFieldModule,
    MatSelectModule,
    MatOptionModule,
  ],
  selector: 'memory-viewer-control',
  templateUrl: './memory_viewer_control.ng.html',
  styleUrls: ['./memory_viewer_control.scss'],
})
export class MemoryViewerControl {
  /** The hlo module list. */
  readonly moduleList = input<string[]>([]);

  /** The initially selected module. */
  readonly firstLoadSelectedModule = input('');

  /** The initially selected memory space color. */
  readonly firstLoadSelectedMemorySpaceColor = input('');

  /** The event when the controls are changed. */
  readonly changed = output<NavigationEvent>();

  selectedModule = '';
  selectedMemorySpaceColor = '';

  constructor() {
    effect(() => {
      this.selectedModule = this.firstLoadSelectedModule();
    });
    effect(() => {
      this.selectedMemorySpaceColor = this.firstLoadSelectedMemorySpaceColor();
    });
  }

  emitUpdateEvent() {
    this.changed.emit({
      moduleName: this.selectedModule,
      memorySpaceColor: this.selectedMemorySpaceColor,
    });
  }
}
