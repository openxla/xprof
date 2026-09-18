import {NgClass} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  effect,
  input,
  output,
} from '@angular/core';
import {MatOption} from '@angular/material/core';
import {MatFormField, MatLabel} from '@angular/material/form-field';
import {MatSelect} from '@angular/material/select';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {DownloadHlo} from '../../controls/download_hlo/download_hlo';
import {SearchableDropdown} from '../../controls/searchable_dropdown/searchable_dropdown';

/** A side navigation component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'memory-viewer-control',
  templateUrl: './memory_viewer_control.ng.html',
  styleUrls: ['./memory_viewer_control.scss'],
  imports: [
    DownloadHlo,
    MatFormField,
    MatLabel,
    MatOption,
    MatSelect,
    NgClass,
    SearchableDropdown,
  ],
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
