import '@material/web/icon/icon.js';
import '@material/web/iconbutton/icon-button.js';

import {
  ChangeDetectionStrategy,
  Component,
  CUSTOM_ELEMENTS_SCHEMA,
  input,
} from '@angular/core';
import {type Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';

/** An diagnostics view component. */
@Component({
  standalone: true,
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'diagnostics-view',
  templateUrl: './diagnostics_view.ng.html',
  styleUrls: ['./diagnostics_view.scss'],
  schemas: [CUSTOM_ELEMENTS_SCHEMA],
})
export class DiagnosticsView {
  /** Error and warning messages for diagnosing profiling issues */
  readonly diagnostics = input<Diagnostics>({
    info: [],
    warnings: [],
    errors: [],
  });
  showErrors = true;
  showWarnings = true;
  showInfo = true;

  /** Dismisses an individual message by category and index. */
  dismissMessage(
    category: 'errors' | 'warnings' | 'info',
    index: number,
  ): void {
    const diag = this.diagnostics();
    if (!diag) return;
    if (category === 'errors' && diag.errors) {
      if (index >= 0 && index < diag.errors.length) {
        diag.errors.splice(index, 1);
      }
    } else if (category === 'warnings' && diag.warnings) {
      if (index >= 0 && index < diag.warnings.length) {
        diag.warnings.splice(index, 1);
      }
    } else if (category === 'info' && diag.info) {
      if (index >= 0 && index < diag.info.length) {
        diag.info.splice(index, 1);
      }
    }
  }

  /** Dismisses an individual error message by index. */
  dismissError(index: number): void {
    this.dismissMessage('errors', index);
  }

  /** Dismisses an individual warning message by index. */
  dismissWarning(index: number): void {
    this.dismissMessage('warnings', index);
  }

  /** Dismisses an individual info message by index. */
  dismissInfo(index: number): void {
    this.dismissMessage('info', index);
  }
}
