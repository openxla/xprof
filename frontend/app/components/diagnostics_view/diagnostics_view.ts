import {ChangeDetectionStrategy, Component, input} from '@angular/core';
import {MatIconButton} from '@angular/material/button';
import {MatIcon} from '@angular/material/icon';
import {type Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';

/** An diagnostics view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'diagnostics-view',
  templateUrl: './diagnostics_view.ng.html',
  styleUrls: ['./diagnostics_view.scss'],
  imports: [MatIcon, MatIconButton],
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
    const diagnostics = this.diagnostics();
    if (!diagnostics) return;
    if (category === 'errors' && diagnostics.errors) {
      if (index >= 0 && index < diagnostics.errors.length) {
        diagnostics.errors.splice(index, 1);
      }
    } else if (category === 'warnings' && diagnostics.warnings) {
      if (index >= 0 && index < diagnostics.warnings.length) {
        diagnostics.warnings.splice(index, 1);
      }
    } else if (category === 'info' && diagnostics.info) {
      if (index >= 0 && index < diagnostics.info.length) {
        diagnostics.info.splice(index, 1);
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
