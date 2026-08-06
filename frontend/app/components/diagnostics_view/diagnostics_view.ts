import {ChangeDetectionStrategy, Component, Input} from '@angular/core';
import {type Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';

/** An diagnostics view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'diagnostics-view',
  templateUrl: './diagnostics_view.ng.html',
  styleUrls: ['./diagnostics_view.scss'],
})
export class DiagnosticsView {
  /** Error and warning messages for diagnosing profiling issues */
  @Input() diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
  showErrors = true;
  showWarnings = true;
  showInfo = true;

  /** Dismisses an individual message by category and index. */
  dismissMessage(
    category: 'errors' | 'warnings' | 'info',
    index: number,
  ): void {
    if (!this.diagnostics) return;
    if (category === 'errors' && this.diagnostics.errors) {
      if (index >= 0 && index < this.diagnostics.errors.length) {
        this.diagnostics.errors.splice(index, 1);
      }
    } else if (category === 'warnings' && this.diagnostics.warnings) {
      if (index >= 0 && index < this.diagnostics.warnings.length) {
        this.diagnostics.warnings.splice(index, 1);
      }
    } else if (category === 'info' && this.diagnostics.info) {
      if (index >= 0 && index < this.diagnostics.info.length) {
        this.diagnostics.info.splice(index, 1);
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
