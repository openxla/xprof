import {NgFor, NgIf} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  Input,
  OnChanges,
  Output,
  SimpleChanges,
} from '@angular/core';
import {MatIconButton} from '@angular/material/button';
import {MatIcon} from '@angular/material/icon';
import {type Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';

/** An diagnostics view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
  selector: 'diagnostics-view',
  templateUrl: './diagnostics_view.ng.html',
  styleUrls: ['./diagnostics_view.scss'],
  imports: [MatIcon, MatIconButton, NgFor, NgIf],
})
export class DiagnosticsView implements OnChanges {
  /** Error and warning messages for diagnosing profiling issues */
  @Input() diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
  /** Emits updated diagnostics when a message is dismissed. */
  @Output() readonly diagnosticsChange = new EventEmitter<Diagnostics>();
  private localDiagnostics?: Diagnostics;
  showErrors = true;
  showWarnings = true;
  showInfo = true;

  get currentDiagnostics(): Diagnostics {
    return this.localDiagnostics ?? this.diagnostics;
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['diagnostics']) {
      this.localDiagnostics = undefined;
    }
  }

  trackByMsg(index: number, msg: string): number {
    return index;
  }

  /** Dismisses an individual message by category and index. */
  dismissMessage(
    category: 'errors' | 'warnings' | 'info',
    index: number,
  ): void {
    const current = this.currentDiagnostics;
    if (!current) return;
    if (category === 'errors' && current.errors) {
      if (index >= 0 && index < current.errors.length) {
        this.localDiagnostics = {
          ...current,
          errors: current.errors.filter((_, i) => i !== index),
        };
        this.diagnosticsChange.emit(this.localDiagnostics);
      }
    } else if (category === 'warnings' && current.warnings) {
      if (index >= 0 && index < current.warnings.length) {
        this.localDiagnostics = {
          ...current,
          warnings: current.warnings.filter((_, i) => i !== index),
        };
        this.diagnosticsChange.emit(this.localDiagnostics);
      }
    } else if (category === 'info' && current.info) {
      if (index >= 0 && index < current.info.length) {
        this.localDiagnostics = {
          ...current,
          info: current.info.filter((_, i) => i !== index),
        };
        this.diagnosticsChange.emit(this.localDiagnostics);
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
