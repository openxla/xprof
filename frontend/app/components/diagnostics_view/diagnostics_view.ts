import {Component, Input, ChangeDetectionStrategy} from '@angular/core';
import {type Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';

/** An diagnostics view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Eager,standalone: false,
  selector: 'diagnostics-view',
  templateUrl: './diagnostics_view.ng.html',
  styleUrls: ['./diagnostics_view.scss']
})
export class DiagnosticsView {
  /** Error and warning messages for diagnosing profiling issues */
  @Input() diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
}
