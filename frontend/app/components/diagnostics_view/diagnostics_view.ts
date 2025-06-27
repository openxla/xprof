import {CommonModule} from '@angular/common';
import {Component, Input} from '@angular/core';
import {type Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';

/** An diagnostics view component. */
@Component({
  standalone: true,
  selector: 'diagnostics-view',
  templateUrl: 'diagnostics_view.ng.html',
  styleUrls: ['diagnostics_view.scss'],
  imports: [CommonModule],
})
export class DiagnosticsView {
  /** Error and warning messages for diagnosing profiling issues */
  @Input() diagnostics: Diagnostics = {info: [], warnings: [], errors: []};
}
