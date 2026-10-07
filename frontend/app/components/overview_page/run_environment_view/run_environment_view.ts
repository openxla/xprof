import {NgIf} from '@angular/common';
import {ChangeDetectionStrategy, Component, input} from '@angular/core';
import {MatCard, MatCardContent, MatCardTitle} from '@angular/material/card';
import {type RunEnvironment} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A run environment view component. */
@Component({
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'run-environment-view',
  templateUrl: './run_environment_view.ng.html',
  styleUrls: ['./run_environment_view.scss'],
  imports: [MatCard, MatCardContent, MatCardTitle, NgIf],
})
export class RunEnvironmentView {
  /** The run environment data. */
  readonly runEnvironment = input<RunEnvironment | null>(null);

  title = 'Run Environment';

  get deviceCoreCount() {
    return this.getProperty('device_core_count', this.runEnvironment());
  }

  get deviceType() {
    return this.getProperty('device_type', this.runEnvironment());
  }

  get hostCount() {
    return this.getProperty('host_count', this.runEnvironment());
  }

  get isTraining() {
    return this.getProperty('is_training', this.runEnvironment());
  }

  get profileStartTime() {
    return this.getProperty('profile_start_time', this.runEnvironment());
  }

  get profileDurationMs() {
    return this.getProperty('profile_duration_ms', this.runEnvironment());
  }

  getProperty(propertyKey: string, data: RunEnvironment | null) {
    return data?.p?.[propertyKey] || '';
  }
}
