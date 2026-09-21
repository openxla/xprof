import {ChangeDetectionStrategy, Component, effect, input} from '@angular/core';
import {MatCard, MatCardContent, MatCardTitle} from '@angular/material/card';
import {type RunEnvironment} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A run environment view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'run-environment-view',
  templateUrl: './run_environment_view.ng.html',
  styleUrls: ['./run_environment_view.scss'],
  imports: [MatCard, MatCardContent, MatCardTitle],
})
export class RunEnvironmentView {
  /** The run environment data. */
  readonly runEnvironment = input<RunEnvironment | null>(null);

  title = 'Run Environment';
  deviceCoreCount = '';
  deviceType = '';
  hostCount = '';
  isTraining = '';
  profileStartTime = '';
  profileDurationMs = '';

  constructor() {
    effect(() => {
      const data = this.runEnvironment();
      this.deviceCoreCount = this.getProperty('device_core_count', data);
      this.deviceType = this.getProperty('device_type', data);
      this.hostCount = this.getProperty('host_count', data);
      this.isTraining = this.getProperty('is_training', data);
      this.profileStartTime = this.getProperty('profile_start_time', data);
      this.profileDurationMs = this.getProperty('profile_duration_ms', data);
    });
  }

  getProperty(propertyKey: string, data: RunEnvironment | null) {
    return data?.p?.[propertyKey] || '';
  }
}
