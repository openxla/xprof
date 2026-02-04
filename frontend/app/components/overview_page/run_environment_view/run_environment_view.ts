import {CommonModule} from '@angular/common';
import {Component, Input} from '@angular/core';
import {MatCardModule} from '@angular/material/card';
import {type RunEnvironment} from 'org_xprof/frontend/app/common/interfaces/data_table';

/** A run environment view component. */
@Component({
  standalone: true,
  selector: 'run-environment-view',
  templateUrl: 'run_environment_view.ng.html',
  styleUrls: ['run_environment_view.scss'],
  imports: [CommonModule, MatCardModule],
})
export class RunEnvironmentView {
  /** The run environment data. */
  @Input()
  set runEnvironment(data: RunEnvironment|null) {
    this.deviceCoreCount = this.getProperty('device_core_count', data);
    this.deviceType = this.getProperty('device_type', data);
    this.hostCount = this.getProperty('host_count', data);
    this.isTraining = this.getProperty('is_training', data);
    this.profileStartTime = this.getProperty('profile_start_time', data);
    this.profileDurationMs = this.getProperty('profile_duration_ms', data);
  }

  title = 'Run Environment';
  deviceCoreCount = '';
  deviceType = '';
  hostCount = '';
  isTraining = '';
  profileStartTime = '';
  profileDurationMs = '';

  getProperty(propertyKey: string, data: RunEnvironment|null) {
    return data?.p?.[propertyKey] || '';
  }
}
