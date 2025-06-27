import {CommonModule} from '@angular/common';
import {Component} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButtonModule} from '@angular/material/button';
import {MatDialogModule, MatDialogRef} from '@angular/material/dialog';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatInputModule} from '@angular/material/input';
import {MatRadioModule} from '@angular/material/radio';
import {MatSelectModule} from '@angular/material/select';
import {MatTooltipModule} from '@angular/material/tooltip';

/** A capture profile dialog component. */
@Component({
  standalone: true,
  selector: 'capture-profile-dialog',
  templateUrl: 'capture_profile_dialog.ng.html',
  styleUrls: ['capture_profile_dialog.scss'],
  imports: [
    CommonModule,
    FormsModule,
    MatButtonModule,
    MatDialogModule,
    MatExpansionModule,
    MatFormFieldModule,
    MatInputModule,
    MatRadioModule,
    MatSelectModule,
    MatTooltipModule,
  ],
})
export class CaptureProfileDialog {
  captureButtonLabel = 'Capture';
  closeButtonLabel = 'Close';
  serviceAddr = '';
  isTpuName = false;
  addressType = 'ip';
  duration = 1000;
  numRetry = 3;
  workerList = '';
  hostTracerLevel = '2';
  hostTracerTooltip = 'lower trace level to reduce amount of host traces ' +
      'collected, some tools will not function well when the host tracer ' +
      'level is less than info';
  deviceTracerLevel = '1';
  pythonTracerLevel = '0';
  delay = 0;
  extraOptions: Array<{key: string, value: string}> = [];

  constructor(private readonly dialogRef:
                  MatDialogRef<CaptureProfileDialog>) {}

  addressTypeChanged(value: string) {
    this.isTpuName = value === 'tpu';
  }

  serviceAddrChanged(value: string) {
    this.serviceAddr = value.trim();
  }

  captureProfile() {
    const options: {[key: string]: string|number|boolean} = {
      serviceAddr: this.serviceAddr,
      isTpuName: this.isTpuName,
      duration: this.duration,
      numRetry: this.numRetry,
      workerList: this.workerList,
      hostTracerLevel: Number(this.hostTracerLevel),
      deviceTracerLevel: Number(this.deviceTracerLevel),
      pythonTracerLevel: Number(this.pythonTracerLevel),
      delay: this.delay,
    };

    for (const option of this.extraOptions) {
      options[option.key] = option.value;
    }

    this.dialogRef.close(options);
  }

  close() {
    this.dialogRef.close();
  }

  addExtraOption() {
    this.extraOptions.push({key: '', value: ''});
  }

  removeExtraOption(index: number) {
    this.extraOptions.splice(index, 1);
  }
}
