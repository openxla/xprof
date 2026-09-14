import {NgFor, UpperCasePipe} from '@angular/common';
import {ChangeDetectionStrategy, Component, inject} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatButton} from '@angular/material/button';
import {MatOption} from '@angular/material/core';
import {
  MatDialogActions,
  MatDialogContent,
  MatDialogRef,
} from '@angular/material/dialog';
import {
  MatExpansionPanel,
  MatExpansionPanelHeader,
  MatExpansionPanelTitle,
} from '@angular/material/expansion';
import {MatFormField, MatLabel} from '@angular/material/form-field';
import {MatInput} from '@angular/material/input';
import {MatRadioButton, MatRadioGroup} from '@angular/material/radio';
import {MatSelect} from '@angular/material/select';
import {MatTooltip} from '@angular/material/tooltip';

/** A capture profile dialog component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'capture-profile-dialog',
  templateUrl: './capture_profile_dialog.ng.html',
  styleUrls: ['./capture_profile_dialog.scss'],
  imports: [
    FormsModule,
    MatButton,
    MatDialogActions,
    MatDialogContent,
    MatExpansionPanel,
    MatExpansionPanelHeader,
    MatExpansionPanelTitle,
    MatFormField,
    MatInput,
    MatLabel,
    MatOption,
    MatRadioButton,
    MatRadioGroup,
    MatSelect,
    MatTooltip,
    NgFor,
    UpperCasePipe,
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
  hostTracerTooltip =
    'lower trace level to reduce amount of host traces ' +
    'collected, some tools will not function well when the host tracer ' +
    'level is less than info';
  deviceTracerLevel = '1';
  pythonTracerLevel = '0';
  delay = 0;
  extraOptions: Array<{key: string; value: string}> = [];

  private readonly dialogRef = inject(MatDialogRef<CaptureProfileDialog>);

  addressTypeChanged(value: string) {
    this.isTpuName = value === 'tpu';
  }

  serviceAddrChanged(value: string) {
    this.serviceAddr = value.trim();
  }

  captureProfile() {
    const options: {[key: string]: string | number | boolean} = {
      'serviceAddr': this.serviceAddr,
      'isTpuName': this.isTpuName,
      'duration': this.duration,
      'numRetry': this.numRetry,
      'workerList': this.workerList,
      'hostTracerLevel': Number(this.hostTracerLevel),
      'deviceTracerLevel': Number(this.deviceTracerLevel),
      'pythonTracerLevel': Number(this.pythonTracerLevel),
      'delay': this.delay,
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
