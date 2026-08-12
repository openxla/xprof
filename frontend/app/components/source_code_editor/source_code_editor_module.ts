import {CommonModule} from '@angular/common';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatOptionModule} from '@angular/material/core';
import {MatSelectModule} from '@angular/material/select';
import {NgModule} from '@angular/core';

import {SourceCodeEditor} from './source_code_editor';

@NgModule({
  declarations: [SourceCodeEditor],
  imports: [
    CommonModule,
    MatProgressSpinnerModule,
    MatFormFieldModule,
    MatSelectModule,
    MatOptionModule,
  ],
  exports: [SourceCodeEditor],
})
export class SourceCodeEditorModule {}
