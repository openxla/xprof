import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {DiagnosticsView} from './diagnostics_view';

@NgModule({
  imports: [CommonModule, MatButtonModule, MatIconModule, DiagnosticsView],
  exports: [DiagnosticsView],
})
export class DiagnosticsViewModule {}
