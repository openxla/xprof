import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatButtonModule} from '@angular/material/button';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressSpinnerModule} from '@angular/material/progress-spinner';
import {MatTooltipModule} from '@angular/material/tooltip';
import {KernelEventTooltip} from 'org_xprof/frontend/app/components/static_kernel_viewer/kernel_event_tooltip';
import {TraceViewerContainer} from 'org_xprof/frontend/app/components/trace_viewer_container/trace_viewer_container';
import {StaticKernelViewer} from './static_kernel_viewer';

@NgModule({
  declarations: [StaticKernelViewer],
  imports: [
    CommonModule,
    KernelEventTooltip,
    MatButtonModule,
    MatIconModule,
    MatProgressSpinnerModule,
    MatTooltipModule,
    TraceViewerContainer,
  ],
  exports: [StaticKernelViewer],
})
export class StaticKernelViewerModule {}
