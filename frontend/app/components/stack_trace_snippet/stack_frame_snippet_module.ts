import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatTooltipModule} from '@angular/material/tooltip';
import {SourceCodeService} from 'org_xprof/frontend/app/services/source_code_service/source_code_service';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';

import {Message} from './message';
import {StackFrameSnippet} from './stack_frame_snippet';

/** A module to show code snippets for a stack frame. */
@NgModule({
  declarations: [StackFrameSnippet],
  exports: [StackFrameSnippet],
  imports: [
    CommonModule, MatExpansionModule, MatIconModule, MatTooltipModule,
    MatProgressBarModule, Message
  ],
  providers: [
    {provide: SOURCE_CODE_SERVICE_INTERFACE_TOKEN, useClass: SourceCodeService}
  ]
})
export class StackFrameSnippetModule {
}
