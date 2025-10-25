import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {FormsModule} from '@angular/forms';
import {MatExpansionModule} from '@angular/material/expansion';
import {MatFormFieldModule} from '@angular/material/form-field';
import {MatIconModule} from '@angular/material/icon';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {MatSelectModule} from '@angular/material/select';
import {MatTooltipModule} from '@angular/material/tooltip';
import {Message} from 'org_xprof/frontend/app/components/stack_trace_snippet/message';
import {StackTraceSnippetModule} from 'org_xprof/frontend/app/components/stack_trace_snippet/stack_trace_snippet_module';

import {SourceMapper} from './source_mapper';

@NgModule({
  declarations: [SourceMapper],
  imports: [
    CommonModule,
    FormsModule,
    StackTraceSnippetModule,
    MatExpansionModule,
    MatFormFieldModule,
    MatIconModule,
    MatSelectModule,
    MatTooltipModule,
    Message,
    MatProgressBarModule,
  ],
  exports: [SourceMapper],
})
export class SourceMapperModule {
}
