import {CommonModule} from '@angular/common';
import {NgModule} from '@angular/core';
import {SourceMapperModule} from 'org_xprof/frontend/app/components/source_mapper/source_mapper_module';

import {StackTracePage} from './stack_trace_page';

/** The stack trace page module. */
@NgModule({
  imports: [CommonModule, SourceMapperModule, StackTracePage],
  exports: [StackTracePage],
})
export class StackTracePageModule {}
