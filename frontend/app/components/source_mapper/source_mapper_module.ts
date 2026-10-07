import {NgModule} from '@angular/core';
import {SourceMapper} from './source_mapper';

/** @deprecated Import the standalone SourceMapper component directly instead. */
@NgModule({
  imports: [SourceMapper],
  exports: [SourceMapper],
})
export class SourceMapperModule {}
