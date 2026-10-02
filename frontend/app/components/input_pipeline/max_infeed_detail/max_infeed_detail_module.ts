import {NgModule} from '@angular/core';
import {MatDividerModule} from '@angular/material/divider';
import {MaxInfeedDetail} from './max_infeed_detail';

@NgModule({
  imports: [MatDividerModule, MaxInfeedDetail],
  exports: [MaxInfeedDetail],
})
export class MaxInfeedDetailModule {}
