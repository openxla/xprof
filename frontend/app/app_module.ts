import {HttpClientModule} from '@angular/common/http';
import {NgModule} from '@angular/core';
import {MatProgressBarModule} from '@angular/material/progress-bar';
import {BrowserModule} from '@angular/platform-browser';
import {BrowserAnimationsModule} from '@angular/platform-browser/animations';
import {EmptyPageModule} from 'org_xprof/frontend/app/components/empty_page/empty_page_module';
import {MainPageModule} from 'org_xprof/frontend/app/components/main_page/main_page_module';
import {PipesModule} from 'org_xprof/frontend/app/pipes/pipes_module';
import {DataDispatcher} from 'org_xprof/frontend/app/services/data_dispatcher/data_dispatcher';
import {DataServiceV2} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {SourceCodeService} from 'org_xprof/frontend/app/services/source_code_service/source_code_service';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {RootStoreModule} from 'org_xprof/frontend/app/store/store_module';

import {App} from './app';

/** The root component module. */
@NgModule({
  declarations: [App],
  imports: [
    BrowserModule,
    HttpClientModule,
    MatProgressBarModule,
    EmptyPageModule,
    MainPageModule,
    BrowserAnimationsModule,
    PipesModule,
    RootStoreModule,
  ],
  providers: [
    DataDispatcher,
    DataServiceV2,
    {provide: DATA_SERVICE_INTERFACE_TOKEN, useClass: DataServiceV2},
    {
      provide: SOURCE_CODE_SERVICE_INTERFACE_TOKEN,
      useClass: SourceCodeService,
    },
  ],
  bootstrap: [App],
})
export class AppModule {
}
