// Angular 9+ using Ivy apps that potentially do i18n, even transitively, must
// import this module, which adds a global symbol at runtime.
// https://angular.io/guide/migration-localize
import '@angular/localize/init';
import {enableProdMode} from '@angular/core';
import {platformBrowserDynamic} from '@angular/platform-browser-dynamic';

import {AppModule} from './app/app_module';

enableProdMode();

platformBrowserDynamic().bootstrapModule(AppModule)
  .catch(err => console.error(err));
