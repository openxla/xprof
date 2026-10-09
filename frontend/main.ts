// Angular 9+ using Ivy apps that potentially do i18n, even transitively, must
// import this module, which adds a global symbol at runtime.
// https://angular.io/guide/migration-localize
import '@angular/localize/init';
import {enableProdMode} from '@angular/core';
import {bootstrapApplication} from '@angular/platform-browser';

import {App} from './app/app';
import {APP_CONFIG} from './app/app_config';

enableProdMode();

bootstrapApplication(App, APP_CONFIG)
  .catch(err => console.error(err));
