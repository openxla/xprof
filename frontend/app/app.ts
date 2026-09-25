import {NgIf} from '@angular/common';
import {
  ChangeDetectionStrategy,
  Component,
  inject,
  OnInit,
} from '@angular/core';
import {MatProgressBar} from '@angular/material/progress-bar';
import {Store} from '@ngrx/store';
import {RunToolsMap} from 'org_xprof/frontend/app/common/interfaces/tool';
import {EmptyPage} from 'org_xprof/frontend/app/components/empty_page/empty_page';
import {MainPage} from 'org_xprof/frontend/app/components/main_page/main_page';
import {DataDispatcher} from 'org_xprof/frontend/app/services/data_dispatcher/data_dispatcher';
import {DataServiceV2} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2';
import * as actions from 'org_xprof/frontend/app/store/actions';
import {firstValueFrom} from 'rxjs';
/** The root component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'app',
  templateUrl: './app.ng.html',
  styleUrls: ['./app.scss'],
  imports: [NgIf, MatProgressBar, EmptyPage, MainPage],
})
export class App implements OnInit {
  loading = true;
  dataFound = false;

  private readonly dataService = inject(DataServiceV2);
  private readonly store: Store<{}> = inject(Store);

  constructor() {
    inject(DataDispatcher);
    document.addEventListener('tensorboard-reload', () => {
      if (!this.loading) {
        this.initRunsAndTools();
      }
    });
  }

  ngOnInit() {
    this.initRunsAndTools();
  }

  async initRunsAndTools() {
    this.loading = true;
    const runs = (await firstValueFrom(this.dataService.getRuns())) as string[];
    if (runs.length === 0) {
      this.loading = false;
      return;
    }
    this.dataFound = true;
    this.store.dispatch(actions.setCurrentRunAction({currentRun: runs[0]}));
    const tools = (await firstValueFrom(
      this.dataService.getRunTools(runs[0]),
    )) as string[];
    const runToolsMap: RunToolsMap = {[runs[0]]: tools};
    for (let i = 1; i < runs.length; i++) {
      runToolsMap[runs[i]] = [];
    }
    this.store.dispatch(
      actions.setRunToolsMapAction({
        runToolsMap,
      }),
    );
    this.loading = false;
  }
}
