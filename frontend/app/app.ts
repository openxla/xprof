import {Component, OnInit} from '@angular/core';
import {Store} from '@ngrx/store';
import {RunToolsMap} from 'org_xprof/frontend/app/common/interfaces/tool';
import {DataDispatcher} from 'org_xprof/frontend/app/services/data_dispatcher/data_dispatcher';
import {DataServiceV2} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2';
import * as actions from 'org_xprof/frontend/app/store/actions';
import {firstValueFrom} from 'rxjs';

/** The root component. */
@Component({
  standalone: false,
  selector: 'app',
  templateUrl: './app.ng.html',
  styleUrls: ['./app.css'],
})
export class App implements OnInit {
  loading = true;
  dataFound = false;

  constructor(
      // tslint:disable-next-line:no-unused-variable declare to instantiate
      private readonly dataDispatcher: DataDispatcher,
      private readonly dataService: DataServiceV2,
      private readonly store: Store<{}>) {
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
    const runs = await firstValueFrom(this.dataService.getRuns()) as string[];
    if (runs.length === 0) {
      this.loading = false;
      return;
    }

    let filteredRuns = runs;

    const config = await firstValueFrom(this.dataService.getConfig());
    if (config?.filterSessions) {
      try {
        const regex = new RegExp(config.filterSessions, 'i');
        filteredRuns = runs.filter(run => regex.test(run));
      } catch (e) {
        console.error('Invalid regex for session filter:', e);
        // fallback to no filtering if regex is invalid
        filteredRuns = runs;
      }

      // fallback to no filtering if regex finds nothing
      if (filteredRuns.length === 0) {
        filteredRuns = runs;
      }
    }

    this.dataFound = true;
    this.store.dispatch(actions.setCurrentRunAction({currentRun: filteredRuns[0]}));
    const tools =
        await firstValueFrom(this.dataService.getRunTools(filteredRuns[0])) as string[];
    const runToolsMap: RunToolsMap = {[filteredRuns[0]]: tools};
    for (let i = 1; i < runs.length; i++) {
      runToolsMap[runs[i]] = [];
    }
    this.store.dispatch(actions.setRunToolsMapAction({
      runToolsMap,
    }));
    this.loading = false;
  }
}
