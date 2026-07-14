import {
  ChangeDetectionStrategy,
  Component,
  EventEmitter,
  inject,
  OnDestroy,
  Output,
} from '@angular/core';
import {ActivatedRoute, Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {OpProfileProto} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {
  DiffNode,
  OpProfileDiff,
} from 'org_xprof/frontend/app/common/interfaces/op_profile_diff';
import {setLoadingState} from 'org_xprof/frontend/app/common/utils/utils';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {BaseDiffService} from 'org_xprof/frontend/app/services/data_service_v2/diff_service';
import {setProfilingDeviceTypeAction} from 'org_xprof/frontend/app/store/actions';
import {
  Metrics,
  Node,
} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';
import {combineLatest, Observable, of, ReplaySubject} from 'rxjs';
import {combineLatestWith, map, takeUntil} from 'rxjs/operators';

const GROUP_BY_RULES = ['program', 'category', 'provenance'];

/** An op profile component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  standalone: false,
  selector: 'op-profile',
  templateUrl: './op_profile.ng.html',
  styleUrls: ['./op_profile_common.scss'],
})
export class OpProfile implements OnDestroy {
  private tool = 'hlo_op_profile';
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  /** EventEmitter that emits when data is loaded and component is ready. */
  @Output() readonly ready = new EventEmitter<void>();

  private readonly throbber = new Throbber(this.tool);
  private readonly dataService: DataServiceV2Interface = inject(
    DATA_SERVICE_INTERFACE_TOKEN,
  );
  private readonly diffService: BaseDiffService = inject(BaseDiffService);
  private readonly opProfileDataCache = new Map<string, OpProfileProto>();

  sessionId = '';
  host = '';
  baseSessionId = '';
  moduleList: string[] = [];
  opProfileData: OpProfileProto | null = null;
  groupBy = GROUP_BY_RULES[0]; // Default value

  constructor(
    route: ActivatedRoute,
    private readonly store: Store<{}>,
  ) {
    combineLatest([route.params, route.queryParams])
      .pipe(takeUntil(this.destroyed))
      .subscribe(([params, queryParams]) => {
        const oldSessionId = this.sessionId;
        const oldTool = this.tool;
        const oldHost = this.host;
        const oldBaseSessionId = this.baseSessionId;

        this.sessionId = params['sessionId'] || this.sessionId;
        this.processQueryParams(queryParams);
        this.baseSessionId = this.diffService.getBaseSessionId() || '';

        // Trigger update only if the parameters actually changed.
        const hasChanged =
          this.sessionId !== oldSessionId ||
          this.tool !== oldTool ||
          this.host !== oldHost ||
          this.baseSessionId !== oldBaseSessionId;
        if (hasChanged) {
          this.update();
        }
      });
  }

  processQueryParams(params: Params) {
    this.sessionId = params['run'] || params['sessionId'] || this.sessionId;
    this.tool = params['tag'] || this.tool;
    this.host = params['host'] || this.host;
    if (
      params['base_session_id'] !== undefined ||
      params['baseSessionID'] !== undefined
    ) {
      this.baseSessionId =
        params['base_session_id'] || params['baseSessionID'] || '';
    }
  }

  private fetchData(groupBy: string): Observable<OpProfileProto | null> {
    const baseSessionId = this.diffService.getBaseSessionId() || '';
    const cacheKey = `${this.sessionId}_${this.host}_${baseSessionId}_${groupBy}`;
    const cachedData = this.opProfileDataCache.get(cacheKey);
    if (cachedData) {
      return of(cachedData);
    }

    setLoadingState(true, this.store, 'Loading op profile data');
    this.throbber.start();

    const params = new Map<string, string>();
    params.set('group_by', groupBy);
    return this.diffService
      .getDiffData(this.sessionId, this.tool, {
        baselineSessionId: this.diffService.getBaseSessionId(),
        host: this.host,
        parameters: params,
      })
      .pipe(
        map(({active, baseline}) => {
          this.throbber.stop();
          setLoadingState(false, this.store);
          if (!baseline) {
            const opProfileData = active as OpProfileProto;
            this.opProfileDataCache.set(cacheKey, opProfileData);
            return opProfileData;
          } else if (!!active && !!baseline) {
            const opProfileDiff = this.mergeProfile(
              active as OpProfileProto,
              baseline as OpProfileProto,
            );
            this.opProfileDataCache.set(cacheKey, opProfileDiff);
            return opProfileDiff;
          }
          return null;
        }),
      );
  }

  update() {
    if (!this.sessionId || !this.tool) {
      return;
    }
    const $data = this.fetchData(this.groupBy);
    const $moduleList = this.dataService.getModuleList(this.sessionId);
    $data
      .pipe(combineLatestWith($moduleList), takeUntil(this.destroyed))
      .subscribe(([data, moduleList]) => {
        if (data) {
          this.opProfileData = data;
          this.store.dispatch(
            setProfilingDeviceTypeAction({
              deviceType: this.opProfileData.deviceType,
            }),
          );
        }
        if (moduleList) {
          this.moduleList = moduleList.split(',');
        }
        this.ready.emit();
      });
  }

  updateTable() {
    this.fetchData(this.groupBy)
      .pipe(takeUntil(this.destroyed))
      .subscribe((data) => {
        if (data) {
          this.opProfileData = data;
        }
      });
  }

  private getNodeKey(node?: Node): string {
    if (!node) return '';
    return node.name || node.xla?.expression || node.xla?.op || '';
  }

  private computeDiffScalar(a?: number, b?: number): number | undefined {
    if (a === undefined && b === undefined) return undefined;
    return (a ?? 0) - (b ?? 0);
  }

  private computeDiffArray(a?: number[], b?: number[]): number[] | undefined {
    if (a === undefined && b === undefined) return undefined;
    const maxLen = Math.max(a?.length || 0, b?.length || 0);
    const diffArr: number[] = [];
    for (let i = 0; i < maxLen; i++) {
      diffArr.push((a?.[i] ?? 0) - (b?.[i] ?? 0));
    }
    return diffArr;
  }

  private computeDiffMetrics(
    activeMetrics?: Metrics,
    baselineMetrics?: Metrics,
  ): Metrics | undefined {
    if (!activeMetrics && !baselineMetrics) {
      return undefined;
    }
    const a = activeMetrics || {};
    const b = baselineMetrics || {};

    const flops = this.computeDiffScalar(a.flops, b.flops);
    const uncappedFlops = this.computeDiffScalar(
      a.uncappedFlops,
      b.uncappedFlops,
    );
    const bf16Flops = this.computeDiffScalar(a.bf16Flops, b.bf16Flops);
    const rawTime = this.computeDiffScalar(a.rawTime, b.rawTime);
    const normalizedTimePs = this.computeDiffScalar(
      a.normalizedTimePs,
      b.normalizedTimePs,
    );
    const rawFlops = this.computeDiffScalar(a.rawFlops, b.rawFlops);
    const occurrences = this.computeDiffScalar(a.occurrences, b.occurrences);
    const avgTimePs = this.computeDiffScalar(a.avgTimePs, b.avgTimePs);

    const bandwidthUtils = this.computeDiffArray(
      a.bandwidthUtils,
      b.bandwidthUtils,
    );
    const rawBytesAccessedArray = this.computeDiffArray(
      a.rawBytesAccessedArray,
      b.rawBytesAccessedArray,
    );

    const diff: Metrics = {};
    if (flops !== undefined) diff.flops = flops;
    if (uncappedFlops !== undefined) diff.uncappedFlops = uncappedFlops;
    if (bf16Flops !== undefined) diff.bf16Flops = bf16Flops;
    if (rawTime !== undefined) diff.rawTime = rawTime;
    if (normalizedTimePs !== undefined) {
      diff.normalizedTimePs = normalizedTimePs;
    }
    if (rawFlops !== undefined) diff.rawFlops = rawFlops;
    if (occurrences !== undefined) diff.occurrences = occurrences;
    if (avgTimePs !== undefined) diff.avgTimePs = avgTimePs;
    if (bandwidthUtils !== undefined) diff.bandwidthUtils = bandwidthUtils;
    if (rawBytesAccessedArray !== undefined) {
      diff.rawBytesAccessedArray = rawBytesAccessedArray;
    }

    return diff;
  }

  mergeNode(active?: Node, baseline?: Node): DiffNode | undefined {
    if (!active && !baseline) {
      return undefined;
    }

    const activeOnly = !baseline;
    const baselineOnly = !active;
    const baseNode = active || baseline!;

    const mergedChildren: DiffNode[] = [];
    const activeChildren = active?.children || [];
    const baselineChildren = baseline?.children || [];

    const baselineMap = new Map<string, Node[]>();
    for (const child of baselineChildren) {
      const key = this.getNodeKey(child);
      if (!baselineMap.has(key)) {
        baselineMap.set(key, []);
      }
      baselineMap.get(key)!.push(child);
    }

    for (const activeChild of activeChildren) {
      const key = this.getNodeKey(activeChild);
      const baselineQueue = baselineMap.get(key);
      let matchedBaselineChild: Node | undefined = undefined;
      if (baselineQueue && baselineQueue.length > 0) {
        matchedBaselineChild = baselineQueue.shift();
      }
      const mergedChild = this.mergeNode(activeChild, matchedBaselineChild);
      if (mergedChild) {
        mergedChildren.push(mergedChild);
      }
    }

    for (const [, remainingChildren] of baselineMap) {
      for (const remainingBaselineChild of remainingChildren) {
        const mergedChild = this.mergeNode(undefined, remainingBaselineChild);
        if (mergedChild) {
          mergedChildren.push(mergedChild);
        }
      }
    }

    const diffMetrics = this.computeDiffMetrics(
      active?.metrics,
      baseline?.metrics,
    );

    const maxNumChildren = Math.max(
      active?.numChildren ?? 0,
      baseline?.numChildren ?? 0,
      mergedChildren.length,
    );

    const diffNode: DiffNode = {
      ...baseNode,
      activeOnly,
      baselineOnly,
      metrics: active?.metrics || baseline?.metrics,
      baseline: baseline ? baseline : undefined,
      diffMetrics,
      children: mergedChildren.length > 0 ? mergedChildren : undefined,
      numChildren: maxNumChildren > 0 ? maxNumChildren : undefined,
    };

    return diffNode;
  }

  mergeProfile(
    active: OpProfileProto,
    baseline: OpProfileProto,
  ): OpProfileDiff {
    const merged: OpProfileDiff = {
      deviceType: active.deviceType || baseline.deviceType,
      aggDvfsTimeScaleMultiplier:
        active.aggDvfsTimeScaleMultiplier ??
        baseline.aggDvfsTimeScaleMultiplier,
    };

    if (active.byCategory || baseline.byCategory) {
      merged.byCategory = this.mergeNode(
        active.byCategory,
        baseline.byCategory,
      );
    }
    if (active.byProgram || baseline.byProgram) {
      merged.byProgram = this.mergeNode(active.byProgram, baseline.byProgram);
    }
    if (active.byProvenance || baseline.byProvenance) {
      merged.byProvenance = this.mergeNode(
        active.byProvenance,
        baseline.byProvenance,
      );
    }
    if (active.byCategoryExcludeIdle || baseline.byCategoryExcludeIdle) {
      merged.byCategoryExcludeIdle = this.mergeNode(
        active.byCategoryExcludeIdle,
        baseline.byCategoryExcludeIdle,
      );
    }
    if (active.byProgramExcludeIdle || baseline.byProgramExcludeIdle) {
      merged.byProgramExcludeIdle = this.mergeNode(
        active.byProgramExcludeIdle,
        baseline.byProgramExcludeIdle,
      );
    }
    if (active.byProvenanceExcludeIdle || baseline.byProvenanceExcludeIdle) {
      merged.byProvenanceExcludeIdle = this.mergeNode(
        active.byProvenanceExcludeIdle,
        baseline.byProvenanceExcludeIdle,
      );
    }

    return merged;
  }

  onGroupByChange(newGroupBy: string) {
    this.groupBy = newGroupBy;
    this.updateTable();
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    setLoadingState(false, this.store);
    this.destroyed.next();
    this.destroyed.complete();
  }
}
