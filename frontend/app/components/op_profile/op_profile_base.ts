import {Component, EventEmitter, inject, Injector, Input, OnChanges, OnDestroy, OnInit, Output, SimpleChanges} from '@angular/core';
import {Params} from '@angular/router';
import {Store} from '@ngrx/store';
import {type OpProfileProto} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {NavigationEvent} from 'org_xprof/frontend/app/common/interfaces/navigation_event';
import {DATA_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {SOURCE_CODE_SERVICE_INTERFACE_TOKEN} from 'org_xprof/frontend/app/services/source_code_service/source_code_service_interface';
import {setCurrentToolStateAction, setOpProfileRootNodeAction} from 'org_xprof/frontend/app/store/actions';
import {getActiveOpProfileNodeState} from 'org_xprof/frontend/app/store/selectors';
import {Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

import {OpProfileData, OpProfileSummary} from './op_profile_data';

/** Rules to group by. */
const GROUP_BY_RULES = ['program', 'category', 'provenance'];

/** Base class of Op Profile component. */
@Component({
  standalone: false,
  selector: 'op-profile-base',
  templateUrl: './op_profile_base.ng.html',
  styleUrls: ['./op_profile_common.scss']
})
export class OpProfileBase implements OnDestroy, OnInit, OnChanges {
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);
  private readonly injector = inject(Injector);
  private readonly dataService = inject(DATA_SERVICE_INTERFACE_TOKEN);
  profile: OpProfileProto|null = null;
  rootNode?: Node;
  data = new OpProfileData();
  groupBy = GROUP_BY_RULES[0];
  readonly GROUP_BY_RULES = GROUP_BY_RULES;
  excludeIdle = true;
  byWasted = false;
  showP90 = false;
  childrenCount = 10;
  deviceType = 'TPU';
  summary: OpProfileSummary[] = [];
  sourceCodeServiceIsAvailable = false;
  sourceFileAndLineNumber = '';
  stackTrace = '';
  focusedOpProgramId = '';
  focusedOpName = '';
  focusedOpCategory = '';
  showStackTrace = false;
  useUncappedFlops = false;

  @Input() sessionId = '';
  @Input() opProfileData: OpProfileProto|null = null;
  @Output() readonly groupByChange = new EventEmitter<string>();

  ngOnInit() {
    // We don't need the source code service to be persistently available.
    // We temporarily use the service to check if it is available and show
    // UI accordingly.
    const sourceCodeService =
        this.injector.get(SOURCE_CODE_SERVICE_INTERFACE_TOKEN, null);
    this.sourceCodeServiceIsAvailable =
        sourceCodeService?.isAvailable() === true;
  }

  processQuery(params: Params) {}
  update(event: NavigationEvent) {}
  parseData(data: OpProfileProto|null) {
    this.profile = data;
    this.updateRoot();
    this.data.update(this.rootNode, this.useUncappedFlops);
    this.summary = this.dataService.getOpProfileSummary(this.data);
  }

  constructor(
      private readonly store: Store<{}>,
  ) {
    this.store.dispatch(
        setCurrentToolStateAction({currentTool: 'hlo_op_profile'}),
    );
    this.store.select(getActiveOpProfileNodeState)
        .pipe(takeUntil(this.destroyed))
        .subscribe((node: Node|null) => {
          this.updateActiveNode(node);
        });
  }

  // Update state for source info given the active node selection in the
  // underneath table.
  private updateActiveNode(node: Node|null) {
    this.sourceFileAndLineNumber = `${node?.xla?.sourceInfo?.fileName || ''}:${
        node?.xla?.sourceInfo?.lineNumber || -1}`;
    this.stackTrace = node?.xla?.sourceInfo?.stackFrame || '';
    this.focusedOpProgramId = node?.xla?.programId || '';
    this.focusedOpName = node?.name || '';
    this.focusedOpCategory = node?.xla?.category || '';
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['opProfileData'] && this.opProfileData) {
      this.parseData(this.opProfileData);
    }
  }

  private updateRoot() {
    if (!this.profile) {
      this.rootNode = undefined;
      return;
    }

    if (this.excludeIdle) {
      if (this.groupBy === 'category') {
        this.rootNode = this.profile.byCategoryExcludeIdle;
      } else if (this.groupBy === 'provenance') {
        this.rootNode = this.profile.byProvenanceExcludeIdle;
      } else {  // 'program' is default
        this.rootNode = this.profile.byProgramExcludeIdle;
      }
    } else {
      if (this.groupBy === 'category') {
        this.rootNode = this.profile.byCategory;
      } else if (this.groupBy === 'provenance') {
        this.rootNode = this.profile.byProvenance;
      } else {  // 'program' is default
        this.rootNode = this.profile.byProgram;
      }
    }

    // Fallback if the expected data for the selected grouping is not present
    // for some reason
    if (!this.rootNode) {
      if (this.excludeIdle) {
        this.rootNode = this.profile.byProgramExcludeIdle ||
            this.profile.byCategoryExcludeIdle;
      } else {
        this.rootNode = this.profile.byProgram || this.profile.byCategory;
      }
    }

    this.deviceType = this.profile.deviceType || 'TPU';
    this.store.dispatch(
        setOpProfileRootNodeAction({rootNode: this.rootNode}),
    );
  }

  updateChildrenCount(event: Event) {
    const value = Number((event.target as HTMLInputElement).value);
    const rounded = Math.round(value / 10) * 10;

    this.childrenCount = Math.max(Math.min(rounded, 100), 10);
  }

  updateGroupBy(value: string) {
    this.groupBy = value;
    this.groupByChange.emit(value);
  }

  updateExcludeIdle() {
    this.excludeIdle = !this.excludeIdle;
    this.updateRoot();
    this.data.update(this.rootNode, this.useUncappedFlops);
  }

  updateShowStackTrace() {
    this.showStackTrace = !this.showStackTrace;
  }

  updateByWasted() {
    this.byWasted = !this.byWasted;
  }

  updateShowP90() {
    this.showP90 = !this.showP90;
  }

  updateFlopsType() {
    this.useUncappedFlops = !this.useUncappedFlops;
    this.data.update(this.rootNode, this.useUncappedFlops);
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.store.dispatch(setOpProfileRootNodeAction({rootNode: undefined}));
    this.destroyed.next();
    this.destroyed.complete();
  }
}
