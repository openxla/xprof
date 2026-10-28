import {Component, Input, OnDestroy, ChangeDetectionStrategy} from '@angular/core';
import {Store} from '@ngrx/store';
import {setActiveOpProfileNodeAction} from 'org_xprof/frontend/app/store/actions';
import {getTagsState} from 'org_xprof/frontend/app/store/selectors';
import {type Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';

/** An op table view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,standalone: false,
  selector: 'op-table',
  templateUrl: './op_table.ng.html',
  styleUrls: ['./op_table.scss']
})
export class OpTable implements OnDestroy {
  private readonly destroyed = new ReplaySubject<void>(1);

  /** The root node. */
  @Input() rootNode?: Node;

  /** The session id. */
  @Input() sessionId = '';

  /** The list of modules. */
  @Input() moduleList: string[] = [];

  /** The property to sort by wasted time. */
  @Input() byWasted: boolean = false;

  /** The property to show top 90%. */
  @Input() showP90: boolean = false;

  /** The number of children nodes to be shown. */
  @Input() childrenCount: number = 10;

  selectedNode?: Node;
  tags: string[] = [];

  constructor(private readonly store: Store<{}>) {
    this.store
        .select(getTagsState)
        .pipe(takeUntil(this.destroyed))
        .subscribe((tags: string[]) => {
          this.tags = tags ?? [];
        });
  }

  updateSelected(node?: Node) {
    this.selectedNode = node;
  }

  ngOnDestroy() {
    this.store.dispatch(
        setActiveOpProfileNodeAction({activeOpProfileNode: null}));
    this.destroyed.next();
    this.destroyed.complete();
  }

  updateActive(node: Node|null) {
    this.store.dispatch(setActiveOpProfileNodeAction(
        {activeOpProfileNode: node || this.selectedNode || null}));
  }
}
