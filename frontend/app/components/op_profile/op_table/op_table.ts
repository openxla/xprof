import {CommonModule} from '@angular/common';
import {Component, Input, OnDestroy} from '@angular/core';
import {MatTooltipModule} from '@angular/material/tooltip';
import {OpTableEntry} from 'org_xprof/frontend/app/components/op_profile/op_table_entry/op_table_entry';
import {Store} from '@ngrx/store';
import {type Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';
import {setActiveOpProfileNodeAction} from 'org_xprof/frontend/app/store/actions';

/** An op table view component. */
@Component({
  standalone: true,
  selector: 'op-table',
  templateUrl: 'op_table.ng.html',
  styleUrls: ['op_table.scss'],
  imports: [CommonModule, MatTooltipModule, OpTableEntry],
})
export class OpTable implements OnDestroy {
  /** The root node. */
  @Input() rootNode?: Node;

  /** The property to sort by wasted time. */
  @Input() byWasted = false;

  /** The property to show top 90%. */
  @Input() showP90 = false;

  /** The number of children nodes to be shown. */
  @Input() childrenCount = 10;

  selectedNode?: Node;

  constructor(private readonly store: Store<{}>) {}

  updateSelected(node?: Node) {
    this.selectedNode = node;
  }

  ngOnDestroy() {
    this.store.dispatch(
        setActiveOpProfileNodeAction({activeOpProfileNode: null}));
  }

  updateActive(node: Node|null) {
    this.store.dispatch(setActiveOpProfileNodeAction(
        {activeOpProfileNode: node || this.selectedNode || null}));
  }
}
