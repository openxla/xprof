import {
  ChangeDetectionStrategy,
  Component,
  OnDestroy,
  inject,
  input,
} from '@angular/core';
import {MatTooltip} from '@angular/material/tooltip';
import {Store} from '@ngrx/store';
import {setActiveOpProfileNodeAction} from 'org_xprof/frontend/app/store/actions';
import {type Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';
import {OpTableEntry} from '../op_table_entry/op_table_entry';

/** An op table view component. */
@Component({
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  selector: 'op-table',
  templateUrl: './op_table.ng.html',
  styleUrls: ['./op_table.scss'],
  imports: [MatTooltip, OpTableEntry],
})
export class OpTable implements OnDestroy {
  /** The root node. */
  readonly rootNode = input<Node>();

  /** The property to sort by wasted time. */
  readonly byWasted = input(false);

  /** The property to show top 90%. */
  readonly showP90 = input(false);

  /** The number of children nodes to be shown. */
  readonly childrenCount = input(10);

  selectedNode?: Node;

  private readonly store = inject<Store<{}>>(Store);

  updateSelected(node?: Node) {
    this.selectedNode = node;
  }

  ngOnDestroy() {
    this.store.dispatch(
      setActiveOpProfileNodeAction({activeOpProfileNode: null}),
    );
  }

  updateActive(node?: Node) {
    this.store.dispatch(
      setActiveOpProfileNodeAction({
        activeOpProfileNode: node || this.selectedNode || null,
      }),
    );
  }
}
