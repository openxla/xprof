import {ChangeDetectionStrategy, Component, Input} from '@angular/core';

/**
 * A component to display a message with a title and content.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'message',
  templateUrl: './message.ng.html',
  styleUrls: ['./message.scss'],
  imports: [],
})
export class Message {
  @Input() title: string | undefined = undefined;
  @Input() content: string | undefined = undefined;
}
