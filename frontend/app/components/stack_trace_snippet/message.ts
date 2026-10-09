import {CommonModule} from '@angular/common';
import {ChangeDetectionStrategy, Component, input} from '@angular/core';

/**
 * A component to display a message with a title and content.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.OnPush,
  standalone: true,
  selector: 'message',
  templateUrl: './message.ng.html',
  styleUrls: ['./message.scss'],
  imports: [CommonModule],
})
export class Message {
  readonly title = input<string>();
  readonly content = input<string>();
}
