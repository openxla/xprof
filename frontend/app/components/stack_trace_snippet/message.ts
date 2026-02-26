import {CommonModule} from '@angular/common';
import {Component, Input, ChangeDetectionStrategy} from '@angular/core';

/**
 * A component to display a message with a title and content.
 */
@Component({
  changeDetection: ChangeDetectionStrategy.Eager,standalone: true,
  selector: 'message',
  templateUrl: './message.ng.html',
  styleUrls: ['./message.scss'],
  imports: [CommonModule],
})
export class Message {
  @Input() title: string|undefined = undefined;
  @Input() content: string|undefined = undefined;
}
