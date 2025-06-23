import {CommonModule} from '@angular/common';
import {Component, Input} from '@angular/core';

/**
 * A component to display a message with a title and content.
 */
@Component({
  standalone: true,
  selector: 'message',
  templateUrl: './message.ng.html',
  styleUrls: ['./message.scss'],
  imports: [CommonModule],
})
export class Message {
  @Input() title: string|undefined = undefined;
  @Input() content: string|undefined = undefined;
}
