import {Component, Input, ChangeDetectionStrategy} from '@angular/core';

/** A model properties view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Eager,standalone: false,
  selector: 'model-properties',
  templateUrl: './model_properties.ng.html',
  styleUrls: ['./model_properties.scss']
})
export class ModelProperties {
  /** The architecture of a model. */
  @Input() architecture: string = '';

  /** The task of a model. */
  @Input() task: string = '';
}
