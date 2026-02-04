import {Component, Input} from '@angular/core';

/** A model properties view component. */
@Component({
  standalone: true,
  selector: 'model-properties',
  templateUrl: 'model_properties.ng.html',
  styleUrls: ['model_properties.scss'],
  imports: [],
})
export class ModelProperties {
  /** The architecture of a model. */
  @Input() architecture = '';

  /** The task of a model. */
  @Input() task = '';
}
