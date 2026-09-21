import {ChangeDetectionStrategy, Component, input} from '@angular/core';

/** A model properties view component. */
@Component({
  changeDetection: ChangeDetectionStrategy.Default,
  selector: 'model-properties',
  templateUrl: './model_properties.ng.html',
  styleUrls: ['./model_properties.scss'],
})
export class ModelProperties {
  /** The architecture of a model. */
  readonly architecture = input('');

  /** The task of a model. */
  readonly task = input('');
}
