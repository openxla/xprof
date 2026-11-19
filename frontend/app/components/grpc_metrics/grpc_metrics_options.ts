import {inject, InjectionToken} from '@angular/core';
import {
  DATA_SERVICE_INTERFACE_TOKEN,
  DataServiceV2Interface,
} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';

/** Options of the DCN Analyzer page (tool `grpc_metrics`). */
export declare interface GrpcMetricsOptions {
  /**
   * Shows the Network Recommendations tab, the insight chips of expanded rows
   * and the `insight:` search prefix.
   */
  enableRecommendations: boolean;
}

/** Options of the DCN Analyzer page. Recommendations are off by default. */
export const GRPC_METRICS_OPTIONS = new InjectionToken<GrpcMetricsOptions>(
  'GrpcMetricsOptions',
  {
    providedIn: 'root',
    factory: () => ({enableRecommendations: false}),
  },
);

/**
 * URL query parameter that turns recommendations on for testing (`true`).
 * It only applies where they are off by default; it never turns them off.
 */
export const RECOMMENDATIONS_QUERY_PARAM = 'dcn_analyzer_recommendations';

/**
 * Data service of the DCN Analyzer page. Defaults to the app's
 * DATA_SERVICE_INTERFACE_TOKEN; an app can provide another instance.
 */
export const GRPC_METRICS_DATA_SERVICE =
  new InjectionToken<DataServiceV2Interface>('GrpcMetricsDataService', {
    providedIn: 'root',
    factory: () => inject(DATA_SERVICE_INTERFACE_TOKEN),
  });

/**
 * Returns whether recommendations are shown. When they are on by default, they
 * stay on. When they are off by default, the query parameter value `true` turns
 * them on; any other value is ignored.
 */
export function resolveEnableRecommendations(
  enabledByDefault: boolean,
  queryParamValue: string | null | undefined,
): boolean {
  return enabledByDefault || queryParamValue === 'true';
}
