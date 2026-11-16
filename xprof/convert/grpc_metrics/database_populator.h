/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XPROF_CONVERT_GRPC_METRICS_DATABASE_POPULATOR_H_
#define XPROF_CONVERT_GRPC_METRICS_DATABASE_POPULATOR_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/grpc_metrics/grpc_metrics_types.h"
#include "xprof/convert/grpc_metrics/xspace_extractor.h"
#include "plugin/xprof/protobuf/grpc_metrics.pb.h"

namespace xprof::grpc_metrics {

class DatabasePopulator {
 public:
  GrpcMetricsParseResults Populate(const RawMetricsData& data,
                                   xprof::GrpcMetricsDatabase* db);

 private:
  std::vector<FlowInfo> flows_;
  absl::flat_hash_map<std::string, std::vector<TcpMetrics>>
      endpoint_tcp_metrics_;
  absl::flat_hash_map<std::string, std::vector<const TcpMetrics*>>
      flow_to_tcp_metrics_;
};

}  // namespace xprof::grpc_metrics

#endif  // XPROF_CONVERT_GRPC_METRICS_DATABASE_POPULATOR_H_
