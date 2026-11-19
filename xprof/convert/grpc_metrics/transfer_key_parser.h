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

#ifndef XPROF_CONVERT_GRPC_METRICS_TRANSFER_KEY_PARSER_H_
#define XPROF_CONVERT_GRPC_METRICS_TRANSFER_KEY_PARSER_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/strings/string_view.h"

namespace xprof::grpc_metrics {

// Fields of a MegaScale transfer key, the `key` stat of the
// `Megascale_gRPC:SendRpc` trace events. A key has the form
// `<label>|s<source>-><destination>d|`, optionally followed by a chunk index
// (`$c<chunk>`), a launch id (`$l<launch_id>`, may be negative) and a loop
// index (`^i<loop_index>`), in that order. Example:
// `all-reduce.1|s100001->200003d|$c5$l3^i4`.
struct TransferKeyFields {
  // Name of the collective: the part of the key before `|s`.
  std::string label;
  // MegaScale device ids of the source and destination endpoints. A MegaScale
  // device id is (slice id + 1) * 100000 + device id within the slice.
  int64_t source_megascale_id = 0;
  int64_t destination_megascale_id = 0;
  // Optional parts of the key. Unset when the key does not have them.
  std::optional<int32_t> chunk;
  std::optional<int32_t> launch_id;
  std::optional<int32_t> loop_index;
};

// Parses a MegaScale transfer key. Returns std::nullopt if `key` is not a
// valid transfer key: for example when the label is empty, contains `~`,
// contains bytes that are not UTF-8 characters or contains one of the
// separators (`|s`, `->`, `d|`, `$c`, `$l`, `^i`), when a number does not fit
// its type, or when a device id is not a valid MegaScale device id.
std::optional<TransferKeyFields> ParseTransferKey(absl::string_view key);

}  // namespace xprof::grpc_metrics

#endif  // XPROF_CONVERT_GRPC_METRICS_TRANSFER_KEY_PARSER_H_
