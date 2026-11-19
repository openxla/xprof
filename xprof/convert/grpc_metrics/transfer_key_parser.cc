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

#include "xprof/convert/grpc_metrics/transfer_key_parser.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"

namespace xprof::grpc_metrics {
namespace {

// Separators of the transfer key grammar.
constexpr absl::string_view kEndpointGroupSeparator = "|s";
constexpr absl::string_view kDestinationSeparator = "->";
constexpr absl::string_view kEndpointGroupTerminator = "d|";
constexpr absl::string_view kChunkSeparator = "$c";
constexpr absl::string_view kLaunchIdSeparator = "$l";
constexpr absl::string_view kLoopIndexSeparator = "^i";

// A MegaScale device id is (slice id + 1) * kSliceIdOffset + device id within
// the slice.
constexpr int64_t kSliceIdOffset = 100000;

// Slice id of the local slice. It is only valid with device id 0.
constexpr int32_t kLocalSliceId = std::numeric_limits<int32_t>::max();

// Returns true if `label` contains one of the separators, which a label may
// not contain.
bool ContainsSeparator(absl::string_view label) {
  for (const absl::string_view separator :
       {kEndpointGroupSeparator, kDestinationSeparator,
        kEndpointGroupTerminator, kLaunchIdSeparator, kChunkSeparator,
        kLoopIndexSeparator}) {
    if (absl::StrContains(label, separator)) return true;
  }
  return false;
}

// Returns true if `label` is a sequence of characters other than `~`. A
// character is an ASCII byte, or a lead byte in [0xC2, 0xDF], [0xE0, 0xEF] or
// [0xF0, 0xF4] followed by one, two or three bytes in [0x80, 0xBF]. This is the
// set of byte sequences that the MegaScale key parser accepts as a label; like
// it, this also accepts overlong three and four byte forms, surrogates and code
// points above U+10FFFF.
bool IsValidLabelText(absl::string_view label) {
  size_t position = 0;
  while (position < label.size()) {
    const unsigned char lead_byte = static_cast<unsigned char>(label[position]);
    size_t continuation_byte_count = 0;
    if (lead_byte < 0x80) {
      if (lead_byte == '~') return false;
    } else if (lead_byte >= 0xC2 && lead_byte <= 0xDF) {
      continuation_byte_count = 1;
    } else if (lead_byte >= 0xE0 && lead_byte <= 0xEF) {
      continuation_byte_count = 2;
    } else if (lead_byte >= 0xF0 && lead_byte <= 0xF4) {
      continuation_byte_count = 3;
    } else {
      return false;
    }
    if (label.size() - position - 1 < continuation_byte_count) return false;
    for (size_t offset = 1; offset <= continuation_byte_count; ++offset) {
      const unsigned char continuation_byte =
          static_cast<unsigned char>(label[position + offset]);
      if (continuation_byte < 0x80 || continuation_byte > 0xBF) return false;
    }
    position += 1 + continuation_byte_count;
  }
  return true;
}

// Removes the leading ASCII digits from `*text` and returns them. Returns an
// empty view if `*text` does not start with a digit.
absl::string_view ConsumeDigits(absl::string_view* text) {
  size_t digit_count = 0;
  while (
      digit_count < text->size() &&
      absl::ascii_isdigit(static_cast<unsigned char>((*text)[digit_count]))) {
    ++digit_count;
  }
  const absl::string_view digits = text->substr(0, digit_count);
  text->remove_prefix(digit_count);
  return digits;
}

// Removes a non-empty run of digits from the start of `*text` and stores its
// value in `*value`. Returns false if there are no digits or the value does
// not fit in 64 bits.
bool ConsumeDeviceId(absl::string_view* text, int64_t* value) {
  const absl::string_view digits = ConsumeDigits(text);
  return !digits.empty() && absl::SimpleAtoi(digits, value);
}

// If `*text` starts with `separator`, removes the separator and the number
// after it (digits, with an optional leading `-` when `allow_negative` is
// true) and stores the number in `*value`. Returns false if the separator is
// not followed by a number that fits in 32 bits. Returns true and leaves
// `*text` and `*value` unchanged if `*text` does not start with `separator`.
bool ConsumeOptionalPart(absl::string_view* text, absl::string_view separator,
                         bool allow_negative, std::optional<int32_t>* value) {
  if (!absl::ConsumePrefix(text, separator)) return true;
  const absl::string_view number_start = *text;
  if (allow_negative) absl::ConsumePrefix(text, "-");
  if (ConsumeDigits(text).empty()) return false;
  const absl::string_view number =
      number_start.substr(0, number_start.size() - text->size());
  int32_t parsed_number = 0;
  if (!absl::SimpleAtoi(number, &parsed_number)) return false;
  *value = parsed_number;
  return true;
}

// Returns the MegaScale device id for a device id read from a key, or
// std::nullopt if it is not valid. This follows the MegaScale runtime: ids
// below kSliceIdOffset are rejected, the slice id is computed in 64 bits and
// then truncated to 32 bits, negative slice ids are rejected, the local slice
// is only valid with device id 0, and the id is rebuilt from the slice id and
// the device id within the slice.
std::optional<int64_t> ToMegascaleDeviceId(int64_t raw_device_id) {
  if (raw_device_id < kSliceIdOffset) return std::nullopt;
  const int32_t slice_id =
      static_cast<int32_t>(raw_device_id / kSliceIdOffset - 1);
  const int32_t device_id_in_slice =
      static_cast<int32_t>(raw_device_id % kSliceIdOffset);
  if (slice_id < 0) return std::nullopt;
  if (slice_id == kLocalSliceId && device_id_in_slice != 0) {
    return std::nullopt;
  }
  return (static_cast<int64_t>(slice_id) + 1) * kSliceIdOffset +
         device_id_in_slice;
}

}  // namespace

std::optional<TransferKeyFields> ParseTransferKey(absl::string_view key) {
  // A label cannot contain a separator, so the first `|s` always ends the
  // label of a valid key.
  const size_t label_end = key.find(kEndpointGroupSeparator);
  if (label_end == absl::string_view::npos || label_end == 0) {
    return std::nullopt;
  }
  const absl::string_view label = key.substr(0, label_end);
  if (!IsValidLabelText(label) || ContainsSeparator(label)) {
    return std::nullopt;
  }

  // The source and destination device ids.
  absl::string_view rest =
      key.substr(label_end + kEndpointGroupSeparator.size());
  int64_t raw_source_id = 0;
  int64_t raw_destination_id = 0;
  if (!ConsumeDeviceId(&rest, &raw_source_id) ||
      !absl::ConsumePrefix(&rest, kDestinationSeparator) ||
      !ConsumeDeviceId(&rest, &raw_destination_id) ||
      !absl::ConsumePrefix(&rest, kEndpointGroupTerminator)) {
    return std::nullopt;
  }

  // The optional parts, in this order, and nothing after them.
  TransferKeyFields fields;
  if (!ConsumeOptionalPart(&rest, kChunkSeparator, /*allow_negative=*/false,
                           &fields.chunk) ||
      !ConsumeOptionalPart(&rest, kLaunchIdSeparator, /*allow_negative=*/true,
                           &fields.launch_id) ||
      !ConsumeOptionalPart(&rest, kLoopIndexSeparator,
                           /*allow_negative=*/false, &fields.loop_index) ||
      !rest.empty()) {
    return std::nullopt;
  }

  const std::optional<int64_t> source_megascale_id =
      ToMegascaleDeviceId(raw_source_id);
  const std::optional<int64_t> destination_megascale_id =
      ToMegascaleDeviceId(raw_destination_id);
  if (!source_megascale_id.has_value() ||
      !destination_megascale_id.has_value()) {
    return std::nullopt;
  }

  fields.label = std::string(label);
  fields.source_megascale_id = *source_megascale_id;
  fields.destination_megascale_id = *destination_megascale_id;
  return fields;
}

}  // namespace xprof::grpc_metrics
