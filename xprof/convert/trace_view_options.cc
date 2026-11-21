/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/trace_view_options.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xprof/convert/tool_options.h"
#include "xprof/convert/trace_viewer/trace_events_filter_interface.h"
#include "xprof/convert/trace_viewer/trace_options.h"
#include "xprof/convert/trace_viewer/trace_viewer_visibility.h"
#include "xprof/convert/xplane_to_trace_container.h"

namespace tensorflow {
namespace profiler {
namespace {

constexpr double kPicosecondsPerMillisecond = 1E9;
// `uint64_t::max()` and `int64_t::max()` round up to 2^64 and 2^63 in IEEE-754
// double, so strict `<` rejects values that would overflow the target integer
// type when cast.
constexpr double kUint64Limit =
    static_cast<double>(std::numeric_limits<uint64_t>::max());
constexpr double kInt64Limit =
    static_cast<double>(std::numeric_limits<int64_t>::max());

// Returns true if `value` is finite, non-negative, and fits in `uint64_t`.
bool IsValidUint64Double(double value) {
  return std::isfinite(value) && value >= 0.0 && value < kUint64Limit;
}

// Returns true if `time_ms` is finite, non-negative, and converts to
// picoseconds strictly below `max_ps_exclusive`.
bool IsValidTraceViewTimeMs(double time_ms,
                            double max_ps_exclusive = kUint64Limit) {
  if (!std::isfinite(time_ms) || time_ms < 0.0) {
    return false;
  }
  const double time_ps = time_ms * kPicosecondsPerMillisecond;
  return std::isfinite(time_ps) && time_ps < max_ps_exclusive;
}

// Converts a millisecond value to signed `int64_t` picoseconds using
// `std::round`. Rounding is preserved for exact single-event lookups
// (`event_name` path) so floating-point milliseconds round-tripped from the UI
// match exact picosecond LevelDB keys without off-by-one truncation drift.
absl::StatusOr<int64_t> CheckedRoundedMilliToInt64Pico(
    double time_ms, absl::string_view option_name) {
  if (!IsValidTraceViewTimeMs(time_ms, kInt64Limit)) {
    return absl::InvalidArgumentError(absl::StrCat(
        option_name,
        " must be finite, non-negative, and fit in int64_t picoseconds"));
  }
  return static_cast<int64_t>(std::round(time_ms * kPicosecondsPerMillisecond));
}

// Converts a millisecond time-window bound to `uint64_t` picoseconds via
// `tsl::profiler::MilliToPico` truncation, matching existing `MilliSpan`
// window-bound semantics.
absl::StatusOr<uint64_t> CheckedMilliToUint64Pico(
    double time_ms, absl::string_view option_name) {
  if (!IsValidTraceViewTimeMs(time_ms, kUint64Limit)) {
    return absl::InvalidArgumentError(absl::StrCat(
        option_name,
        " must be finite, non-negative, and fit in uint64_t picoseconds"));
  }
  return tsl::profiler::MilliToPico(time_ms);
}

struct EventLookupKey {
  int64_t start_ps = 0;
  int64_t duration_ps = 0;
  int64_t unique_id = 0;
};

// Validates single-event lookup parameters and converts them into the signed
// picosecond key expected by LevelDB lookups.
absl::StatusOr<EventLookupKey> BuildEventLookupKey(
    const TraceViewOption& trace_option) {
  absl::StatusOr<int64_t> start_ps = CheckedRoundedMilliToInt64Pico(
      trace_option.start_time_ms, "start_time_ms");
  if (!start_ps.ok()) {
    return start_ps.status();
  }
  absl::StatusOr<int64_t> duration_ps =
      CheckedRoundedMilliToInt64Pico(trace_option.duration_ms, "duration_ms");
  if (!duration_ps.ok()) {
    return duration_ps.status();
  }
  if (trace_option.unique_id >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return absl::InvalidArgumentError(
        "unique_id must fit in non-negative int64_t");
  }
  return EventLookupKey{
      .start_ps = *start_ps,
      .duration_ps = *duration_ps,
      .unique_id = static_cast<int64_t>(trace_option.unique_id),
  };
}

// Validates time-window bounds and builds the corresponding visibility window.
// Valid inverted bounds (`start_time_ms > *end_time_ms`, including
// sub-picosecond inversions prior to `MilliToPico` truncation) normalize to
// `TraceViewerVisibility::Window::Empty()`.
absl::StatusOr<TraceViewerVisibility::Window> BuildVisibilityWindow(
    const TraceViewOption& trace_option) {
  absl::StatusOr<uint64_t> start_ps =
      CheckedMilliToUint64Pico(trace_option.start_time_ms, "start_time_ms");
  if (!start_ps.ok()) {
    return start_ps.status();
  }
  std::optional<uint64_t> end_ps;
  if (trace_option.end_time_ms.has_value()) {
    absl::StatusOr<uint64_t> parsed_end_ps =
        CheckedMilliToUint64Pico(*trace_option.end_time_ms, "end_time_ms");
    if (!parsed_end_ps.ok()) {
      return parsed_end_ps.status();
    }
    if (trace_option.start_time_ms > *trace_option.end_time_ms) {
      return TraceViewerVisibility::Window::Empty();
    }
    end_ps = *parsed_end_ps;
  }
  return TraceViewerVisibility::Window::FromEndPoints(*start_ps, end_ps);
}

// Parses `option_name` from `options` as a finite, non-negative millisecond
// time that fits in `uint64_t` picoseconds. Returns `std::nullopt` when
// `option_name` is absent.
absl::StatusOr<std::optional<double>> ParseOptionalTimeMsOption(
    const ToolOptions& options, absl::string_view option_name) {
  std::optional<std::string> raw_value =
      GetParam<std::string>(options, option_name);
  if (!raw_value.has_value()) {
    return std::nullopt;
  }
  double parsed = 0.0;
  if (!absl::SimpleAtod(*raw_value, &parsed)) {
    return absl::InvalidArgumentError(
        absl::StrCat(option_name, " must be a number"));
  }
  TF_RETURN_IF_ERROR(CheckedMilliToUint64Pico(parsed, option_name).status());
  return parsed;
}

enum class FractionPolicy { kAllowTruncate, kRequireIntegral };

// Parses a non-negative decimal string (optionally using fractional or
// scientific notation such as `"42.0"`, `"+42.0"`, `"1.5e1"`, or `"100e-2"`)
// directly into an exact `uint64_t`, rejecting negative signs (including
// `"-0.0"`), non-decimal literals (such as hex floats), non-zero fractional
// values, and values exceeding `uint64_t::max()`.
absl::StatusOr<uint64_t> ParseExactIntegralUint64(
    absl::string_view raw_value, absl::string_view option_name) {
  auto invalid = [&]() {
    return absl::InvalidArgumentError(absl::StrCat(
        option_name,
        " must be an exact non-negative integer within uint64_t"));
  };

  const absl::string_view trimmed = absl::StripAsciiWhitespace(raw_value);
  if (trimmed.empty()) {
    return invalid();
  }

  size_t pos = 0;
  if (trimmed[pos] == '+') {
    ++pos;
  }

  constexpr size_t kMaxUint64Digits = 20;
  std::string sig_digits;
  bool has_any_digit = false;
  bool seen_nonzero = false;
  int64_t trailing_zeros = 0;
  int64_t frac_digits = 0;

  auto consume_digit = [&](char c, bool is_frac) {
    has_any_digit = true;
    if (is_frac) {
      ++frac_digits;
    }
    if (c == '0') {
      if (seen_nonzero) {
        ++trailing_zeros;
      }
      return;
    }
    if (seen_nonzero && trailing_zeros > 0) {
      if (sig_digits.size() <= kMaxUint64Digits) {
        const size_t zeros_to_append =
            std::min<size_t>(static_cast<size_t>(trailing_zeros),
                             kMaxUint64Digits + 1 - sig_digits.size());
        sig_digits.append(zeros_to_append, '0');
      }
      trailing_zeros = 0;
    }
    seen_nonzero = true;
    if (sig_digits.size() <= kMaxUint64Digits) {
      sig_digits.push_back(c);
    }
  };

  while (pos < trimmed.size() &&
         absl::ascii_isdigit(static_cast<unsigned char>(trimmed[pos]))) {
    consume_digit(trimmed[pos], /*is_frac=*/false);
    ++pos;
  }
  if (pos < trimmed.size() && trimmed[pos] == '.') {
    ++pos;
    while (pos < trimmed.size() &&
           absl::ascii_isdigit(static_cast<unsigned char>(trimmed[pos]))) {
      consume_digit(trimmed[pos], /*is_frac=*/true);
      ++pos;
    }
  }
  if (!has_any_digit) {
    return invalid();
  }

  int64_t exp = 0;
  if (pos < trimmed.size() && (trimmed[pos] == 'e' || trimmed[pos] == 'E')) {
    ++pos;
    bool exp_negative = false;
    if (pos < trimmed.size() && (trimmed[pos] == '+' || trimmed[pos] == '-')) {
      exp_negative = (trimmed[pos] == '-');
      ++pos;
    }
    if (pos >= trimmed.size() ||
        !absl::ascii_isdigit(static_cast<unsigned char>(trimmed[pos]))) {
      return invalid();
    }
    constexpr int64_t kExponentSaturationMargin = 100;
    const int64_t exp_limit =
        static_cast<int64_t>(trimmed.size()) + kExponentSaturationMargin;
    int64_t exp_mag = 0;
    while (pos < trimmed.size() &&
           absl::ascii_isdigit(static_cast<unsigned char>(trimmed[pos]))) {
      const int digit = trimmed[pos] - '0';
      if (exp_mag > (exp_limit - digit) / 10) {
        exp_mag = exp_limit;
      } else {
        exp_mag = exp_mag * 10 + digit;
      }
      ++pos;
    }
    exp = exp_negative ? -exp_mag : exp_mag;
  }
  if (pos != trimmed.size()) {
    return invalid();
  }

  if (!seen_nonzero) {
    return 0;
  }

  const int64_t pow10 = exp + trailing_zeros - frac_digits;
  if (pow10 < 0 || sig_digits.size() > kMaxUint64Digits) {
    return invalid();
  }

  constexpr uint64_t kMaxUint64 = std::numeric_limits<uint64_t>::max();
  uint64_t value = 0;
  for (char c : sig_digits) {
    const uint64_t digit = static_cast<uint64_t>(c - '0');
    if (value > (kMaxUint64 - digit) / 10) {
      return invalid();
    }
    value = value * 10 + digit;
  }
  for (int64_t i = 0; i < pow10; ++i) {
    if (value > kMaxUint64 / 10) {
      return invalid();
    }
    value *= 10;
  }
  return value;
}

// Parses `option_name` from `options` as an unsigned 64-bit integer. When
// `fraction_policy == FractionPolicy::kRequireIntegral`, requires an exact
// non-negative integer value within `uint64_t`. When
// `fraction_policy == FractionPolicy::kAllowTruncate`, also accepts finite
// non-negative floating-point strings representable in `uint64_t` and truncates
// toward zero. Returns `default_value` when `option_name` is absent.
absl::StatusOr<uint64_t> ParseUint64Option(const ToolOptions& options,
                                           absl::string_view option_name,
                                           uint64_t default_value,
                                           FractionPolicy fraction_policy) {
  std::optional<std::string> raw_value =
      GetParam<std::string>(options, option_name);
  if (!raw_value.has_value()) {
    return default_value;
  }
  if (fraction_policy == FractionPolicy::kRequireIntegral) {
    return ParseExactIntegralUint64(*raw_value, option_name);
  }
  uint64_t parsed_uint = 0;
  if (absl::SimpleAtoi(*raw_value, &parsed_uint)) {
    return parsed_uint;
  }
  double parsed_double = 0.0;
  if (!absl::SimpleAtod(*raw_value, &parsed_double)) {
    return absl::InvalidArgumentError(
        absl::StrCat(option_name, " must be a number"));
  }
  if (!IsValidUint64Double(parsed_double)) {
    return absl::InvalidArgumentError(absl::StrCat(
        option_name, " must be a finite non-negative number within uint64_t"));
  }
  return static_cast<uint64_t>(parsed_double);
}

}  // namespace

absl::StatusOr<TraceViewOption> GetTraceViewOption(const ToolOptions& options) {
  TraceViewOption trace_options;

  absl::StatusOr<std::optional<double>> start_time_ms =
      ParseOptionalTimeMsOption(options, "start_time_ms");
  if (!start_time_ms.ok()) {
    return start_time_ms.status();
  }
  trace_options.start_time_ms =
      start_time_ms->value_or(trace_options.start_time_ms);

  absl::StatusOr<std::optional<double>> end_time_ms =
      ParseOptionalTimeMsOption(options, "end_time_ms");
  if (!end_time_ms.ok()) {
    return end_time_ms.status();
  }
  trace_options.end_time_ms = *end_time_ms;

  absl::StatusOr<std::optional<double>> duration_ms =
      ParseOptionalTimeMsOption(options, "duration_ms");
  if (!duration_ms.ok()) {
    return duration_ms.status();
  }
  trace_options.duration_ms = duration_ms->value_or(trace_options.duration_ms);

  absl::StatusOr<uint64_t> resolution =
      ParseUint64Option(options, "resolution", trace_options.resolution,
                        FractionPolicy::kAllowTruncate);
  if (!resolution.ok()) {
    return resolution.status();
  }
  trace_options.resolution = *resolution;

  absl::StatusOr<uint64_t> unique_id =
      ParseUint64Option(options, "unique_id", trace_options.unique_id,
                        FractionPolicy::kRequireIntegral);
  if (!unique_id.ok()) {
    return unique_id.status();
  }
  trace_options.unique_id = *unique_id;

  trace_options.event_name =
      GetParamWithDefault<std::string>(options, "event_name", "");
  trace_options.search_prefix =
      GetParamWithDefault<std::string>(options, "search_prefix", "");
  trace_options.format =
      GetParamWithDefault<std::string>(options, "format", "json");
  trace_options.search_metadata =
      GetParamWithDefault<bool>(options, "search_metadata", false);

  if (!trace_options.event_name.empty()) {
    TF_RETURN_IF_ERROR(BuildEventLookupKey(trace_options).status());
  }
  return trace_options;
}

absl::Status LoadTraceEventsContainer(
    const TraceEventsLevelDbFilePaths& file_paths,
    const TraceViewOption& trace_option,
    const TraceOptions& profiler_trace_options,
    TraceEventsContainer* trace_container) {
  if (!trace_option.event_name.empty()) {
    absl::StatusOr<EventLookupKey> lookup_key =
        BuildEventLookupKey(trace_option);
    if (!lookup_key.ok()) {
      return lookup_key.status();
    }
    TF_RETURN_IF_ERROR(trace_container->ReadFullEventFromLevelDbTable(
        file_paths.trace_events_metadata_file_path,
        file_paths.trace_events_file_path, trace_option.event_name,
        lookup_key->start_ps, lookup_key->duration_ps, lookup_key->unique_id));
  } else if (!trace_option.search_prefix.empty()) {  // Search Events Request
    if (tsl::Env::Default()
            ->FileExists(file_paths.trace_events_prefix_trie_file_path)
            .ok()) {
      std::unique_ptr<TraceEventsFilterInterface> trace_events_filter =
          CreateTraceEventsFilterFromTraceOptions(profiler_trace_options);
      TF_RETURN_IF_ERROR(trace_container->SearchInLevelDbTable(
          file_paths, trace_option.search_prefix,
          std::move(trace_events_filter),
          {.search_metadata = trace_option.search_metadata}));
    }
  } else {
    absl::StatusOr<TraceViewerVisibility::Window> window =
        BuildVisibilityWindow(trace_option);
    if (!window.ok()) {
      return window.status();
    }
    std::unique_ptr<TraceVisibilityFilter> visibility_filter =
        std::make_unique<TraceVisibilityFilter>(
            *window, trace_option.resolution, profiler_trace_options);
    constexpr int64_t kDisableStreamingThreshold = 500000;
    std::unique_ptr<TraceEventsFilterInterface> trace_events_filter =
        CreateTraceEventsFilterFromTraceOptions(profiler_trace_options);
    TF_RETURN_IF_ERROR(trace_container->LoadFromLevelDbTable(
        file_paths, std::move(trace_events_filter),
        std::move(visibility_filter), kDisableStreamingThreshold));
  }
  return absl::OkStatus();
}

}  // namespace profiler
}  // namespace tensorflow
