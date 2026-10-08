#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"

#include <cmath>
#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

namespace traceviewer {
namespace {

constexpr double kPicosPerMicro = 1000'000.0;
constexpr double kNanosPerMicro = 1000.0;
constexpr double kMicrosPerMilli = 1000.0;
constexpr double kMicrosPerSecond = 1000'000.0;

// The format specifiers use a non-breaking space (\xc2\xa0) to prevent the
// time unit from being separated from the number by a line break.
constexpr absl::string_view kNonBreakingSpace = "\xc2\xa0";
constexpr absl::string_view kPicosecondsFormat = "%.4g\xc2\xa0ps";
constexpr absl::string_view kNanosecondsFormat = "%.4g\xc2\xa0ns";
constexpr absl::string_view kMicrosecondsFormat = "%.4g\xc2\xa0us";
constexpr absl::string_view kMillisecondsFormat = "%.4g\xc2\xa0ms";
constexpr absl::string_view kSecondsFormat = "%.4g\xc2\xa0s";

std::string FormatDoubleWithTrimmedZeros(double value,
                                         int max_fraction_digits = 3) {
  if (std::abs(value) < 1e-12) {
    return "0";
  }
  std::string s = absl::StrFormat("%.*f", max_fraction_digits, value);
  if (absl::StrContains(s, '.')) {
    while (!s.empty() && s.back() == '0') {
      s.pop_back();
    }
    if (!s.empty() && s.back() == '.') {
      s.pop_back();
    }
  }
  if (s == "-0") {
    return "0";
  }
  return s;
}

}  // namespace

std::string FormatTime(Microseconds time_us) {
  if (!std::isfinite(time_us) || time_us < 0) {
    return "-";
  }

  if (time_us == 0.0) {
    return "0";
  }

  if (time_us < 1 / kNanosPerMicro) {
    return absl::StrFormat(kPicosecondsFormat, time_us * kPicosPerMicro);
  } else if (time_us < 1.0) {
    return absl::StrFormat(kNanosecondsFormat, time_us * kNanosPerMicro);
  } else if (time_us < kMicrosPerMilli) {
    return absl::StrFormat(kMicrosecondsFormat, time_us);
  } else if (time_us < kMicrosPerSecond) {
    return absl::StrFormat(kMillisecondsFormat, time_us / kMicrosPerMilli);
  } else {  // >= 1 second
    return absl::StrFormat(kSecondsFormat, time_us / kMicrosPerSecond);
  }
}

std::string FormatTime(Microseconds time_us, Microseconds tick_interval) {
  if (!std::isfinite(time_us)) {
    return "-";
  }
  if (time_us < 0.0) {
    if (time_us > -1e-9) {
      time_us = 0.0;
    } else {
      return "-";
    }
  }
  if (!std::isfinite(tick_interval) || tick_interval <= 0) {
    return FormatTime(time_us);
  }

  double scaled_value = 0.0;
  absl::string_view unit;

  if (tick_interval >= kMicrosPerSecond) {
    scaled_value = time_us / kMicrosPerSecond;
    unit = "s";
  } else if (tick_interval >= kMicrosPerMilli) {
    scaled_value = time_us / kMicrosPerMilli;
    unit = "ms";
  } else if (tick_interval >= 1.0) {
    scaled_value = time_us;
    unit = "us";
  } else if (tick_interval >= 1.0 / kNanosPerMicro) {
    scaled_value = time_us * kNanosPerMicro;
    unit = "ns";
  } else {
    scaled_value = time_us * kPicosPerMicro;
    unit = "ps";
  }

  const std::string formatted_number =
      FormatDoubleWithTrimmedZeros(scaled_value, 3);
  return absl::StrCat(formatted_number, kNonBreakingSpace, unit);
}

Microseconds MillisToMicros(double time_ms) {
  return time_ms * kMicrosPerMilli;
}

Milliseconds MicrosToMillis(Microseconds time_us) {
  return time_us / kMicrosPerMilli;
}

Microseconds CalculateNiceInterval(Microseconds min_interval) {
  if (min_interval <= 0) return 1.0;

  const double power_of_10 =
      std::pow(10.0, std::floor(std::log10(min_interval)));

  constexpr double kNiceIntervals[] = {1.0, 2.0, 5.0, 10.0};

  for (double nice_interval : kNiceIntervals) {
    Microseconds interval = nice_interval * power_of_10;
    if (interval >= min_interval) {
      return interval;
    }
  }
  // This part is unreachable because the loop is guaranteed to return.
  // The last value (10.0 * power_of_10) will always be greater than or equal to
  // min_interval. However, a return statement is required by the compiler.
  return 10.0 * power_of_10;
}

}  // namespace traceviewer
