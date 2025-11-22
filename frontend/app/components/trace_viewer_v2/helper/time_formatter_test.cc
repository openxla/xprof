#include "xprof/frontend/app/components/trace_viewer_v2/helper/time_formatter.h"

#include <limits>
#include <string>

#include "<gtest/gtest.h>"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace traceviewer {
namespace {

constexpr absl::string_view kNonBreakingSpace = "\xc2\xa0";

// Returns a time string with value and unit separated by a non-breaking space.
std::string TimeWithUnit(absl::string_view time, absl::string_view unit) {
  return absl::StrCat(time, kNonBreakingSpace, unit);
}

TEST(TimeFormatterTest, FormatTime) {
  EXPECT_EQ(FormatTime(0.0), "0");
  EXPECT_EQ(FormatTime(0.000000456), TimeWithUnit("0.456", "ps"));
  EXPECT_EQ(FormatTime(0.00000123), TimeWithUnit("1.23", "ps"));
  EXPECT_EQ(FormatTime(0.0005), TimeWithUnit("500", "ps"));
  EXPECT_EQ(FormatTime(0.001), TimeWithUnit("1", "ns"));
  EXPECT_EQ(FormatTime(0.5), TimeWithUnit("500", "ns"));
  EXPECT_EQ(FormatTime(1.2345), TimeWithUnit("1.234", "us"));
  EXPECT_EQ(FormatTime(999.9995), TimeWithUnit("1000", "us"));
  EXPECT_EQ(FormatTime(1000.0), TimeWithUnit("1", "ms"));
  EXPECT_EQ(FormatTime(1234.567), TimeWithUnit("1.235", "ms"));
  EXPECT_EQ(FormatTime(999999.9), TimeWithUnit("1000", "ms"));
  EXPECT_EQ(FormatTime(1000000.0), TimeWithUnit("1", "s"));
  EXPECT_EQ(FormatTime(1234567.0), TimeWithUnit("1.235", "s"));
  EXPECT_EQ(FormatTime(60000000.0), TimeWithUnit("60", "s"));
}

TEST(TimeFormatterTest, FormatTimeNotFiniteOrNegative) {
  EXPECT_EQ(FormatTime(std::numeric_limits<double>::infinity()), "-");
  EXPECT_EQ(FormatTime(-std::numeric_limits<double>::infinity()), "-");
  EXPECT_EQ(FormatTime(std::numeric_limits<double>::quiet_NaN()), "-");
  EXPECT_EQ(FormatTime(-0.5), "-");
  EXPECT_EQ(FormatTime(-100.0), "-");
}

TEST(TimeFormatterTest, CalculateNiceInterval) {
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(0.0), 1.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(-1.0), 1.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(0.8), 1.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(1.0), 1.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(1.1), 2.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(2.0), 2.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(3.5), 5.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(5.0), 5.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(7.0), 10.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(10.0), 10.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(12.0), 20.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(35.0), 50.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(80.0), 100.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(100.0), 100.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(120.0), 200.0);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(0.0012), 0.002);
  EXPECT_DOUBLE_EQ(CalculateNiceInterval(0.0035), 0.005);
}
}  // namespace
}  // namespace traceviewer
