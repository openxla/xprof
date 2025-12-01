#include "xprof/convert/trace_viewer/trace_events_util.h"
#include <cstdint>
#include <limits>

#include "<gtest/gtest.h>"
#include "xla/tsl/profiler/utils/timespan.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::profiler::Timespan;

TEST(ExpandTraceSpanTest, EmptyTraceSetsBounds) {
  Trace trace;
  ExpandTraceSpan(Timespan::FromEndPoints(10, 20), &trace);
  EXPECT_EQ(trace.min_timestamp_ps(), 10);
  EXPECT_EQ(trace.max_timestamp_ps(), 20);
}

TEST(ExpandTraceSpanTest, LargerSpanExpandsBounds) {
  Trace trace;
  trace.set_min_timestamp_ps(10);
  trace.set_max_timestamp_ps(20);
  ExpandTraceSpan(Timespan::FromEndPoints(5, 25), &trace);
  EXPECT_EQ(trace.min_timestamp_ps(), 5);
  EXPECT_EQ(trace.max_timestamp_ps(), 25);
  ExpandTraceSpan(
      Timespan::FromEndPoints(5, std::numeric_limits<uint64_t>::max() / 2),
      &trace);
  EXPECT_EQ(trace.min_timestamp_ps(), 5);
  EXPECT_EQ(trace.max_timestamp_ps(), std::numeric_limits<uint64_t>::max() / 2);
}

TEST(ExpandTraceSpanTest, SmallerSpanDoesNotChangeBounds) {
  Trace trace;
  trace.set_min_timestamp_ps(10);
  trace.set_max_timestamp_ps(20);
  ExpandTraceSpan(Timespan::FromEndPoints(12, 18), &trace);
  EXPECT_EQ(trace.min_timestamp_ps(), 10);
  EXPECT_EQ(trace.max_timestamp_ps(), 20);
}

TEST(ExpandTraceSpanTest, BadTimestampsAreIgnored) {
  Trace trace;
  trace.set_min_timestamp_ps(10);
  trace.set_max_timestamp_ps(20);
  ExpandTraceSpan(
      Timespan::FromEndPoints(5, 1 + std::numeric_limits<uint64_t>::max() / 2),
      &trace);
  EXPECT_EQ(trace.min_timestamp_ps(), 5);
  EXPECT_EQ(trace.max_timestamp_ps(), 20);
  // This will become
  // [std::numeric_limits<uint64_t>::max(),std::numeric_limits<uint64_t>::max()].
  // (end_time must be >= start_time)
  ExpandTraceSpan(
      Timespan::FromEndPoints(1 + std::numeric_limits<uint64_t>::max() / 2, 25),
      &trace);
  EXPECT_EQ(trace.min_timestamp_ps(), 5);
  EXPECT_EQ(trace.max_timestamp_ps(), 20);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
