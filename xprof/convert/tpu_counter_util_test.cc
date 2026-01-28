#include "xprof/convert/tpu_counter_util.h"

#include <cstdint>
#include <sstream>
#include <string>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/container/flat_hash_map.h"

namespace xprof {
namespace {

TEST(TpuCounterUtilTest, BasicFunctionality) {
  absl::flat_hash_map<uint64_t, uint64_t> counters;
  counters[1] = 100;
  counters[2] = 200;

  TpuCounterUtil util(/*host_id=*/10, /*device_id=*/5, /*correlation_id=*/123,
                      std::move(counters));

  EXPECT_EQ(util.host_id(), 10);
  EXPECT_EQ(util.device_id(), 5);
  EXPECT_EQ(util.correlation_id(), 123);

  EXPECT_EQ(util.GetValue(1), 100);
  EXPECT_EQ(util.GetValue(2), 200);
  EXPECT_EQ(util.GetValue(3), 0);  // Default value
}

TEST(TpuCounterUtilTest, DebugStringAndStream) {
  absl::flat_hash_map<uint64_t, uint64_t> counters;
  counters[1] = 100;

  TpuCounterUtil util(/*host_id=*/10, /*device_id=*/5, /*correlation_id=*/123,
                      std::move(counters));

  std::string debug_string = util.DebugString();
  EXPECT_THAT(debug_string, testing::HasSubstr("host_id_: 10"));
  EXPECT_THAT(debug_string, testing::HasSubstr("device_id_: 5"));
  EXPECT_THAT(debug_string, testing::HasSubstr("correlation_id_: 123"));
  EXPECT_THAT(debug_string, testing::HasSubstr("1: 100"));

  std::stringstream ss;
  ss << util;
  EXPECT_EQ(ss.str(), debug_string);
}

TEST(UtilizationCountersTest, DebugStringAndStream) {
  UtilizationCounters counters;
  counters.cs_cycles = 1000;
  counters.host_id = 99;

  UtilizationMetrics metric;
  metric.metric = "TestMetric";
  metric.achieved = 50.0;
  metric.peak = 100.0;
  metric.unit = "GB/s";
  counters.metrics.push_back(metric);

  std::string debug_string = counters.DebugString();
  EXPECT_THAT(debug_string, testing::HasSubstr("cs_cycles: 1000"));
  EXPECT_THAT(debug_string, testing::HasSubstr("host_id: 99"));
  EXPECT_THAT(debug_string, testing::HasSubstr("metric: TestMetric"));
  EXPECT_THAT(debug_string, testing::HasSubstr("achieved: 50"));

  std::stringstream ss;
  ss << counters;
  EXPECT_EQ(ss.str(), debug_string);
}

}  // namespace
}  // namespace xprof
