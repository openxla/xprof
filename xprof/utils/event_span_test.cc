#include "xprof/utils/event_span.h"

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::EqualsProto;

TEST(StepDetailsTest, CombineFlatOpMetricsDb) {
  StepDetails step1;
  FlatOpMetricsDb db1;
  db1.set_total_op_time_ps(100);
  step1.SetPerCoreFlatOpMetricsDb(db1, 1);

  StepDetails step2;
  FlatOpMetricsDb db2;
  db2.set_total_op_time_ps(200);
  step2.SetPerCoreFlatOpMetricsDb(db2, 2);

  step1.Combine(step2);

  auto& combined_dbs = step1.PerCoreFlatOpMetricsDb();
  ASSERT_EQ(combined_dbs.size(), 2);
  EXPECT_THAT(combined_dbs.at(1), EqualsProto(db1));
  EXPECT_THAT(combined_dbs.at(2), EqualsProto(db2));
}

TEST(StepDetailsTest, ToNonOverlappedStepDetailsCopiesFlatOpMetricsDb) {
  StepDetails step;
  FlatOpMetricsDb db;
  db.set_total_op_time_ps(100);
  step.SetPerCoreFlatOpMetricsDb(db, 1);

  StepDetails non_overlapped = step.ToNonOverlapped();

  auto& copied_dbs = non_overlapped.PerCoreFlatOpMetricsDb();
  ASSERT_EQ(copied_dbs.size(), 1);
  EXPECT_THAT(copied_dbs.at(1), EqualsProto(db));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
