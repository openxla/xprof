#include "xprof/utils/event_span.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "plugin/xprof/protobuf/flat_op_metrics.pb.h"
#include "plugin/xprof/protobuf/op_metrics.pb.h"

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

TEST(StepDetailsTest, CombineMovePerCoreOpMetricsDb) {
  StepDetails step1;
  OpMetricsDb db1;
  db1.set_total_op_time_ps(100);
  step1.SetPerCoreOpMetricsDb(db1, 1);

  StepDetails step2;
  OpMetricsDb db2;
  db2.set_total_op_time_ps(200);
  step2.SetPerCoreOpMetricsDb(db2, 2);

  step1.Combine(std::move(step2));

  const auto& combined_dbs = step1.PerCoreOpMetricsDb();
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

TEST(EventSpanTest, IntersectCombineStepEventsMove) {
  StepEvents src;
  StepMarker marker1(StepMarkerType::kExplicitHostStepMarker, "step1",
                     tsl::profiler::Timespan(0, 100));
  src[1].AddMarker(marker1);
  StepMarker marker2(StepMarkerType::kExplicitHostStepMarker, "step2",
                     tsl::profiler::Timespan(100, 100));
  src[2].AddMarker(marker2);

  StepEvents dst;
  StepMarker marker3(StepMarkerType::kImplicitHostStepMarker, "step1_implicit",
                     tsl::profiler::Timespan(10, 80));
  dst[1].AddMarker(marker3);
  StepMarker marker4(StepMarkerType::kExplicitHostStepMarker, "step3",
                     tsl::profiler::Timespan(200, 100));
  dst[3].AddMarker(marker4);

  IntersectMoveCombineStepEvents(std::move(src), &dst);

  // Intersection of {1, 2} and {1, 3} is {1}.
  EXPECT_EQ(dst.size(), 1);
  ASSERT_TRUE(dst.contains(1));
  EXPECT_EQ(dst[1].Markers().size(), 2);
}

TEST(EventSpanTest, UnionCombineStepEventsMove) {
  StepEvents src;
  StepMarker marker1(StepMarkerType::kExplicitHostStepMarker, "step1",
                     tsl::profiler::Timespan(0, 100));
  src[1].AddMarker(marker1);
  StepMarker marker2(StepMarkerType::kExplicitHostStepMarker, "step2",
                     tsl::profiler::Timespan(100, 100));
  src[2].AddMarker(marker2);

  StepEvents dst;
  StepMarker marker3(StepMarkerType::kExplicitHostStepMarker, "step3",
                     tsl::profiler::Timespan(200, 100));
  dst[3].AddMarker(marker3);

  UnionMoveCombineStepEvents(std::move(src), &dst);

  // Union of {1, 2} and {3} is {1, 2, 3}.
  EXPECT_EQ(dst.size(), 3);
  EXPECT_TRUE(dst.contains(1));
  EXPECT_TRUE(dst.contains(2));
  EXPECT_TRUE(dst.contains(3));
}

TEST(EventSpanTest, IntersectCombineStepEventsConstRef) {
  StepEvents src;
  StepMarker marker1(StepMarkerType::kExplicitHostStepMarker, "step1",
                     tsl::profiler::Timespan(0, 100));
  src[1].AddMarker(marker1);
  StepMarker marker2(StepMarkerType::kExplicitHostStepMarker, "step2",
                     tsl::profiler::Timespan(100, 100));
  src[2].AddMarker(marker2);

  StepEvents dst;
  StepMarker marker3(StepMarkerType::kImplicitHostStepMarker, "step1_implicit",
                     tsl::profiler::Timespan(10, 80));
  dst[1].AddMarker(marker3);
  StepMarker marker4(StepMarkerType::kExplicitHostStepMarker, "step3",
                     tsl::profiler::Timespan(200, 100));
  dst[3].AddMarker(marker4);

  IntersectCombineStepEvents(src, &dst);

  // Intersection of {1, 2} and {1, 3} is {1}.
  EXPECT_EQ(dst.size(), 1);
  ASSERT_TRUE(dst.contains(1));
  EXPECT_EQ(dst[1].Markers().size(), 2);
}

TEST(EventSpanTest, UnionCombineStepEventsConstRef) {
  StepEvents src;
  StepMarker marker1(StepMarkerType::kExplicitHostStepMarker, "step1",
                     tsl::profiler::Timespan(0, 100));
  src[1].AddMarker(marker1);
  StepMarker marker2(StepMarkerType::kExplicitHostStepMarker, "step2",
                     tsl::profiler::Timespan(100, 100));
  src[2].AddMarker(marker2);

  StepEvents dst;
  StepMarker marker3(StepMarkerType::kExplicitHostStepMarker, "step3",
                     tsl::profiler::Timespan(200, 100));
  dst[3].AddMarker(marker3);

  UnionCombineStepEvents(src, &dst);

  // Union of {1, 2} and {3} is {1, 2, 3}.
  EXPECT_EQ(dst.size(), 3);
  EXPECT_TRUE(dst.contains(1));
  EXPECT_TRUE(dst.contains(2));
  EXPECT_TRUE(dst.contains(3));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
