#include "xprof/utils/roofline_model_utils.h"

#include "gtest/gtest.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(RooflineModelUtilsTest, RidgePointNormalCase) {
  double peak_gigaflops_per_second = 1000.0;
  double peak_gibibytes_per_second = 100.0;
  double ridge_point =
      RidgePoint(peak_gigaflops_per_second, peak_gibibytes_per_second);
  EXPECT_NEAR(ridge_point, 9.31322, 1e-5);
}

TEST(RooflineModelUtilsTest, RidgePointAnotherNormalCase) {
  double peak_gigaflops_per_second = 500.0;
  double peak_gibibytes_per_second = 200.0;
  double ridge_point =
      RidgePoint(peak_gigaflops_per_second, peak_gibibytes_per_second);
  EXPECT_NEAR(ridge_point, 2.32830, 1e-5);
}

TEST(RooflineModelUtilsTest, RidgePointZeroMemoryBandwidth) {
  double peak_gigaflops_per_second = 100.0;
  double peak_gibibytes_per_second = 0.0;
  double ridge_point =
      RidgePoint(peak_gigaflops_per_second, peak_gibibytes_per_second);
  EXPECT_DOUBLE_EQ(ridge_point, 0.0);
}

TEST(RooflineModelUtilsTest, RidgePointZeroFlops) {
  double peak_gigaflops_per_second = 0.0;
  double peak_gibibytes_per_second = 100.0;
  double ridge_point =
      RidgePoint(peak_gigaflops_per_second, peak_gibibytes_per_second);
  EXPECT_DOUBLE_EQ(ridge_point, 0.0);
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
