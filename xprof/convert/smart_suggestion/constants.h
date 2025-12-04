/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_CONSTANTS_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_CONSTANTS_H_

namespace tensorflow {
namespace profiler {

// Thresholds for HBM utilization percentage for bottleneck classification.
// If HBM utilization is higher than kHbmUtilizationHighThreshold, it is
// considered high. If it is lower than kHbmUtilizationLowThreshold, it is
// considered low.
inline constexpr double kHbmUtilizationLowThreshold = 50.0;
inline constexpr double kHbmUtilizationHighThreshold = 70.0;

// Thresholds for MXU utilization percentage for bottleneck classification.
// If MXU utilization is higher than kMxuUtilizationHighThreshold, it is
// considered high. If it is lower than kMxuUtilizationLowThreshold, it is
// considered low.
inline constexpr double kMxuUtilizationLowThreshold = 50.0;
inline constexpr double kMxuUtilizationHighThreshold = 70.0;

// Thresholds for input percentage of step time for bottleneck classification.
// If input percentage is higher than kInfeedPercentageThreshold, it is
// considered input bound.
inline constexpr double kInfeedPercentageThreshold = 10.0;

// Threshold for collective op percentage of step time for bottleneck
// classification.
inline constexpr double kCollectiveBoundThresholdInPercent = 30.0;

// Threshold for data shuffle op percentage of step time for bottleneck
// classification.
inline constexpr double kDataShuffleBoundThresholdInPercent = 30.0;

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_CONSTANTS_H_
