/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xprof/convert/file_utils.h"
#include "xprof/convert/trace_viewer/lite_trace_events.h"
#include "xprof/convert/xplane_to_trace_container.h"
#include "xprof/utils/tensorflow_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

TEST(XPlaneToTraceContainerTest, CounterLine) {
  XSpace xspace;
  CHECK_OK(ParseTextFormatFromString(
      absl::Substitute(
          "planes {"
          "  name: \"/device:GPU:0\""
          "  lines {"
          "    name: \"_counters_\""
          "    events {"
          "      metadata_id: 100"
          "      offset_ps: $0"
          "      stats { metadata_id: 200 uint64_value: 100 }"
          "    }"
          "    events {"
          "      metadata_id: 100"
          "      offset_ps: $1"
          "      stats { metadata_id: 200 uint64_value: 200 }"
          "    }"
          "    events {"
          "      metadata_id: 101"
          "      offset_ps: $0"
          "      stats { metadata_id: 201 uint64_value: 300 }"
          "    }"
          "    events {"
          "      metadata_id: 101"
          "      offset_ps: $1"
          "      stats { metadata_id: 201 uint64_value: 400 }"
          "    }"
          "  }"
          "  lines {"
          "    id: 14"
          "    name: \"Stream #14(MemcpyH2D)\""
          "    timestamp_ns: $3"
          "    events {"
          "      metadata_id: 10"
          "      offset_ps: 0"
          "      duration_ps: $1"
          "      stats { metadata_id: 8 uint64_value: 100 }"
          "      stats { metadata_id: 9 str_value: \"$$1\" }"
          "    }"
          "    events {"
          "      metadata_id: 10"
          "      offset_ps: $0"
          "      duration_ps: $3"
          "      stats { metadata_id: 8 uint64_value: 200 }"
          "      stats { metadata_id: 9 str_value: \"abcd\" }"
          "    }"
          "  }"
          "  event_metadata {key: 10 value: { id: 10 name: \"MemcpyD2D\" }}"
          "  event_metadata {key: 100 value: { id: 100 name: \"Counter 1\" }}"
          "  event_metadata {key: 101 value: { id: 101 name: \"Counter 2\" }}"
          "  stat_metadata {key: 8 value: { id: 8 name: \"RemoteCall\"}}"
          "  stat_metadata {key: 9 value: { id: 8 name: \"context_id\"}}"
          "  stat_metadata {key: 200 value: { id: 200 name: \"counter_1\"}}"
          "  stat_metadata {key: 201 value: { id: 201 name: \"counter_2\"}}"
          "}",
          tsl::profiler::UniToPico(1), tsl::profiler::UniToPico(2),
          tsl::profiler::UniToNano(1), tsl::profiler::UniToNano(500)),
      &xspace));
  TraceEventsContainer container;
  ConvertXSpaceToTraceEventsContainer("localhost", xspace, &container);
  absl::flat_hash_map<std::string, absl::flat_hash_map<uint64_t, uint64_t>>
      counter_offset_to_values;
  container.ForAllEvents([&counter_offset_to_values](const TraceEvent& event) {
    if (absl::StrContains(event.name(), "Counter")) {
      uint64_t offset = event.timestamp_ps();
      RawData raw_data;
      raw_data.ParseFromString(event.raw_data());
      counter_offset_to_values[event.name()][offset] =
          raw_data.args().arg(0).uint_value();
    }
  });
  EXPECT_THAT(
      counter_offset_to_values,
      UnorderedElementsAre(
          Pair("Counter 1",
               UnorderedElementsAre(Pair(tsl::profiler::UniToPico(1), 100),
                                    Pair(tsl::profiler::UniToPico(2), 200))),
          Pair("Counter 2",
               UnorderedElementsAre(Pair(tsl::profiler::UniToPico(1), 300),
                                    Pair(tsl::profiler::UniToPico(2), 400)))));
}

TEST(XPlaneToTraceContainerTest, AsyncLine) {
  XSpace xspace;
  constexpr absl::string_view kAsyncOpEventName = "Async Op";
  CHECK_OK(ParseTextFormatFromString(
      absl::Substitute(
          "planes {"
          "  name: \"/device:GPU:0\""
          "  lines {"
          "    id: 15"
          "    name: \"$2\""
          "    timestamp_ns: $0"
          "    events {"
          "      metadata_id: 11"
          "      offset_ps: 0"
          "      duration_ps: $1"
          "    }"
          "  }"
          "  event_metadata {key: 11 value: { id: 11 name: \"$3\" }}"
          "}",
          tsl::profiler::UniToNano(2), tsl::profiler::UniToPico(1),
          tsl::profiler::kXlaAsyncOpLineName, kAsyncOpEventName),
      &xspace));
  TraceEventsContainer container;
  ConvertXSpaceToTraceEventsContainer("localhost", xspace, &container);
  bool async_event_found = false;
  container.ForAllEvents(
      [&async_event_found, &kAsyncOpEventName](const TraceEvent& event) {
        if (event.name() == kAsyncOpEventName) {
          async_event_found = true;
          EXPECT_EQ(event.timestamp_ps(), tsl::profiler::UniToPico(2));
          EXPECT_EQ(event.duration_ps(), tsl::profiler::UniToPico(1));
        }
      });
  EXPECT_TRUE(async_event_found);
}

TEST(XPlaneToTraceContainerTest, SubprocessHostTrace_ConvertsToTraceEvents) {
  XSpace xspace;
  CHECK_OK(ParseTextFormatFromString(
      R"pb(
        planes {
          name: "/host:CPU [100]"
          lines {
            id: 2
            name: "Thread 2"
            timestamp_ns: 2000
            events { metadata_id: 2 offset_ps: 0 duration_ps: 200 }
          }
          event_metadata {
            key: 2
            value: { id: 2 name: "Event 2" }
          }
        }
        planes {
          name: "/host:CPU"
          lines {
            id: 1
            name: "Thread 1"
            timestamp_ns: 1000
            events { metadata_id: 1 offset_ps: 0 duration_ps: 100 }
          }
          event_metadata {
            key: 1
            value: { id: 1 name: "Event 1" }
          }
        }
      )pb",
      &xspace));
  TraceEventsContainer container;
  ConvertXSpaceToTraceEventsContainer("localhost", xspace, &container);

  bool event1_found = false;
  bool event2_found = false;
  container.ForAllEvents([&](const TraceEvent& event) {
    if (event.name() == "Event 1") {
      event1_found = true;
      EXPECT_EQ(event.device_id(), tsl::profiler::kHostThreadsDeviceId);
    } else if (event.name() == "Event 2") {
      event2_found = true;
      EXPECT_EQ(event.device_id(), tsl::profiler::kHostThreadsDeviceId + 1);
    }
  });
  EXPECT_TRUE(event1_found);
  EXPECT_TRUE(event2_found);
}

TEST(XPlaneToTraceContainerTest,
     LiteTraceEventsMultiLineAsyncAndSimultaneousEventsAreMonotonic) {
  XSpace xspace;
  CHECK_OK(ParseTextFormatFromString(
      R"pb(
        planes {
          name: "/host:CPU"
          lines {
            id: 1
            name: "MegaScale Thread 1"
            events {
              metadata_id: 1
              offset_ps: 5000000
              duration_ps: 100000
              stats { metadata_id: 10 uint64_value: 1 }
              stats { metadata_id: 11 int64_value: 1 }
            }
            events {
              metadata_id: 1
              offset_ps: 5000000
              duration_ps: 300000
              stats { metadata_id: 10 uint64_value: 2 }
              stats { metadata_id: 11 int64_value: 1 }
            }
          }
          lines {
            id: 2
            name: "MegaScale Thread 2"
            events {
              metadata_id: 1
              offset_ps: 1000000
              duration_ps: 200000
              stats { metadata_id: 10 uint64_value: 3 }
              stats { metadata_id: 11 int64_value: 1 }
            }
            events {
              metadata_id: 1
              offset_ps: 5000000
              duration_ps: 200000
              stats { metadata_id: 10 uint64_value: 4 }
              stats { metadata_id: 11 int64_value: 1 }
            }
            events {
              metadata_id: 3
              offset_ps: 5000000
              duration_ps: 200000
              stats { metadata_id: 10 uint64_value: 5 }
              stats { metadata_id: 11 int64_value: 1 }
            }
          }
          event_metadata {
            key: 1
            value: {
              id: 1
              name: "MegaScale:NetworkSendDone_CollectiveChannel_Beta"
            }
          }
          event_metadata {
            key: 3
            value: {
              id: 3
              name: "MegaScale:NetworkSendDone_CollectiveChannel_Alpha"
            }
          }
          stat_metadata {
            key: 10
            value: { id: 10 name: "flow" }
          }
          stat_metadata {
            key: 11
            value: { id: 11 name: "is_async" }
          }
        }
      )pb",
      &xspace));

  TraceEventLiteContainer lite_container;
  ConvertXSpaceToLiteTraceEventsContainer("localhost", xspace, &lite_container);
  EXPECT_EQ(NumEvents(lite_container), 5);

  auto events_by_level = LiteTraceEventsByLevel(&lite_container);
  EXPECT_FALSE(events_by_level.empty());
  size_t total_events = 0;
  for (const auto& level_events : events_by_level) {
    total_events += level_events.size();
    for (size_t i = 1; i < level_events.size(); ++i) {
      const TraceEventLite* prev = level_events[i - 1];
      const TraceEventLite* curr = level_events[i];
      EXPECT_TRUE(prev->timestamp_ps < curr->timestamp_ps ||
                  (prev->timestamp_ps == curr->timestamp_ps &&
                   prev->serial < curr->serial));
    }
  }
  EXPECT_EQ(total_events, NumEvents(lite_container));
}

TEST(XPlaneToTraceContainerTest,
     LiteTraceEventsComparatorTieBreakersStayMonotonicAcrossLevels) {
  XSpace xspace;
  CHECK_OK(ParseTextFormatFromString(
      R"pb(
        planes {
          name: "/host:CPU"
          lines {
            id: 0
            name: "_counters_"
            events { metadata_id: 1 offset_ps: 4999999 }
            events { metadata_id: 1 offset_ps: 5000000 }
          }
          lines {
            id: 1
            name: "Sync 1"
            events {
              metadata_id: 1
              offset_ps: 1000000
              duration_ps: 2000000000000
            }
          }
          lines {
            id: 2
            name: "Sync 2"
            events {
              metadata_id: 1
              offset_ps: 2000000
              duration_ps: 2000000000000
            }
            events { metadata_id: 1 offset_ps: 2000001 duration_ps: 1 }
            events { metadata_id: 1 offset_ps: 2000001 duration_ps: 1 }
          }
          lines {
            id: 3
            name: "Async 1"
            events {
              metadata_id: 1
              offset_ps: 4999999
              stats { metadata_id: 10 uint64_value: 33 }
              stats { metadata_id: 11 int64_value: 1 }
            }
            events {
              metadata_id: 1
              offset_ps: 5000000
              stats { metadata_id: 10 uint64_value: 37 }
              stats { metadata_id: 11 int64_value: 1 }
            }
          }
          lines {
            id: 4
            name: "Async 2"
            events {
              metadata_id: 1
              offset_ps: 5000000
              stats { metadata_id: 10 uint64_value: 41 }
              stats { metadata_id: 11 int64_value: 1 }
            }
          }
          event_metadata {
            key: 1
            value: { id: 1 name: "SameName" }
          }
          stat_metadata {
            key: 10
            value: { id: 10 name: "flow" }
          }
          stat_metadata {
            key: 11
            value: { id: 11 name: "is_async" }
          }
        }
      )pb",
      &xspace));

  TraceEventLiteContainer lite_container;
  ConvertXSpaceToLiteTraceEventsContainer("localhost", xspace, &lite_container);
  ASSERT_EQ(NumEvents(lite_container), 9);

  auto events_by_level = LiteTraceEventsByLevel(&lite_container);
  size_t total_events = 0;
  std::vector<std::pair<size_t, uint32_t>> forward_level_serials;
  for (size_t level = 0; level < events_by_level.size(); ++level) {
    const auto& level_events = events_by_level[level];
    total_events += level_events.size();
    for (size_t i = 0; i < level_events.size(); ++i) {
      forward_level_serials.emplace_back(level, level_events[i]->serial);
      if (i > 0) {
        const TraceEventLite* prev = level_events[i - 1];
        const TraceEventLite* curr = level_events[i];
        EXPECT_TRUE(prev->timestamp_ps < curr->timestamp_ps ||
                    (prev->timestamp_ps == curr->timestamp_ps &&
                     prev->serial < curr->serial));
      }
    }
  }
  EXPECT_EQ(total_events, NumEvents(lite_container));
  EXPECT_FALSE(events_by_level[0].empty());
  EXPECT_FALSE(events_by_level[12].empty());

  XSpace reversed_xspace = xspace;
  auto* mutable_lines = reversed_xspace.mutable_planes(0)->mutable_lines();
  for (int i = 0, j = mutable_lines->size() - 1; i < j; ++i, --j) {
    mutable_lines->SwapElements(i, j);
  }
  TraceEventLiteContainer reversed_container;
  ConvertXSpaceToLiteTraceEventsContainer("localhost", reversed_xspace,
                                          &reversed_container);
  auto reversed_by_level = LiteTraceEventsByLevel(&reversed_container);
  std::vector<std::pair<size_t, uint32_t>> reversed_level_serials;
  for (size_t level = 0; level < reversed_by_level.size(); ++level) {
    for (const TraceEventLite* ev : reversed_by_level[level]) {
      reversed_level_serials.emplace_back(level, ev->serial);
    }
  }
  EXPECT_EQ(forward_level_serials, reversed_level_serials);
}

}  // namespace

}  // namespace profiler
}  // namespace tensorflow
