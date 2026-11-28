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
#include "google/protobuf/arena.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
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
            events {
              metadata_id: 1
              offset_ps: 4999999
              stats { metadata_id: 12 uint64_value: 1 }
            }
            events {
              metadata_id: 1
              offset_ps: 5000000
              stats { metadata_id: 12 uint64_value: 2 }
            }
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
          stat_metadata {
            key: 12
            value: { id: 12 name: "val" }
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

// TPU non-core planes such as "#Chip0 HBM 0" hold the periodic HBM (CMN)
// counters on TPU7x. "#Chip0 Misc" has no events and must be skipped.
constexpr char kTpuXSpaceWithNonCorePlanes[] = R"pb(
  planes {
    id: 2
    name: "/device:TPU:0"
    lines {
      id: 1
      name: "XLA Ops"
      events { metadata_id: 1 offset_ps: 0 duration_ps: 100 }
    }
    event_metadata {
      key: 1
      value: { id: 1 name: "fusion" }
    }
  }
  planes {
    id: 1
    name: "#Chip0 HBM 0"
    lines {
      name: "_counters_"
      events {
        metadata_id: 1
        offset_ps: 50
        stats { metadata_id: 1 double_value: 42 }
      }
    }
    event_metadata {
      key: 1
      value: { id: 1 name: "RD_RSP_BEAT_0" }
    }
    stat_metadata {
      key: 1
      value: { id: 1 name: "CMN Sample Stats (MB/sec)" }
    }
  }
  planes {
    id: 6
    name: "#Chip0 Misc"
    lines { name: "_counters_" }
  }
)pb";

constexpr uint32_t kHbmDeviceId = tsl::profiler::kFirstDeviceId + 1;
constexpr uint32_t kMiscDeviceId = tsl::profiler::kFirstDeviceId + 6;

TEST(XPlaneToTraceContainerTest, TpuNonCorePlaneWithEventsIsConverted) {
  XSpace xspace;
  CHECK_OK(ParseTextFormatFromString(kTpuXSpaceWithNonCorePlanes, &xspace));
  TraceEventsContainer container;
  ConvertXSpaceToTraceEventsContainer("localhost", xspace, &container);

  ASSERT_EQ(container.trace().devices().count(kHbmDeviceId), 1);
  EXPECT_EQ(container.trace().devices().at(kHbmDeviceId).name(),
            "localhost #Chip0 HBM 0");
  EXPECT_EQ(container.trace().devices().count(kMiscDeviceId), 0);
  bool hbm_counter_found = false;
  container.ForAllEvents([&](const TraceEvent& event) {
    if (event.name() == "RD_RSP_BEAT_0") {
      hbm_counter_found = true;
      EXPECT_EQ(event.device_id(), kHbmDeviceId);
    }
  });
  EXPECT_TRUE(hbm_counter_found);
}

TEST(XPlaneToTraceContainerTest, TpuNonCorePlaneWithEventsIsConvertedToLite) {
  XSpace xspace;
  CHECK_OK(ParseTextFormatFromString(kTpuXSpaceWithNonCorePlanes, &xspace));
  TraceEventLiteContainer container;
  ConvertXSpaceToLiteTraceEventsContainer("localhost", xspace, &container);

  ASSERT_EQ(container.trace.devices().count(kHbmDeviceId), 1);
  EXPECT_EQ(container.trace.devices().at(kHbmDeviceId).name(),
            "localhost #Chip0 HBM 0");
  EXPECT_EQ(container.trace.devices().count(kMiscDeviceId), 0);
  size_t num_hbm_counter_events = 0;
  for (const auto& [metadata, events] : container.counter_events) {
    if (metadata.device_id == kHbmDeviceId) {
      num_hbm_counter_events += events.size();
    }
  }
  EXPECT_EQ(num_hbm_counter_events, 1);
}

TEST(XPlaneToTraceContainerTest,
     InvalidCounterEventsOnTpuNonCorePlaneAreSkipped) {
  XSpace xspace;
  CHECK_OK(ParseTextFormatFromString(
      R"pb(
        planes {
          id: 2
          name: "/device:TPU:0"
          lines {
            id: 1
            name: "XLA Ops"
            events { metadata_id: 1 offset_ps: 0 duration_ps: 100 }
          }
          event_metadata {
            key: 1
            value: { id: 1 name: "fusion" }
          }
        }
        planes {
          id: 1
          name: "#Chip0 HBM 0"
          lines {
            name: "_counters_"
            events { metadata_id: 1 offset_ps: 10 }
            events {
              metadata_id: 2
              offset_ps: 20
              stats { metadata_id: 1 double_value: 10 }
            }
            events {
              metadata_id: 1
              offset_ps: 30
              stats { metadata_id: 2 str_value: "bad" }
            }
            events {
              metadata_id: 1
              offset_ps: 40
              stats { metadata_id: 1 double_value: 20 }
              stats { metadata_id: 3 uint64_value: 30 }
            }
            events {
              metadata_id: 1
              offset_ps: 50
              stats { metadata_id: 1 double_value: 42 }
            }
          }
          event_metadata {
            key: 1
            value: { id: 1 name: "RD_RSP_BEAT_0" }
          }
          event_metadata {
            key: 2
            value: {
              id: 2
              name: "RD_RSP_BEAT_1"
              display_name: "RD_RSP_BEAT_1_DISPLAY"
            }
          }
          stat_metadata {
            key: 1
            value: { id: 1 name: "CMN Sample Stats (MB/sec)" }
          }
          stat_metadata {
            key: 2
            value: { id: 2 name: "label" }
          }
          stat_metadata {
            key: 3
            value: { id: 3 name: "extra" }
          }
        }
      )pb",
      &xspace));

  TraceEventsContainer container;
  ConvertXSpaceToTraceEventsContainer("localhost", xspace, &container);
  size_t num_hbm_events = 0;
  container.ForAllEvents([&](const TraceEvent& event) {
    if (event.device_id() == kHbmDeviceId) {
      ++num_hbm_events;
      EXPECT_EQ(event.name(), "RD_RSP_BEAT_0");
      EXPECT_EQ(event.timestamp_ps(), 50);
    }
  });
  EXPECT_EQ(num_hbm_events, 1);

  TraceEventLiteContainer lite_container;
  ConvertXSpaceToLiteTraceEventsContainer("localhost", xspace, &lite_container);
  size_t num_lite_hbm_events = 0;
  google::protobuf::Arena arena;
  absl::flat_hash_map<uint64_t, std::string> local_name_table;
  for (const auto& [metadata, events] : lite_container.counter_events) {
    if (metadata.device_id != kHbmDeviceId) continue;
    for (const TraceEventLite& lite_event : events) {
      ++num_lite_hbm_events;
      tsl::profiler::XEventVisitor visitor(
          metadata.plane_visitor.get(), lite_event.line,
          &lite_event.line->events(lite_event.event_idx));
      TraceEvent full_event;
      ConvertLiteTraceEventToFullTraceEvent(visitor, lite_event, lite_container,
                                            &local_name_table, &full_event,
                                            &arena);
      EXPECT_EQ(full_event.name(), "RD_RSP_BEAT_0");
      EXPECT_EQ(full_event.timestamp_ps(), 50);
    }
  }
  EXPECT_EQ(num_lite_hbm_events, 1);
}

TEST(XPlaneToTraceContainerTest,
     EmitsUnsignedProgramIdStringOnlyWhenHloModuleIsPresent) {
  XSpace xspace;
  CHECK_OK(ParseTextFormatFromString(
      R"pb(
        planes {
          name: "/device:GPU:0"
          lines {
            id: 50
            name: "Stream #50(Compute)"
            events {
              metadata_id: 1
              offset_ps: 1000
              duration_ps: 100
              stats { metadata_id: 10 str_value: "fusion.1" }
              stats {
                metadata_id: 11
                str_value: "p0_layer_60_64.local_recovery_prefill"
              }
              # static_cast<int64_t>(17991872812008296838ULL) ==
              # -454871261701254778
              stats { metadata_id: 12 int64_value: -454871261701254778 }
            }
            events {
              metadata_id: 2
              offset_ps: 2000
              duration_ps: 100
              stats { metadata_id: 10 str_value: "mpmd_map.3272" }
              stats { metadata_id: 11 str_value: "jit_generate" }
              stats { metadata_id: 12 uint64_value: 8623923867354161468 }
            }
            events {
              metadata_id: 3
              offset_ps: 3000
              duration_ps: 100
              stats { metadata_id: 10 str_value: "fusion.1" }
              stats { metadata_id: 12 uint64_value: 42 }
            }
            events {
              metadata_id: 4
              offset_ps: 4000
              duration_ps: 100
              stats { metadata_id: 11 str_value: "jit_prefill" }
              stats { metadata_id: 12 str_value: "9876543210" }
            }
            events {
              metadata_id: 5
              offset_ps: 5000
              duration_ps: 100
              stats { metadata_id: 11 str_value: "" }
              stats { metadata_id: 12 uint64_value: 99 }
            }
            events {
              metadata_id: 6
              offset_ps: 6000
              duration_ps: 100
              stats { metadata_id: 11 str_value: "standalone_module" }
            }
            events {
              metadata_id: 7
              offset_ps: 7000
              duration_ps: 100
              stats { metadata_id: 11 str_value: "jit_empty_pid" }
              stats { metadata_id: 12 str_value: "" }
            }
          }
          event_metadata {
            key: 1
            value: { id: 1 name: "gpu_kernel_msb" }
          }
          event_metadata {
            key: 2
            value: { id: 2 name: "gpu_kernel_uint" }
          }
          event_metadata {
            key: 3
            value: { id: 3 name: "tpu_xla_op" }
          }
          event_metadata {
            key: 4
            value: { id: 4 name: "gpu_kernel_str" }
          }
          event_metadata {
            key: 5
            value: { id: 5 name: "empty_hlo_module" }
          }
          event_metadata {
            key: 6
            value: { id: 6 name: "module_without_program_id" }
          }
          event_metadata {
            key: 7
            value: { id: 7 name: "empty_program_id" }
          }
          stat_metadata {
            key: 10
            value: { id: 10 name: "hlo_op" }
          }
          stat_metadata {
            key: 11
            value: { id: 11 name: "hlo_module" }
          }
          stat_metadata {
            key: 12
            value: { id: 12 name: "program_id" }
          }
        }
      )pb",
      &xspace));

  TraceEventsContainer container;
  ConvertXSpaceToTraceEventsContainer("localhost", xspace, &container);

  const auto& name_table = container.trace().name_table();
  absl::flat_hash_map<std::string,
                      absl::flat_hash_map<std::string, std::string>>
      args_by_event;
  container.ForAllEvents([&args_by_event,
                          &name_table](const TraceEvent& event) {
    auto& event_args = args_by_event[event.name()];
    if (!event.has_raw_data()) return;
    RawData raw_data;
    CHECK(raw_data.ParseFromString(event.raw_data()));
    for (const auto& arg : raw_data.args().arg()) {
      event_args[arg.name()] = arg.has_ref_value()
                                   ? name_table.at(arg.ref_value())
                                   : arg.str_value();
    }
  });

  ASSERT_TRUE(args_by_event.contains("gpu_kernel_msb"));
  EXPECT_EQ(args_by_event["gpu_kernel_msb"]["hlo_module"],
            "p0_layer_60_64.local_recovery_prefill");
  EXPECT_EQ(args_by_event["gpu_kernel_msb"]["program_id"],
            "17991872812008296838");

  ASSERT_TRUE(args_by_event.contains("gpu_kernel_uint"));
  EXPECT_EQ(args_by_event["gpu_kernel_uint"]["hlo_module"], "jit_generate");
  EXPECT_EQ(args_by_event["gpu_kernel_uint"]["program_id"],
            "8623923867354161468");

  ASSERT_TRUE(args_by_event.contains("tpu_xla_op"));
  EXPECT_EQ(args_by_event["tpu_xla_op"]["hlo_op"], "fusion.1");
  EXPECT_FALSE(args_by_event["tpu_xla_op"].contains("hlo_module"));
  EXPECT_FALSE(args_by_event["tpu_xla_op"].contains("program_id"));

  ASSERT_TRUE(args_by_event.contains("gpu_kernel_str"));
  EXPECT_EQ(args_by_event["gpu_kernel_str"]["hlo_module"], "jit_prefill");
  EXPECT_EQ(args_by_event["gpu_kernel_str"]["program_id"], "9876543210");

  ASSERT_TRUE(args_by_event.contains("empty_hlo_module"));
  EXPECT_EQ(args_by_event["empty_hlo_module"]["hlo_module"], "");
  EXPECT_FALSE(args_by_event["empty_hlo_module"].contains("program_id"));

  ASSERT_TRUE(args_by_event.contains("module_without_program_id"));
  EXPECT_EQ(args_by_event["module_without_program_id"]["hlo_module"],
            "standalone_module");
  EXPECT_FALSE(
      args_by_event["module_without_program_id"].contains("program_id"));

  ASSERT_TRUE(args_by_event.contains("empty_program_id"));
  EXPECT_EQ(args_by_event["empty_program_id"]["hlo_module"], "jit_empty_pid");
  EXPECT_FALSE(args_by_event["empty_program_id"].contains("program_id"));
}

}  // namespace

}  // namespace profiler
}  // namespace tensorflow
