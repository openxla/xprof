#ifndef THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_LITE_TRACE_EVENTS_VISIBILITY_H_
#define THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_LITE_TRACE_EVENTS_VISIBILITY_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xprof/convert/trace_viewer/lite_trace_events.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"

namespace tensorflow {
namespace profiler {

class TraceViewerVisibilityLite {
 public:
  explicit TraceViewerVisibilityLite(tsl::profiler::Timespan visible_span,
                                     uint64_t resolution_ps = 0)
      : visible_span_(visible_span), resolution_ps_(resolution_ps) {}

  bool Visible(const TraceEventLite& event,
               const TraceTrackMetadata& metadata) {
    if (visible_span_.Instant()) return true;

    tsl::profiler::Timespan span(event.timestamp_ps, event.duration_ps);
    if (!visible_span_.Overlaps(span)) return false;

    if (resolution_ps_ == 0) return true;

    return VisibleAtResolution(event, metadata);
  }

  bool VisibleAtResolution(const TraceEventLite& event,
                            const TraceTrackMetadata& metadata) {
    DCHECK_NE(resolution_ps_, 0);

    tsl::profiler::Timespan span(event.timestamp_ps, event.duration_ps);

    if (metadata.resource_id == UINT32_MAX || event.is_async) {
      CounterRowId counter_row_id(metadata.device_id, metadata.name_ref);
      auto iter = last_counter_timestamp_ps_.find(counter_row_id);
      bool found = (iter != last_counter_timestamp_ps_.end());
      bool visible =
          !found || ((event.timestamp_ps - iter->second) >= resolution_ps_);
      if (visible) {
        if (found) {
          iter->second = event.timestamp_ps;
        } else {
          last_counter_timestamp_ps_.emplace(counter_row_id,
                                             event.timestamp_ps);
        }
      }
      return visible;
    }

    // Complete & Async events
    bool visible = (span.duration_ps() >= resolution_ps_);

    auto& row = rows_[RowId(metadata.device_id, metadata.resource_id)];
    size_t depth = row.Depth(span.begin_ps());

    if (!visible) {
      auto last_end_timestamp_ps = row.LastEndTimestampPs(depth);
      visible = !last_end_timestamp_ps ||
                (span.begin_ps() - *last_end_timestamp_ps >= resolution_ps_);
    }

    if (event.flow_id != UINT64_MAX) {
      auto result = flows_.try_emplace(event.flow_id, visible);
      if (!visible) {
        if (result.second) {
          auto last_flow_timestamp_ps = row.LastFlowTimestampPs();
          result.first->second =
              !last_flow_timestamp_ps ||
              (span.end_ps() - *last_flow_timestamp_ps >= resolution_ps_);
        }
        visible = result.first->second;
      }
      if (event.flow_entry_type == TraceEvent::FLOW_END) {
        flows_.erase(event.flow_id);
      }
      if (visible) {
        row.SetLastFlowTimestampPs(span.end_ps());
      }
    }

    if (visible) {
      row.SetLastEndTimestampPs(depth, span.end_ps());
    }
    return visible;
  }

  void SetVisibleAtResolution(const TraceEventLite& event,
                              const TraceTrackMetadata& metadata) {
    DCHECK_NE(resolution_ps_, 0);
    tsl::profiler::Timespan span(event.timestamp_ps, event.duration_ps);

    if (metadata.resource_id == UINT32_MAX || event.is_async) {
      CounterRowId counter_row_id(metadata.device_id, metadata.name_ref);
      last_counter_timestamp_ps_.insert_or_assign(counter_row_id,
                                                   event.timestamp_ps);
      return;
    }

    auto& row = rows_[RowId(metadata.device_id, metadata.resource_id)];
    if (event.flow_id != UINT64_MAX) {
      if (event.flow_entry_type == TraceEvent::FLOW_END) {
        flows_.erase(event.flow_id);
      } else {
        flows_.try_emplace(event.flow_id, true);
      }
      row.SetLastFlowTimestampPs(span.end_ps());
    }
    size_t depth = row.Depth(span.begin_ps());
    row.SetLastEndTimestampPs(depth, span.end_ps());
  }

  tsl::profiler::Timespan VisibleSpan() const { return visible_span_; }
  uint64_t ResolutionPs() const { return resolution_ps_; }
  const absl::flat_hash_map<uint64_t, bool>& Flows() const { return flows_; }

 private:
  using RowId = std::pair<uint32_t, uint32_t>;
  using CounterRowId =
      std::pair<uint32_t, uint64_t>;  // name_ref instead of name string!

  class RowVisibility {
   public:
    size_t Depth(uint64_t begin_timestamp_ps) const {
      size_t depth = 0;
      for (; depth < last_end_timestamp_ps_.size(); ++depth) {
        if (last_end_timestamp_ps_[depth] <= begin_timestamp_ps) break;
      }
      return depth;
    }

    std::optional<uint64_t> LastEndTimestampPs(size_t depth) const {
      std::optional<uint64_t> result;
      if (depth < last_end_timestamp_ps_.size()) {
        result = last_end_timestamp_ps_[depth];
      }
      return result;
    }

    std::optional<uint64_t> LastFlowTimestampPs() const {
      return last_flow_timestamp_ps_;
    }

    void SetLastEndTimestampPs(size_t depth, uint64_t timestamp_ps) {
      last_end_timestamp_ps_.resize(depth);
      last_end_timestamp_ps_.push_back(timestamp_ps);
    }

    void SetLastFlowTimestampPs(uint64_t timestamp_ps) {
      last_flow_timestamp_ps_ = timestamp_ps;
    }

   private:
    std::vector<uint64_t> last_end_timestamp_ps_;
    std::optional<uint64_t> last_flow_timestamp_ps_;
  };

  tsl::profiler::Timespan visible_span_;
  uint64_t resolution_ps_;

  absl::flat_hash_map<RowId, RowVisibility> rows_;
  absl::flat_hash_map<uint64_t, bool> flows_;
  absl::flat_hash_map<CounterRowId, uint64_t> last_counter_timestamp_ps_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_TRACE_VIEWER_LITE_TRACE_EVENTS_VISIBILITY_H_
