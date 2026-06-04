// C API implementation wrapping profile_data_lib.
//
// Bridges the C++ ProfileData/ProfilePlane/ProfileLine/ProfileEvent classes
// to a flat C ABI that Python ctypes can consume directly.

#include "xprof/pywrap/profile_data_c_api.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

using tensorflow::profiler::XEvent;
using tensorflow::profiler::XLine;
using tensorflow::profiler::XPlane;
using tensorflow::profiler::XSpace;
using tensorflow::profiler::XStat;

namespace {

std::string resolve_stat_name(const XPlane* plane, int64_t metadata_id) {
  auto it = plane->stat_metadata().find(metadata_id);
  if (it != plane->stat_metadata().end()) {
    return it->second.name();
  }
  return std::to_string(metadata_id);
}

std::string resolve_event_name(const XPlane* plane, int64_t metadata_id) {
  auto it = plane->event_metadata().find(metadata_id);
  if (it != plane->event_metadata().end()) {
    return it->second.name();
  }
  return std::to_string(metadata_id);
}

std::string stat_value_str(const XStat& stat) {
  switch (stat.value_case()) {
    case XStat::kDoubleValue:
      return std::to_string(stat.double_value());
    case XStat::kUint64Value:
      return std::to_string(stat.uint64_value());
    case XStat::kInt64Value:
      return std::to_string(stat.int64_value());
    case XStat::kStrValue:
      return stat.str_value();
    case XStat::kRefValue: {
      return std::to_string(stat.ref_value());
    }
    case XStat::kBytesValue:
      return "<bytes>";
    default:
      return "";
  }
}

struct CProfileData {
  std::shared_ptr<XSpace> xspace;
  std::vector<const XPlane*> planes;
};

struct CProfilePlane {
  const XPlane* plane;
  std::shared_ptr<XSpace> xspace;
  std::vector<const XLine*> lines;
  mutable std::vector<std::string> stat_names;
  mutable std::vector<std::string> stat_values;
};

struct CProfileLine {
  const XLine* line;
  const XPlane* plane;
  std::shared_ptr<XSpace> xspace;
  int64_t line_timestamp_ns;
  std::vector<const XEvent*> events;
};

struct CProfileEvent {
  const XEvent* event;
  const XPlane* plane;
  int64_t line_timestamp_ns;
  std::shared_ptr<XSpace> xspace;
  mutable std::string resolved_name;
  mutable std::vector<std::string> stat_names;
  mutable std::vector<std::string> stat_values;
};

}  // namespace

extern "C" {

ProfileData profile_data_from_bytes(const char* buf, size_t len) {
  auto xspace = std::make_shared<XSpace>();
  if (!xspace->ParseFromString(absl::string_view(buf, len))) {
    return nullptr;
  }
  auto* data = new CProfileData;
  data->xspace = xspace;
  data->planes.reserve(xspace->planes_size());
  for (int i = 0; i < xspace->planes_size(); i++) {
    data->planes.push_back(&xspace->planes(i));
  }
  return static_cast<ProfileData>(data);
}

void profile_data_free(ProfileData data) {
  delete static_cast<CProfileData*>(data);
}

int profile_data_num_planes(ProfileData data) {
  return static_cast<int>(static_cast<CProfileData*>(data)->planes.size());
}

ProfilePlane profile_data_get_plane(ProfileData data, int index) {
  auto* d = static_cast<CProfileData*>(data);
  auto* p = new CProfilePlane;
  p->plane = d->planes[index];
  p->xspace = d->xspace;
  p->lines.reserve(p->plane->lines_size());
  for (int i = 0; i < p->plane->lines_size(); i++) {
    p->lines.push_back(&p->plane->lines(i));
  }
  p->stat_names.resize(p->plane->stats_size());
  p->stat_values.resize(p->plane->stats_size());
  return static_cast<ProfilePlane>(p);
}

const char* profile_plane_name(ProfilePlane plane) {
  return static_cast<CProfilePlane*>(plane)->plane->name().c_str();
}

int profile_plane_num_lines(ProfilePlane plane) {
  return static_cast<int>(static_cast<CProfilePlane*>(plane)->lines.size());
}

ProfileLine profile_plane_get_line(ProfilePlane plane, int index) {
  auto* pp = static_cast<CProfilePlane*>(plane);
  auto* l = new CProfileLine;
  l->line = pp->lines[index];
  l->plane = pp->plane;
  l->xspace = pp->xspace;
  l->line_timestamp_ns = l->line->timestamp_ns();
  l->events.reserve(l->line->events_size());
  for (int i = 0; i < l->line->events_size(); i++) {
    l->events.push_back(&l->line->events(i));
  }
  return static_cast<ProfileLine>(l);
}

int profile_plane_num_stats(ProfilePlane plane) {
  return static_cast<CProfilePlane*>(plane)->plane->stats_size();
}

void profile_plane_get_stat(ProfilePlane plane, int index,
                            const char** out_name, const char** out_value) {
  auto* pp = static_cast<CProfilePlane*>(plane);
  const auto& stat = pp->plane->stats(index);

  if (pp->stat_names[index].empty()) {
    pp->stat_names[index] = resolve_stat_name(pp->plane, stat.metadata_id());
  }
  *out_name = pp->stat_names[index].c_str();

  if (pp->stat_values[index].empty()) {
    pp->stat_values[index] = stat_value_str(stat);
  }
  *out_value = pp->stat_values[index].c_str();
}

void profile_plane_free(ProfilePlane plane) {
  delete static_cast<CProfilePlane*>(plane);
}

const char* profile_line_name(ProfileLine line) {
  auto* ll = static_cast<CProfileLine*>(line);
  return ll->line->name().c_str();
}

int profile_line_num_events(ProfileLine line) {
  return static_cast<int>(static_cast<CProfileLine*>(line)->events.size());
}

ProfileEvent profile_line_get_event(ProfileLine line, int index) {
  auto* ll = static_cast<CProfileLine*>(line);
  auto* e = new CProfileEvent;
  e->event = ll->events[index];
  e->plane = ll->plane;
  e->line_timestamp_ns = ll->line_timestamp_ns;
  e->xspace = ll->xspace;
  e->stat_names.resize(e->event->stats_size());
  e->stat_values.resize(e->event->stats_size());
  return static_cast<ProfileEvent>(e);
}

void profile_line_free(ProfileLine line) {
  delete static_cast<CProfileLine*>(line);
}

const char* profile_event_name(ProfileEvent event) {
  auto* ee = static_cast<CProfileEvent*>(event);
  if (ee->resolved_name.empty()) {
    ee->resolved_name = resolve_event_name(ee->plane, ee->event->metadata_id());
  }
  return ee->resolved_name.c_str();
}

double profile_event_start_ns(ProfileEvent event) {
  auto* ee = static_cast<CProfileEvent*>(event);
  // XEvent stores offset_ps relative to the line's timestamp_ns.
  // Convert: line_timestamp_ns + (offset_ps / 1000)
  return static_cast<double>(ee->line_timestamp_ns) +
         static_cast<double>(ee->event->offset_ps()) / 1000.0;
}

double profile_event_duration_ns(ProfileEvent event) {
  auto* ee = static_cast<CProfileEvent*>(event);
  return static_cast<double>(ee->event->duration_ps()) / 1000.0;
}

int profile_event_num_stats(ProfileEvent event) {
  return static_cast<CProfileEvent*>(event)->event->stats_size();
}

void profile_event_get_stat(ProfileEvent event, int index,
                            const char** out_name, const char** out_value) {
  auto* ee = static_cast<CProfileEvent*>(event);
  const auto& stat = ee->event->stats(index);

  if (ee->stat_names[index].empty()) {
    ee->stat_names[index] = resolve_stat_name(ee->plane, stat.metadata_id());
  }
  *out_name = ee->stat_names[index].c_str();

  if (ee->stat_values[index].empty()) {
    ee->stat_values[index] = stat_value_str(stat);
  }
  *out_value = ee->stat_values[index].c_str();
}

void profile_event_free(ProfileEvent event) {
  delete static_cast<CProfileEvent*>(event);
}

}  // extern "C"
