#include "frontend/app/components/trace_viewer_v2/timeline/timeline.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "imgui.h"
#include "imgui_internal.h"
#include "frontend/app/components/trace_viewer_v2/color/color_generator.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"
#include "frontend/app/components/trace_viewer_v2/event_data.h"
#include "frontend/app/components/trace_viewer_v2/fonts/fonts.h"
#include "frontend/app/components/trace_viewer_v2/helper/clipboard.h"
#include "frontend/app/components/trace_viewer_v2/helper/counter_formatter.h"
#include "frontend/app/components/trace_viewer_v2/helper/time_formatter.h"
#include "frontend/app/components/trace_viewer_v2/timeline/constants.h"
#include "frontend/app/components/trace_viewer_v2/timeline/draw_helpers.h"
#include "frontend/app/components/trace_viewer_v2/timeline/motion.h"
#include "frontend/app/components/trace_viewer_v2/timeline/time_range.h"
#include "frontend/app/components/trace_viewer_v2/timeline/timeline_hlo_dependencies.h"
#include "frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"

#ifdef __EMSCRIPTEN__
#include "frontend/app/components/trace_viewer_v2/imgui_webgpu_backend.h"
#endif

namespace traceviewer {
namespace {

using FallbackKey = std::tuple<ProcessId, ThreadId, absl::string_view>;

absl::string_view GetEventNameFromData(
    const FlameChartTimelineData& timeline_data, int event_index) {
  if (!timeline_data.interned_string_pool.empty() && event_index >= 0 &&
      static_cast<size_t>(event_index) <
          timeline_data.entry_name_indices.size()) {
    const uint32_t idx = timeline_data.entry_name_indices[event_index];
    if (idx < timeline_data.interned_string_pool.size()) {
      return timeline_data.interned_string_pool[idx];
    }
  }
  if (event_index >= 0 &&
      static_cast<size_t>(event_index) < timeline_data.entry_names.size()) {
    return timeline_data.entry_names[event_index];
  }
  return "";
}

EventId GetEventIdFromData(const FlameChartTimelineData& timeline_data,
                           int event_index) {
  if (event_index >= 0 &&
      static_cast<size_t>(event_index) < timeline_data.entry_event_ids.size()) {
    return timeline_data.entry_event_ids[event_index];
  }
  if (event_index >= 0 && static_cast<size_t>(event_index) <
                              timeline_data.entry_start_times.size()) {
    const Microseconds dur = (static_cast<size_t>(event_index) <
                              timeline_data.entry_total_times.size())
                                 ? timeline_data.entry_total_times[event_index]
                                 : 0.0;
    return GenerateEventId(GetEventNameFromData(timeline_data, event_index),
                           timeline_data.entry_start_times[event_index], dur);
  }
  return 0;
}

absl::flat_hash_map<FallbackKey, std::vector<int>> BuildFallbackMap(
    const FlameChartTimelineData& timeline_data,
    const absl::flat_hash_set<FallbackKey>& wanted_keys) {
  absl::flat_hash_map<FallbackKey, std::vector<int>> fallback_map;
  if (wanted_keys.empty()) {
    return fallback_map;
  }
  const int num_loaded = static_cast<int>(std::max({
      timeline_data.entry_start_times.size(),
      timeline_data.entry_names.size(),
      timeline_data.entry_name_indices.size(),
  }));
  fallback_map.reserve(wanted_keys.size());

  const int num_levels = timeline_data.total_levels();
  if (num_levels > 0 &&
      static_cast<int>(timeline_data.level_event_indices.size()) ==
          num_loaded) {
    absl::flat_hash_map<std::pair<ProcessId, ThreadId>,
                        absl::flat_hash_set<absl::string_view>>
        wanted_names;
    wanted_names.reserve(wanted_keys.size());
    for (const auto& [key_pid, key_tid, key_name] : wanted_keys) {
      wanted_names[{key_pid, key_tid}].insert(key_name);
    }
    for (int level = 0; level < num_levels; ++level) {
      const Group* grp = timeline_data.FindGroupForLevel(level);
      const ProcessId pid = grp != nullptr ? grp->pid : 0;
      const ThreadId tid = grp != nullptr ? grp->tid : 0;
      const auto names_it = wanted_names.find({pid, tid});
      if (names_it == wanted_names.end()) {
        continue;
      }
      for (const int i : timeline_data.level_events(level)) {
        const absl::string_view name = GetEventNameFromData(timeline_data, i);
        if (!names_it->second.contains(name)) {
          continue;
        }
        fallback_map[FallbackKey(pid, tid, name)].push_back(i);
      }
    }
    for (auto& [key, bucket] : fallback_map) {
      absl::c_sort(bucket);
    }
    return fallback_map;
  }

  for (int i = 0; i < num_loaded; ++i) {
    ProcessId loaded_pid = 0;
    ThreadId loaded_tid = 0;
    if (static_cast<size_t>(i) < timeline_data.entry_levels.size()) {
      if (const Group* grp =
              timeline_data.FindGroupForLevel(timeline_data.entry_levels[i])) {
        loaded_pid = grp->pid;
        loaded_tid = grp->tid;
      }
    }
    const absl::string_view loaded_name =
        GetEventNameFromData(timeline_data, i);
    const FallbackKey key(loaded_pid, loaded_tid, loaded_name);
    if (wanted_keys.contains(key)) {
      fallback_map[key].push_back(i);
    }
  }
  return fallback_map;
}

int FindFallbackMatchedIndex(
    const absl::flat_hash_map<FallbackKey, std::vector<int>>& fallback_map,
    const FlameChartTimelineData& timeline_data, ProcessId pid, ThreadId tid,
    absl::string_view name, Microseconds start_time, Microseconds duration,
    absl::string_view target_group_name = "") {
  const auto bucket_it = fallback_map.find(FallbackKey(pid, tid, name));
  if (bucket_it == fallback_map.end()) {
    return -1;
  }
  for (const int i : bucket_it->second) {
    if (!target_group_name.empty() &&
        static_cast<size_t>(i) < timeline_data.entry_levels.size()) {
      const Group* grp =
          timeline_data.FindGroupForLevel(timeline_data.entry_levels[i]);
      if (grp == nullptr || grp->name != target_group_name) {
        continue;
      }
    }
    const Microseconds loaded_start =
        (static_cast<size_t>(i) < timeline_data.entry_start_times.size())
            ? timeline_data.entry_start_times[i]
            : 0.0;
    const Microseconds loaded_dur =
        (static_cast<size_t>(i) < timeline_data.entry_total_times.size())
            ? timeline_data.entry_total_times[i]
            : 0.0;
    if (std::abs(loaded_start - start_time) < 1.0 &&
        std::abs(loaded_dur - duration) < 1.0) {
      return i;
    }
  }
  return -1;
}

void ApplySnappingToEdge(Microseconds time, Microseconds threshold,
                         const std::vector<Microseconds>& candidates,
                         Microseconds& best_diff, Microseconds& snapped_edge,
                         bool& snapped) {
  auto it =
      std::lower_bound(candidates.begin(), candidates.end(), time - threshold);
  for (; it != candidates.end() && *it <= time + threshold; ++it) {
    Microseconds diff = std::abs(time - *it);
    if (diff < best_diff) {
      best_diff = diff;
      snapped_edge = *it;
      snapped = true;
    }
  }
}
// Calculates a speed multiplier based on how long a key has been held down and
// whether the Shift modifier key is pressed.
// Provides acceleration for continuous actions like panning and zooming.
float GetSpeedMultiplier(const ImGuiIO& io, ImGuiKey key,
                         float shift_factor = 1.0f) {
  const float down_duration = ImGui::GetKeyData(key)->DownDuration;
  float multiplier = 1.0f;

  if (down_duration >= kAccelerateThreshold) {
    const float accelerated_time = down_duration - kAccelerateThreshold;
    multiplier +=
        std::min(accelerated_time * kAccelerateRate, kMaxAccelerateFactor);
  }

  if (io.KeyShift) {
    multiplier *= shift_factor;
  }

  return multiplier;
}

// Extracts process sort indices from metadata events in search results.
absl::flat_hash_map<ProcessId, uint32_t> GetProcessSortIndices(
    const ParsedTraceEvents& search_results) {
  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices;
  for (const TraceEvent& event : search_results.flame_events) {
    if (event.ph == Phase::kMetadata && event.name == kProcessSortIndex) {
      if (const auto it = event.args.find(kSortIndex); it != event.args.end()) {
        double sort_index_double;
        if (absl::SimpleAtod(it->second, &sort_index_double)) {
          process_sort_indices[event.pid] =
              static_cast<uint32_t>(sort_index_double);
        }
      }
    }
  }
  return process_sort_indices;
}

// Returns `color` with its alpha channel multiplied by `opacity` in [0, 1].
ImU32 WithOpacity(ImU32 color, float opacity) {
  const float alpha = static_cast<float>((color >> IM_COL32_A_SHIFT) & 0xFF);
  const ImU32 new_alpha =
      static_cast<ImU32>(std::clamp(alpha * opacity, 0.0f, 255.0f) + 0.5f);
  return (color & ~IM_COL32_A_MASK) | (new_alpha << IM_COL32_A_SHIFT);
}

// Linearly interpolates between two colors, channel by channel (including
// alpha). `t` = 0 returns `from` and `t` = 1 returns `to`.
ImU32 MixColors(ImU32 from, ImU32 to, float t) {
  auto channel = [&](int shift) {
    const float a = static_cast<float>((from >> shift) & 0xFF);
    const float b = static_cast<float>((to >> shift) & 0xFF);
    return static_cast<ImU32>(std::clamp(a + (b - a) * t, 0.0f, 255.0f) + 0.5f)
           << shift;
  };
  return channel(IM_COL32_R_SHIFT) | channel(IM_COL32_G_SHIFT) |
         channel(IM_COL32_B_SHIFT) | channel(IM_COL32_A_SHIFT);
}

// Draws a vertical "drop shadow": a gradient from `opacity` black at `top` to
// transparent at `top + height`, spanning [left, right). This mimics a CSS
// `box-shadow` cast downwards by an element whose bottom edge is at `top`.
void DrawDropShadow(ImDrawList* draw_list, Pixel left, Pixel right, Pixel top,
                    Pixel height, float opacity) {
  const ImU32 top_color = WithOpacity(kBlackColor, opacity);
  const ImU32 bottom_color = WithOpacity(kBlackColor, 0.0f);
  draw_list->AddRectFilledMultiColor(ImVec2(left, top),
                                     ImVec2(right, top + height), top_color,
                                     top_color, bottom_color, bottom_color);
}

// Draws a rounded "pill" highlight inset inside the row [min, max], the hover
// style of modern sidebars.
void DrawRowPill(ImDrawList* draw_list, const ImVec2& min, const ImVec2& max,
                 ImU32 color) {
  // Edges sit on pixel centers, so the anti-aliased fringe ImGui adds around
  // rounded shapes (+/- 0.5px) falls on pixel boundaries and never straddles a
  // pixel center. This keeps the rasterized edge identical across machines.
  const ImVec2 pill_min(std::floor(min.x + kTrackRowPillInsetX) + 0.5f,
                        std::floor(min.y + kTrackRowPillInsetY) + 0.5f);
  const ImVec2 pill_max(std::ceil(max.x - kTrackRowPillInsetX) - 0.5f,
                        std::ceil(max.y - kTrackRowPillInsetY) - 0.5f);
  if (pill_max.x <= pill_min.x || pill_max.y <= pill_min.y) return;
  draw_list->AddRectFilled(pill_min, pill_max, color, kTrackRowPillRounding);
}

// Draws `text` in upper case with extra letter spacing (CSS `letter-spacing`),
// the style used for small section labels. Returns the drawn width. Draws one
// glyph at a time to avoid allocating an upper-cased copy every frame.
Pixel DrawSpacedUppercaseText(ImDrawList* draw_list, ImFont* font,
                              absl::string_view text, const ImVec2& pos,
                              ImU32 color) {
  ImGui::PushFont(font);
  Pixel x = pos.x;
  for (size_t i = 0; i < text.size(); ++i) {
    const char glyph[2] = {absl::ascii_toupper(text[i]), '\0'};
    draw_list->AddText(ImVec2(x, pos.y), color, glyph, glyph + 1);
    x += ImGui::CalcTextSize(glyph, glyph + 1).x;
    if (i + 1 < text.size()) x += kSectionLabelLetterSpacing;
  }
  ImGui::PopFont();
  return x - pos.x;
}

// Shows `text` in a tooltip styled like a modern floating card: rounded
// corners, comfortable padding and a hairline border.
void ShowCardTooltip(absl::string_view text) {
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, kCardTooltipRounding);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, kCardTooltipPadding);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 1.0f);
  ImGui::SetTooltip("%.*s", static_cast<int>(text.size()), text.data());
  ImGui::PopStyleVar(3);
}

// Draws a stroked chevron centered at `center`, rotated by `openness` in
// [0, 1]: 0 points right (collapsed), 1 points down (expanded), and values in
// between rotate smoothly. `size` is the length of the chevron's open side.
void DrawChevron(ImDrawList* draw_list, const ImVec2& center, Pixel size,
                 float openness, ImU32 color) {
  const Pixel half = size * 0.5f;
  const Pixel depth = size * 0.25f;
  // The down-pointing chevron, rotated by -90 degrees when fully collapsed.
  const float angle = -(1.0f - std::clamp(openness, 0.0f, 1.0f)) * IM_PI * 0.5f;
  const float cos_a = std::cos(angle);
  const float sin_a = std::sin(angle);
  const ImVec2 down[3] = {ImVec2(-half, -depth), ImVec2(0.0f, depth),
                          ImVec2(half, -depth)};
  ImVec2 points[3];
  for (int i = 0; i < 3; ++i) {
    points[i] = ImVec2(center.x + down[i].x * cos_a - down[i].y * sin_a,
                       center.y + down[i].x * sin_a + down[i].y * cos_a);
  }
  draw_list->AddPolyline(points, 3, color, ImDrawFlags_None,
                         kTrackChevronThickness);
}

// Draws the expand/collapse chevron of a label row. Collapsed rows draw it at
// full strength inside a small rounded "button" so their state stands out at a
// glance; expanded rows draw a quieter chevron. `openness` (see DrawChevron)
// cross-fades between the two looks while the chevron rotates, and
// `hover_emphasis` in [0, 1] brightens it.
void DrawExpandChevron(ImDrawList* draw_list, const ImVec2& center,
                       float openness, float hover_emphasis,
                       ImU32 foreground_color) {
  const float collapsedness = 1.0f - std::clamp(openness, 0.0f, 1.0f);
  if (collapsedness > 0.0f) {
    // Edges on pixel centers keep the anti-aliased fringe deterministic.
    const Pixel half = kTrackChevronButtonSize * 0.5f;
    const float button_opacity = kTrackCollapsedChevronButtonOpacity +
                                 (kTrackCollapsedChevronButtonHoverOpacity -
                                  kTrackCollapsedChevronButtonOpacity) *
                                     hover_emphasis;
    draw_list->AddRectFilled(
        ImVec2(std::floor(center.x - half) + 0.5f,
               std::floor(center.y - half) + 0.5f),
        ImVec2(std::floor(center.x + half) + 0.5f,
               std::floor(center.y + half) + 0.5f),
        WithOpacity(foreground_color, button_opacity * collapsedness),
        kTrackChevronButtonRounding);
  }
  const float emphasis = std::max(collapsedness, hover_emphasis);
  DrawChevron(draw_list, center, kTrackChevronSize, openness,
              WithOpacity(foreground_color,
                          kTrackMutedTextOpacity +
                              (1.0f - kTrackMutedTextOpacity) * emphasis));
}

// Draws a "stack of cards" hint below a collapsed row, between `left` and
// `right`: two short hairlines in the track's accent color, each narrower and
// fainter than the one above, suggesting content folded underneath.
void DrawCollapsedStackHint(ImDrawList* draw_list, Pixel left, Pixel right,
                            Pixel row_bottom, ImU32 accent_color) {
  for (int layer = 0; layer < 2; ++layer) {
    const Pixel inset = kTrackCollapsedStackInset * (layer + 1);
    // Lines sit on pixel centers so each layer stays a crisp 1px line.
    const Pixel y =
        std::floor(row_bottom) + 1.5f + kTrackCollapsedStackSpacing * layer;
    if (right - inset <= left + inset) return;
    draw_list->AddLine(
        ImVec2(left + inset, y), ImVec2(right - inset, y),
        WithOpacity(accent_color, layer == 0
                                      ? kTrackCollapsedStackOpacity
                                      : kTrackCollapsedStackOpacity * 0.5f),
        1.0f);
  }
}

// Returns the line height of `font` without pushing it. The fonts are never
// scaled, so their line height is their size. Falls back to the current font
// if `font` is not loaded.
Pixel FontLineHeight(const ImFont* font) {
  return font != nullptr ? font->LegacySize : ImGui::GetFontSize();
}

// Returns the width of the count chip drawn by `DrawCountChip` for `text`.
Pixel CountChipWidth(absl::string_view text) {
  ImGui::PushFont(traceviewer::fonts::label_small);
  const Pixel width =
      ImGui::CalcTextSize(text.data(), text.data() + text.size()).x +
      2.0f * kTrackCountChipPaddingX;
  ImGui::PopFont();
  return width;
}

// Draws a rounded "pill" containing `text`, right-aligned to `right_x` and
// vertically centered on `center_y`. Returns the pill's left edge.
Pixel DrawCountChip(ImDrawList* draw_list, absl::string_view text,
                    Pixel right_x, Pixel center_y, ImU32 background_color,
                    ImU32 text_color) {
  ImGui::PushFont(traceviewer::fonts::label_small);
  const ImVec2 text_size =
      ImGui::CalcTextSize(text.data(), text.data() + text.size());
  const Pixel width = text_size.x + 2.0f * kTrackCountChipPaddingX;
  const ImVec2 chip_min(right_x - width,
                        center_y - kTrackCountChipHeight * 0.5f);
  const ImVec2 chip_max(right_x, center_y + kTrackCountChipHeight * 0.5f);
  draw_list->AddRectFilled(chip_min, chip_max, background_color,
                           kTrackCountChipHeight * 0.5f);
  draw_list->AddText(ImVec2(chip_min.x + kTrackCountChipPaddingX,
                            std::round(center_y - text_size.y * 0.5f)),
                     text_color, text.data(), text.data() + text.size());
  ImGui::PopFont();
  return chip_min.x;
}

// Draws `text` with the given font and color in the box starting at `pos`,
// truncating it with an ellipsis if it would extend past `max_x`. Returns true
// if the text was truncated.
bool DrawTextWithEllipsis(ImFont* font, absl::string_view text, ImU32 color,
                          const ImVec2& pos, Pixel max_x) {
  ImGui::PushFont(font);
  const char* text_end = text.data() + text.size();
  const ImVec2 text_size = ImGui::CalcTextSize(text.data(), text_end);
  const bool truncated = pos.x + text_size.x > max_x;
  ImGui::PushStyleColor(ImGuiCol_Text, color);
  ImGui::RenderTextEllipsis(ImGui::GetWindowDrawList(), pos,
                            ImVec2(max_x, pos.y + text_size.y), max_x,
                            text.data(), text_end, &text_size);
  ImGui::PopStyleColor();
  ImGui::PopFont();
  return truncated;
}

// Returns the width reserved on the right side of a process label row for the
// track management (pin and hide) buttons.
Pixel CalculateTrackManagementButtonWidth() {
  return 2.0f * kTrackIconButtonSize + kTrackIconButtonGap +
         kTrackIconButtonMarginRight;
}

// The result of `TrackIconButton`.
struct IconButtonState {
  bool clicked = false;
  bool hovered = false;
  // The animated hover emphasis in [0, 1].
  float emphasis = 0.0f;
  ImVec2 center;
};

// Submits a square icon button at the cursor, vertically centered in a row of
// `row_height`, and draws its background: a circular "state layer" while
// hovered (stronger while pressed, as on Material icon buttons) or a faint
// persistent circle while `active` (e.g. the track is pinned). The caller draws
// the icon at the returned center. `tooltip` is shown, card-styled, after the
// standard hover delay.
IconButtonState TrackIconButton(
    const char* id, Pixel row_height, bool active, absl::string_view tooltip,
    ImU32 foreground_color,
    absl::FunctionRef<float(ImGuiID, bool)> hover_fade) {
  IconButtonState state;
  const ImVec2 cursor = ImGui::GetCursorScreenPos();
  const Pixel top = std::floor(
      cursor.y + std::max(0.0f, (row_height - kTrackIconButtonSize) * 0.5f));
  const ImVec2 min(std::floor(cursor.x), top);
  state.center = ImVec2(min.x + kTrackIconButtonSize * 0.5f,
                        min.y + kTrackIconButtonSize * 0.5f);
  ImGui::SetCursorScreenPos(min);
  ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
  state.clicked = ImGui::InvisibleButton(
      id, ImVec2(kTrackIconButtonSize, kTrackIconButtonSize),
      ImGuiButtonFlags_PressedOnClick);
  ImGui::PopItemFlag();
  state.hovered = ImGui::IsItemHovered();
  const bool pressed = ImGui::IsItemActive();
  state.emphasis = hover_fade(ImGui::GetItemID(), state.hovered);

  ImDrawList* const draw_list = ImGui::GetWindowDrawList();
  const Pixel radius = kTrackIconButtonSize * 0.5f;
  if (active) {
    draw_list->AddCircleFilled(
        state.center, radius,
        WithOpacity(foreground_color, kIconButtonActiveOpacity),
        /*num_segments=*/24);
  }
  if (pressed) {
    draw_list->AddCircleFilled(
        state.center, radius,
        WithOpacity(foreground_color, kIconButtonPressedOpacity),
        /*num_segments=*/24);
  } else if (state.emphasis > 0.0f) {
    // The hover circle grows out from the icon as it fades in, like a
    // Material state layer.
    const float scale = kMotionIconRippleStartScale +
                        (1.0f - kMotionIconRippleStartScale) * state.emphasis;
    draw_list->AddCircleFilled(
        state.center, radius * scale,
        WithOpacity(foreground_color, kIconButtonHoverOpacity * state.emphasis),
        /*num_segments=*/24);
  }
  if (state.hovered) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip)) {
      ShowCardTooltip(tooltip);
    }
  }
  return state;
}

// Returns the color of a track icon button's glyph: full strength while hovered
// or active, muted otherwise.
ImU32 TrackIconColor(const IconButtonState& state, bool active,
                     ImU32 foreground_color) {
  const float emphasis = active ? 1.0f : state.emphasis;
  return WithOpacity(
      foreground_color,
      kTrackMutedTextOpacity + (1.0f - kTrackMutedTextOpacity) * emphasis);
}

// Checks if a group is one of the virtual headers (Hidden or All).
bool IsVirtualHeader(const Group* group) {
  return group->nesting_level == kHeaderNestingLevel &&
         (group->name == kHiddenHeaderName || group->name == kAllHeaderName ||
          group->name == kPinnedHeaderName);
}

// Returns true if any argument key or value in `args` contains `query`
// (case-insensitive), ignoring synthetic default HLO module entries.
bool MatchesArgsMap(const absl::flat_hash_map<std::string, std::string>& args,
                    absl::string_view query) {
  for (const auto& [key, value] : args) {
    if (key == kHloModule && value == kHloModuleDefault) {
      continue;
    }
    if (absl::StrContainsIgnoreCase(key, query) ||
        absl::StrContainsIgnoreCase(value, query)) {
      return true;
    }
  }
  return false;
}

// Returns true if the event at `index` in `data` matches `query` by event name,
// dense `entry_args`, `sparse_entry_args`, or columnar event args
// (case-insensitive).
bool MatchesSearchQuery(const FlameChartTimelineData& data, int index,
                        absl::string_view query) {
  if (index < 0) {
    return false;
  }
  const absl::string_view name = GetEventNameFromData(data, index);
  if (!name.empty() && absl::StrContainsIgnoreCase(name, query)) {
    return true;
  }
  const size_t idx = static_cast<size_t>(index);
  if (idx < data.entry_args.size()) {
    return MatchesArgsMap(data.entry_args[idx], query);
  }
  if (const auto it = data.sparse_entry_args.find(idx);
      it != data.sparse_entry_args.end() && MatchesArgsMap(it->second, query)) {
    return true;
  }
  const bool has_columnar_args =
      data.non_zero_uids_by_event_index.contains(index) ||
      data.sparse_entry_group_ids.Find(idx).has_value() ||
      !data.GetEntryHloOp(idx).empty() || data.GetEntryHloModuleIndex(idx) != 0;
  return has_columnar_args && MatchesArgsMap(data.GetEntryArgs(idx), query);
}
}  // namespace

int Timeline::GetNextGroupStartLevel(const FlameChartTimelineData& data,
                                     int group_index) {
  if (group_index >= 0 && group_index < data.groups.size()) {
    const auto& group = data.groups[group_index];
    if (group.has_children && group.nesting_level == kProcessNestingLevel) {
      return group.start_level;
    }
    return group.start_level + group.level_count;
  }
  return data.total_levels();
}

// Finds the index of the first visible ancestor (or the group itself if it is
// visible) for a given group index.
//
// During layout pre-computation (`UpdateLevelPositions`), any child group
// nesting under a collapsed (collapsed == !expanded) parent will have its
// visibility flag (`group_visible_[index]`) set to false. All of its
// descendants form a contiguous range of invisible groups because of the
// parent's collapsed state.
//
// If the queried group index points to an invisible group (for instance, when
// locating groups from physical screen coordinates or user interactions), we
// backtrack (decrement index) to find the nearest ancestor group that is
// currently visible.
int Timeline::FindFirstVisibleAncestorIndex(int start_idx) const {
  while (start_idx > 0 && start_idx < group_visible_.size() &&
         !group_visible_[start_idx]) {
    start_idx--;
  }
  return start_idx;
}

Pixel Timeline::GetGroupTop(const Group* group) const {
  if (group == nullptr) return 0.0f;
  if (group == &header_hidden_) return header_hidden_offset_;
  if (group == &header_all_) return header_all_offset_;
  if (group == &header_pinned_) return header_pinned_offset_;

  if (timeline_data_.groups.empty()) return 0.0f;

  const Group* const data = timeline_data_.groups.data();
  const size_t size = timeline_data_.groups.size();
  if (std::less<const Group*>()(group, data) ||
      std::greater_equal<const Group*>()(group, data + size)) {
    return 0.0f;
  }

  const int group_index = static_cast<int>(group - data);
  return group_offsets_[group_index];
}

Pixel Timeline::GetGroupBottom(const Group* group) const {
  if (group == nullptr) return 0.0f;
  if (group == &header_hidden_) {
    return header_hidden_offset_ + kVirtualHeaderHeight;
  }
  if (group == &header_all_) {
    return header_all_offset_ + kVirtualHeaderHeight;
  }
  if (group == &header_pinned_) {
    return header_pinned_offset_ + kVirtualHeaderHeight;
  }

  if (timeline_data_.groups.empty()) return 0.0f;

  const Group* const data = timeline_data_.groups.data();
  const size_t size = timeline_data_.groups.size();
  if (std::less<const Group*>()(group, data) ||
      std::greater_equal<const Group*>()(group, data + size)) {
    return 0.0f;
  }

  const int group_index = static_cast<int>(group - data);
  return group_offsets_[group_index] + group_heights_[group_index];
}

void Timeline::BuildFlattenedGroups(const FlameChartTimelineData& data) {
  flattened_groups_.clear();
  const int group_count = data.groups.size();

  all_processes_count_ = 0;
  hidden_processes_count_ = 0;
  pinned_processes_count_ = 0;

  if (data.groups.empty()) return;

  // Fast-path: when track management (hiding/pinning) is disabled, copy the
  // raw groups sequence directly to avoid categorization overhead or inserting
  // virtual headers.
  if (!track_management_enabled_) {
    flattened_groups_.reserve(group_count);
    for (int i = 0; i < group_count; ++i) {
      flattened_groups_.push_back(&data.groups[i]);
      if (data.groups[i].nesting_level == kProcessNestingLevel) {
        all_processes_count_++;
      }
    }
    return;
  }

  // Since a process and all its nested subtracks form a visual block, we
  // propagate the process's status (hidden/pinned) down to its descendant
  // tracks. Retaining this state across iterations avoids performing
  // redundant string-key lookup queries on `hidden_track_names_` and
  // `pinned_track_names_` for every subtrack.
  std::vector<const Group*> hidden_groups;
  std::vector<const Group*> pinned_groups;
  std::vector<const Group*> all_groups;
  CategorizeGroupsForTrackManagement(data, hidden_groups, pinned_groups,
                                     all_groups);

  // Reserve capacity on flattened_groups_ to prevent multiple internal
  // reallocations as elements are added (headers + groups).
  flattened_groups_.reserve(3 + hidden_groups.size() + pinned_groups.size() +
                            all_groups.size());

  // 1. Hidden processes / "Hidden" Header
  flattened_groups_.push_back(&header_hidden_);
  // 2. Hidden processes and their children
  flattened_groups_.insert(flattened_groups_.end(), hidden_groups.begin(),
                           hidden_groups.end());

  // 3. Pinned processes / "Pinned" Header
  flattened_groups_.push_back(&header_pinned_);
  // 4. Pinned processes and their children
  flattened_groups_.insert(flattened_groups_.end(), pinned_groups.begin(),
                           pinned_groups.end());

  // 5. All Processes / "All" Header
  flattened_groups_.push_back(&header_all_);
  // 6. All processes and their children
  flattened_groups_.insert(flattened_groups_.end(), all_groups.begin(),
                           all_groups.end());
}

void Timeline::CategorizeGroupsForTrackManagement(
    const FlameChartTimelineData& data,
    std::vector<const Group*>& hidden_groups,
    std::vector<const Group*>& pinned_groups,
    std::vector<const Group*>& all_groups) {
  const int group_count = data.groups.size();
  hidden_groups.reserve(group_count);
  pinned_groups.reserve(group_count);
  all_groups.reserve(group_count);

  bool current_process_hidden = false;
  bool current_process_pinned = false;

  // Groups are expected in DFS pre-order: each process is immediately followed
  // by its child thread/counter tracks.
  for (int i = 0; i < group_count; ++i) {
    const Group& group = data.groups[i];
    if (group.nesting_level == kProcessNestingLevel) {
      current_process_hidden = hidden_track_names_.contains(group.name);
      current_process_pinned =
          !current_process_hidden && pinned_track_names_.contains(group.name);

      if (current_process_hidden) {
        hidden_processes_count_++;
      } else if (current_process_pinned) {
        pinned_processes_count_++;
      } else {
        all_processes_count_++;
      }
    }

    if (current_process_hidden) {
      hidden_groups.push_back(&data.groups[i]);
    } else if (current_process_pinned) {
      pinned_groups.push_back(&data.groups[i]);
    } else {
      all_groups.push_back(&data.groups[i]);
    }
  }
}

void Timeline::UpdateLevelPositions(const FlameChartTimelineData& data) {
  const int level_count = data.total_levels();
  const int group_count = data.groups.size();

  // The section header offsets currently on screen, so that an animated
  // layout change starts from them.
  const Pixel previous_header_offsets[3] = {
      header_hidden_offset_, header_pinned_offset_, header_all_offset_};

  // Populate flattened_groups_ based on data
  BuildFlattenedGroups(data);

  std::vector<Pixel> new_visible_level_offsets(level_count, 0.0f);
  std::vector<Pixel> new_group_offsets(group_count + 1, 0.0f);
  std::vector<Pixel> new_group_heights(group_count, 0.0f);
  std::vector<bool> new_group_visible(group_count, true);
  std::vector<bool> new_group_striped(group_count, false);
  // Whether the next visible child track gets the alternating row stripe.
  bool next_child_striped = false;

  Pixel current_offset =
      ImGui::GetCurrentContext() ? ImGui::GetStyle().CellPadding.y : 0.0f;

  int hidden_nesting_level = std::numeric_limits<int>::max();
  Pixel hidden_group_center_y = 0.0f;
  bool has_visible_group = false;

  // Track collapsed status of headers
  bool section_collapsed = false;

  for (const Group* group_ptr : flattened_groups_) {
    if (group_ptr->nesting_level == kHeaderNestingLevel) {
      if (group_ptr->name == kAllHeaderName) {
        header_all_offset_ = current_offset;
        section_collapsed = !header_all_expanded_;
        current_offset += kVirtualHeaderHeight;
        continue;
      }
      if (group_ptr->name == kHiddenHeaderName) {
        header_hidden_offset_ = current_offset;
        section_collapsed = !header_hidden_expanded_;
        current_offset += kVirtualHeaderHeight;
        // Reset collapsed ancestor tracker between sections
        hidden_nesting_level = std::numeric_limits<int>::max();
        continue;
      }
      if (group_ptr->name == kPinnedHeaderName) {
        header_pinned_offset_ = current_offset;
        section_collapsed = !header_pinned_expanded_;
        current_offset += kVirtualHeaderHeight;
        // Reset collapsed ancestor tracker between sections
        hidden_nesting_level = std::numeric_limits<int>::max();
        continue;
      }
    }

    // Now we are dealing with a standard group track
    const int group_index = group_ptr - &data.groups[0];
    const Group& group = *group_ptr;

    // If the whole header section is collapsed, this group disappears.
    if (section_collapsed) {
      new_group_offsets[group_index] = current_offset;
      new_group_visible[group_index] = false;
      const int next_group_start_level =
          GetNextGroupStartLevel(data, group_index);
      for (int level = group.start_level; level < next_group_start_level;
           ++level) {
        if (level < level_count) {
          // Point level offset to current header offset
          new_visible_level_offsets[level] = current_offset;
        }
      }
      continue;
    }

    if (group.nesting_level <= hidden_nesting_level) {
      hidden_nesting_level = std::numeric_limits<int>::max();
    }

    const int next_group_start_level =
        GetNextGroupStartLevel(data, group_index);

    if (hidden_nesting_level != std::numeric_limits<int>::max()) {
      new_group_offsets[group_index] = current_offset;
      new_group_visible[group_index] = false;
      for (int level = group.start_level; level < next_group_start_level;
           ++level) {
        if (level < level_count) {
          new_visible_level_offsets[level] = hidden_group_center_y;
        }
      }
      continue;
    }

    if (has_visible_group) {
      current_offset += (group.nesting_level == kProcessNestingLevel)
                            ? kProcessTrackGap
                            : kThreadTrackGap;
    }

    new_group_offsets[group_index] = current_offset;
    has_visible_group = true;

    if (group.nesting_level <= kProcessNestingLevel) {
      next_child_striped = false;
    } else {
      new_group_striped[group_index] = next_child_striped;
      next_child_striped = !next_child_striped;
    }

    const bool has_children = group.has_children;
    const bool has_multiple_levels =
        next_group_start_level - group.start_level > 1;

    const bool expandable = group.type == Group::Type::kFlame &&
                            (has_children || has_multiple_levels);

    const bool is_collapsed = expandable && !group.expanded;

    Pixel group_height = kEventHeight;
    if (group.nesting_level == kProcessNestingLevel) {
      group_height = kProcessTrackHeight;
    } else if (!is_collapsed) {
      if (group.type == Group::Type::kCounter) {
        group_height = kCounterTrackHeight;
      } else if (group.type == Group::Type::kFlame) {
        group_height = std::max(1, next_group_start_level - group.start_level) *
                       (kEventHeight + kEventPaddingBottom);
      }
    }

    if (is_collapsed &&
        hidden_nesting_level == std::numeric_limits<int>::max()) {
      hidden_nesting_level = group.nesting_level;
      hidden_group_center_y = current_offset + group_height * 0.5f;
    }

    const int start_level = group.start_level;

    if (is_collapsed) {
      for (int level = start_level; level < next_group_start_level; ++level) {
        if (level < level_count) {
          new_visible_level_offsets[level] =
              current_offset + group_height * 0.5f;
        }
      }
    } else {
      for (int level = start_level; level < next_group_start_level; ++level) {
        if (level < level_count) {
          new_visible_level_offsets[level] =
              current_offset +
              (level - start_level) * (kEventHeight + kEventPaddingBottom) +
              kEventHeight * 0.5f;
        }
      }
    }

    new_group_heights[group_index] = group_height;
    current_offset += group_height;
  }

  // Set the dummy offset at the very end
  new_group_offsets[group_count] = current_offset;

  ApplyLayout(data, std::move(new_group_offsets), std::move(new_group_heights),
              std::move(new_visible_level_offsets), new_group_visible,
              previous_header_offsets);
  group_visible_ = std::move(new_group_visible);
  group_striped_ = std::move(new_group_striped);
}

void Timeline::set_animations_enabled(bool enabled) {
  animations_enabled_ = enabled;
  if (!enabled) {
    FinishLayoutTransition();
    animate_next_layout_ = false;
    chevron_openness_.clear();
    hover_fades_.clear();
  }
}

Pixel Timeline::AnimatedRowHeight(int group_index, Pixel full_height) const {
  if (!layout_transition_.active || group_index < 0 ||
      static_cast<size_t>(group_index) >= group_heights_.size()) {
    return full_height;
  }
  return std::min(full_height, group_heights_[group_index]);
}

bool Timeline::IsGroupDrawn(int group_index) const {
  if (group_index < 0) return false;
  const size_t index = static_cast<size_t>(group_index);
  if (index < group_visible_.size() && group_visible_[index]) return true;
  return layout_transition_.active && index < layout_transition_.drawn.size() &&
         layout_transition_.drawn[index];
}

void Timeline::BeginExpandToggleMotion(int chevron_key, bool was_expanded) {
  if (!animations_enabled_) return;
  animate_next_layout_ = true;
  // If the chevron is already rotating (a quick double toggle), keep its
  // current angle so it reverses smoothly instead of jumping.
  chevron_openness_.try_emplace(chevron_key, was_expanded ? 1.0f : 0.0f);
}

float Timeline::ChevronOpenness(int chevron_key, bool expanded) {
  const float target = expanded ? 1.0f : 0.0f;
  auto it = chevron_openness_.find(chevron_key);
  if (it == chevron_openness_.end()) return target;
  it->second = motion::Approach(it->second, target, kMotionChevronRate,
                                ImGui::GetIO().DeltaTime);
  const float openness = it->second;
  if (openness == target) {
    chevron_openness_.erase(it);
  } else {
    motion_in_progress_ = true;
  }
  return openness;
}

float Timeline::HoverFade(ImGuiID id, bool hovered) {
  const float target = hovered ? 1.0f : 0.0f;
  if (!animations_enabled_) return target;
  auto it = hover_fades_.find(id);
  if (it == hover_fades_.end()) {
    if (!hovered) return 0.0f;
    it = hover_fades_.emplace(id, 0.0f).first;
  }
  it->second = motion::Approach(
      it->second, target,
      hovered ? kMotionHoverFadeInRate : kMotionHoverFadeOutRate,
      ImGui::GetIO().DeltaTime);
  const float emphasis = it->second;
  if (emphasis != target) {
    motion_in_progress_ = true;
  } else if (!hovered) {
    hover_fades_.erase(it);
  }
  return emphasis;
}

void Timeline::ApplyLayout(const FlameChartTimelineData& data,
                           std::vector<Pixel> group_offsets,
                           std::vector<Pixel> group_heights,
                           std::vector<Pixel> level_offsets,
                           const std::vector<bool>& group_visible,
                           const Pixel (&from_header_offsets)[3]) {
  // Only expand / collapse toggles animate, and only if the layout's shape is
  // unchanged (a data update in between changes it and snaps instead).
  const bool animate = animate_next_layout_ && animations_enabled_ &&
                       group_offsets_.size() == group_offsets.size() &&
                       group_heights_.size() == group_heights.size() &&
                       visible_level_offsets_.size() == level_offsets.size() &&
                       group_visible_.size() == group_visible.size();
  animate_next_layout_ = false;
  if (!animate) {
    layout_transition_ = LayoutTransition();
    group_offsets_ = std::move(group_offsets);
    group_heights_ = std::move(group_heights);
    visible_level_offsets_ = std::move(level_offsets);
    return;
  }

  LayoutTransition& transition = layout_transition_;
  // Rows visible before or after the change are drawn while it plays, so
  // collapsing rows fold away instead of vanishing. This also covers rows
  // still folding from an interrupted transition.
  std::vector<bool> was_drawn(group_visible.size());
  std::vector<bool> drawn(group_visible);
  for (size_t i = 0; i < drawn.size(); ++i) {
    was_drawn[i] = IsGroupDrawn(static_cast<int>(i));
    drawn[i] = drawn[i] || was_drawn[i];
  }
  // Clip edges of rows that are mid-slide in an interrupted transition, so the
  // new transition continues from them.
  std::vector<bool> previously_clipped = std::move(transition.clipped);
  std::vector<Pixel> previous_clip_tops = std::move(transition.clip_tops);

  // Start from what is on screen right now, which may be mid-transition, so a
  // quick second toggle retargets smoothly.
  Pixel max_travel = 0.0f;
  for (size_t i = 0; i < group_offsets.size(); ++i) {
    max_travel =
        std::max(max_travel, std::abs(group_offsets[i] - group_offsets_[i]));
  }
  for (size_t i = 0; i < group_heights.size(); ++i) {
    max_travel =
        std::max(max_travel, std::abs(group_heights[i] - group_heights_[i]));
  }
  const Pixel old_total = group_offsets_.empty() ? 0.0f : group_offsets_.back();
  const Pixel new_total = group_offsets.empty() ? 0.0f : group_offsets.back();
  transition.expanding = new_total >= old_total;
  transition.duration = std::clamp(
      kMotionLayoutMinSeconds + max_travel * kMotionLayoutSecondsPerPixel,
      kMotionLayoutMinSeconds, kMotionLayoutMaxSeconds);
  if (!transition.expanding) {
    transition.duration *= kMotionLayoutCollapseTimeScale;
  }
  transition.elapsed = 0.0f;
  transition.from_group_offsets = group_offsets_;
  transition.from_group_heights = group_heights_;
  transition.from_level_offsets = visible_level_offsets_;
  transition.to_group_offsets = std::move(group_offsets);
  transition.to_group_heights = std::move(group_heights);
  transition.to_level_offsets = std::move(level_offsets);
  transition.drawn = std::move(drawn);
  transition.clipped = std::move(previously_clipped);
  transition.clip_tops = std::move(previous_clip_tops);
  transition.final_group_offsets = transition.to_group_offsets;
  transition.final_group_heights = transition.to_group_heights;
  SetUpAccordionRuns(data, was_drawn, group_visible_, group_visible);
  // The rows on screen start where SetUpAccordionRuns placed them.
  group_offsets_ = transition.from_group_offsets;
  group_heights_ = transition.from_group_heights;
  transition.to_header_offsets[0] = header_hidden_offset_;
  transition.to_header_offsets[1] = header_pinned_offset_;
  transition.to_header_offsets[2] = header_all_offset_;
  for (int i = 0; i < 3; ++i) {
    transition.from_header_offsets[i] = from_header_offsets[i];
  }
  header_hidden_offset_ = from_header_offsets[0];
  header_pinned_offset_ = from_header_offsets[1];
  header_all_offset_ = from_header_offsets[2];
  transition.active = true;
  motion_in_progress_ = true;
}

void Timeline::SetUpAccordionRuns(const FlameChartTimelineData& data,
                                  const std::vector<bool>& was_drawn,
                                  const std::vector<bool>& was_visible,
                                  const std::vector<bool>& will_be_visible) {
  LayoutTransition& transition = layout_transition_;
  const size_t group_count = transition.to_group_offsets.empty()
                                 ? 0
                                 : transition.to_group_offsets.size() - 1;
  // Clip state carried over from an interrupted transition, if any.
  const std::vector<bool> previously_clipped = transition.clipped;
  const std::vector<Pixel> previous_clip_tops = transition.clip_tops;
  auto was_clipped = [&](size_t i) {
    return i < previously_clipped.size() && previously_clipped[i] &&
           i < previous_clip_tops.size();
  };
  transition.clipped.assign(group_count, false);
  transition.from_clip_tops.assign(group_count, 0.0f);
  transition.to_clip_tops.assign(group_count, 0.0f);
  transition.clip_tops.assign(group_count, 0.0f);
  if (group_count == 0 || was_drawn.size() < group_count ||
      was_visible.size() < group_count ||
      will_be_visible.size() < group_count) {
    return;
  }

  std::vector<Pixel>& from_offsets = transition.from_group_offsets;
  std::vector<Pixel>& to_offsets = transition.to_group_offsets;
  std::vector<Pixel>& from_heights = transition.from_group_heights;
  std::vector<Pixel>& to_heights = transition.to_group_heights;

  enum class RunKind { kNone, kLeaving, kEntering };
  std::vector<int> run;
  RunKind run_kind = RunKind::kNone;

  // Lays out one maximal run of consecutive rows that all leave (or all enter)
  // the screen. Leaving rows keep their height and translate up so that the
  // run's bottom ends exactly where the rows below it end up; entering rows
  // start tucked above their final place the same way. A clip edge that
  // moves with the boundary of the run hides whatever is above it, so rows
  // appear to slide out from (or back under) the row above.
  auto flush_run = [&]() {
    if (run.empty()) return;
    const int first = run.front();
    if (run_kind == RunKind::kLeaving) {
      const Pixel anchor = to_offsets[first];
      Pixel block_end = from_offsets[first];
      for (int i : run) {
        block_end = std::max(block_end, from_offsets[i] + from_heights[i]);
      }
      const Pixel run_top =
          was_clipped(first) ? previous_clip_tops[first] : from_offsets[first];
      for (int i : run) {
        to_offsets[i] = anchor - (block_end - from_offsets[i]);
        to_heights[i] = from_heights[i];
        transition.clipped[i] = true;
        transition.from_clip_tops[i] =
            was_clipped(i) ? previous_clip_tops[i] : run_top;
        transition.to_clip_tops[i] = anchor;
      }
    } else if (run_kind == RunKind::kEntering) {
      const Pixel anchor = from_offsets[first];
      Pixel block_end = to_offsets[first];
      for (int i : run) {
        block_end = std::max(block_end, to_offsets[i] + to_heights[i]);
      }
      for (int i : run) {
        if (!was_drawn[i]) {
          // Truly hidden rows start tucked above the anchor; rows that are
          // still on screen from an interrupted transition continue from
          // where they are.
          from_offsets[i] = anchor - (block_end - to_offsets[i]);
          from_heights[i] = to_heights[i];
        }
        transition.clipped[i] = true;
        transition.from_clip_tops[i] =
            was_clipped(i) ? previous_clip_tops[i] : anchor;
        transition.to_clip_tops[i] = to_offsets[first];
      }
    }
    run.clear();
    run_kind = RunKind::kNone;
  };

  for (const Group* group_ptr : flattened_groups_) {
    if (IsVirtualHeader(group_ptr)) {
      flush_run();
      continue;
    }
    const size_t i = static_cast<size_t>(group_ptr - data.groups.data());
    if (i >= group_count) continue;
    const bool leaving = was_drawn[i] && !will_be_visible[i];
    const bool entering = will_be_visible[i] && !was_visible[i];
    if (!leaving && !entering) {
      // Rows hidden both before and after do not interrupt a run; any other
      // row ends it.
      if (was_drawn[i] || will_be_visible[i]) flush_run();
      continue;
    }
    const RunKind kind = leaving ? RunKind::kLeaving : RunKind::kEntering;
    if (kind != run_kind) flush_run();
    run_kind = kind;
    run.push_back(static_cast<int>(i));
  }
  flush_run();
  transition.clip_tops = transition.from_clip_tops;
}

std::optional<Pixel> Timeline::AccordionClipTop(int group_index) const {
  const LayoutTransition& transition = layout_transition_;
  if (!transition.active || group_index < 0) return std::nullopt;
  const size_t i = static_cast<size_t>(group_index);
  if (i >= transition.clipped.size() || !transition.clipped[i] ||
      i >= transition.clip_tops.size()) {
    return std::nullopt;
  }
  return tracks_start_screen_pos_.y + transition.clip_tops[i];
}

void Timeline::AdvanceLayoutTransition(float delta_time) {
  LayoutTransition& transition = layout_transition_;
  if (!transition.active) return;
  transition.elapsed += std::max(0.0f, delta_time);
  const float progress = transition.duration > 0.0f
                             ? transition.elapsed / transition.duration
                             : 1.0f;
  if (progress >= 1.0f ||
      group_offsets_.size() != transition.to_group_offsets.size()) {
    FinishLayoutTransition();
    return;
  }
  // Expanding uses an emphasized decelerate curve (content rushes in and
  // settles); collapsing uses the standard curve.
  const float eased = transition.expanding
                          ? motion::EmphasizedDecelerate(progress)
                          : motion::Standard(progress);
  auto lerp = [eased](const std::vector<Pixel>& from,
                      const std::vector<Pixel>& to, std::vector<Pixel>& out) {
    for (size_t i = 0; i < out.size() && i < from.size() && i < to.size();
         ++i) {
      out[i] = from[i] + (to[i] - from[i]) * eased;
    }
  };
  lerp(transition.from_group_offsets, transition.to_group_offsets,
       group_offsets_);
  lerp(transition.from_group_heights, transition.to_group_heights,
       group_heights_);
  lerp(transition.from_level_offsets, transition.to_level_offsets,
       visible_level_offsets_);
  lerp(transition.from_clip_tops, transition.to_clip_tops,
       transition.clip_tops);
  Pixel* const header_offsets[3] = {
      &header_hidden_offset_, &header_pinned_offset_, &header_all_offset_};
  for (int i = 0; i < 3; ++i) {
    *header_offsets[i] =
        transition.from_header_offsets[i] +
        (transition.to_header_offsets[i] - transition.from_header_offsets[i]) *
            eased;
  }
  motion_in_progress_ = true;
}

void Timeline::FinishLayoutTransition() {
  LayoutTransition& transition = layout_transition_;
  if (!transition.active) return;
  if (transition.final_group_offsets.size() == group_offsets_.size()) {
    group_offsets_ = std::move(transition.final_group_offsets);
    group_heights_ = std::move(transition.final_group_heights);
    visible_level_offsets_ = std::move(transition.to_level_offsets);
    header_hidden_offset_ = transition.to_header_offsets[0];
    header_pinned_offset_ = transition.to_header_offsets[1];
    header_all_offset_ = transition.to_header_offsets[2];
  }
  // Releases the transient copies.
  transition = LayoutTransition();
}

void Timeline::SetVisibleRange(const TimeRange& range, bool animate) {
  if (animate) {
    visible_range_ = range;
  } else {
    visible_range_.snap_to(range);
  }
  SyncMinimapState();
  if (redraw_callback_) redraw_callback_();
}

void Timeline::BackfillGroupLevelCount(FlameChartTimelineData& data) {
  // Backfill level_count for tests that only set start_level.
  for (size_t i = 0; i < data.groups.size(); ++i) {
    if (data.groups[i].level_count <= 0) {
      int next_level = (i + 1 < data.groups.size())
                           ? data.groups[i + 1].start_level
                           : data.total_levels();
      data.groups[i].level_count =
          std::max(1, next_level - data.groups[i].start_level);
    }
  }
}

void Timeline::CaptureSelectionStateIfNeeded() {
  if (saved_selection_state_.has_value()) {
    return;
  }
  if (selected_event_index_ >= 0 &&
      (static_cast<size_t>(selected_event_index_) <
           timeline_data_.entry_start_times.size() ||
       static_cast<size_t>(selected_event_index_) <
           timeline_data_.entry_names.size() ||
       static_cast<size_t>(selected_event_index_) <
           timeline_data_.entry_name_indices.size())) {
    SavedSelectionState state;
    state.event_id = GetEventIdFromData(timeline_data_, selected_event_index_);
    if (static_cast<size_t>(selected_event_index_) <
        timeline_data_.entry_levels.size()) {
      if (const Group* grp = timeline_data_.FindGroupForLevel(
              timeline_data_.entry_levels[selected_event_index_])) {
        state.pid = grp->pid;
        state.tid = grp->tid;
        state.group_name = grp->name;
      }
    }
    state.name = std::string(GetEventName(selected_event_index_));
    state.start = (static_cast<size_t>(selected_event_index_) <
                   timeline_data_.entry_start_times.size())
                      ? timeline_data_.entry_start_times[selected_event_index_]
                      : 0.0;
    state.dur = (static_cast<size_t>(selected_event_index_) <
                 timeline_data_.entry_total_times.size())
                    ? timeline_data_.entry_total_times[selected_event_index_]
                    : 0.0;
    saved_selection_state_ = std::move(state);
  }
}

void Timeline::ReleaseTimelineEntriesPreservingState() {
  CaptureSelectionStateIfNeeded();
  const size_t entry_count = GetEntryCount();
  const bool overview_is_compacted =
      is_compacted_ ||
      absl::c_any_of(level_mip_pyramids_, [](const LevelMipPyramid& p) {
        return p.raw_events_compacted;
      });
  const size_t uncompacted_stash_limit = std::min(
      max_uncompacted_overview_entries_, kMaxUncompactedOverviewStashEntries);
  if (stash_full_overview_on_next_update_ && incremental_fetch_enabled_ &&
      (overview_is_compacted || entry_count <= uncompacted_stash_limit) &&
      !saved_full_overview_.has_value() && entry_count > 0) {
    // Keep the original `timeline_data_.groups` vector buffer in place on
    // `timeline_data_` so `flattened_groups_` Group* pointers remain valid
    // during scroll-anchor capture in `UpdateTimelineData()`.
    std::vector<Group> original_groups;
    original_groups.swap(timeline_data_.groups);
    saved_full_overview_ = SavedFullOverview{
        .data = std::move(timeline_data_),
        .entry_colors = std::move(entry_colors_),
        .cached_trace_colors_version = cached_trace_colors_version_,
        .level_mip_pyramids = std::move(level_mip_pyramids_),
        .process_utilization_pyramids =
            std::move(process_utilization_pyramids_),
        .gpu_flame_instances = std::move(gpu_flame_instances_),
        .process_utilization_pyramids_dirty =
            process_utilization_pyramids_dirty_,
        .fetched_data_time_range = fetched_data_time_range_,
    };
    saved_full_overview_->data.groups = original_groups;
    timeline_data_ = FlameChartTimelineData{};
    timeline_data_.groups.swap(original_groups);
  } else {
    ReleaseEntryMemory(timeline_data_);
    std::vector<ImU32>().swap(entry_colors_);
    std::vector<LevelMipPyramid>().swap(level_mip_pyramids_);
    std::vector<ProcessUtilizationPyramid>().swap(
        process_utilization_pyramids_);
  }
  process_utilization_pyramids_dirty_ = true;
  stash_full_overview_on_next_update_ = false;
  is_full_overview_data_ = false;
  matching_event_indices_.release();
  selected_event_indices_.release();
  filtered_event_indices_.release();
}

void Timeline::SetTimelineData(FlameChartTimelineData data) {
  UpdateTimelineData([&]() { return std::move(data); });
}

void Timeline::UpdateTimelineData(
    absl::FunctionRef<FlameChartTimelineData()> builder) {
  // Capture anchor track and local pixel offset prior to updating layout.
  GroupKey anchor_group_key = {};
  Pixel local_pixel_offset = 0.0f;
  bool has_anchor = false;

  if (is_incremental_loading_ && !flattened_groups_.empty()) {
    const Group* first_concrete_track = nullptr;
    for (const Group* group : flattened_groups_) {
      if (!IsVirtualHeader(group)) {
        first_concrete_track = group;
        break;
      }
    }

    // NOMUTANTS(LCR): If first_concrete_track is null, all flattened groups are
    // virtual headers so has_anchor is never set, making short-circuiting a
    // semantic no-op.
    if (first_concrete_track != nullptr &&
        last_scroll_y_ >= GetGroupTop(first_concrete_track)) {
      auto it =
          std::lower_bound(flattened_groups_.begin(), flattened_groups_.end(),
                           last_scroll_y_, [this](const Group* group, Pixel y) {
                             return this->GetGroupBottom(group) < y;
                           });

      // Advance past virtual headers and invisible groups to anchor on a
      // concrete visible track.
      while (it != flattened_groups_.end()) {
        if (IsVirtualHeader(*it)) {
          ++it;
          continue;
        }
        const int group_idx = *it - timeline_data_.groups.data();
        if (group_idx < 0 ||
            group_idx >= static_cast<int>(timeline_data_.groups.size())) {
          ++it;
          continue;
        }
        if (group_idx < static_cast<int>(group_visible_.size()) &&
            !group_visible_[group_idx]) {
          ++it;
          continue;
        }
        break;
      }

      if (it != flattened_groups_.end()) {
        const Group* const anchor_group = *it;
        anchor_group_key.nesting_level = anchor_group->nesting_level;
        anchor_group_key.name = anchor_group->name;
        anchor_group_key.parent_name =
            (anchor_group->parent_index >= 0 &&
             anchor_group->parent_index <
                 static_cast<int>(timeline_data_.groups.size()))
                ? timeline_data_.groups[anchor_group->parent_index].name
                : "";
        local_pixel_offset =
            std::max(0.0f, last_scroll_y_ - GetGroupTop(anchor_group));
        has_anchor = true;
      }
    }
  }

  // Capture selection 5-tuple prior to releasing/replacing timeline_data_.
  CaptureSelectionStateIfNeeded();
  const std::optional<SavedSelectionState> saved_selection =
      std::move(saved_selection_state_);
  saved_selection_state_.reset();
  const bool had_selected_event = saved_selection.has_value();
  const EventId selected_event_id =
      had_selected_event ? saved_selection->event_id : 0;
  const ProcessId selected_pid = had_selected_event ? saved_selection->pid : 0;
  const ThreadId selected_tid = had_selected_event ? saved_selection->tid : 0;
  const std::string& selected_group_name =
      had_selected_event ? saved_selection->group_name : "";
  const std::string& selected_name =
      had_selected_event ? saved_selection->name : "";
  const Microseconds selected_start =
      had_selected_event ? saved_selection->start : 0.0;
  const Microseconds selected_dur =
      had_selected_event ? saved_selection->dur : 0.0;

  const bool recompute_local_search =
      search_results_.empty() && !search_query_lower_.empty();
  const bool had_selected_event_dependencies =
      had_selected_event && selected_event_index_ >= 0 &&
      dependencies_.event_index == selected_event_index_;

  // Release old entry columns, sparse maps, CSR, flows, entry_colors_, and
  // level_mip_pyramids_ before allocating new layout, colors, and pyramids.
  ReleaseTimelineEntriesPreservingState();
  std::vector<Group>().swap(timeline_data_.groups);

  FlameChartTimelineData data = builder();
  BackfillGroupLevelCount(data);

  // Pre-calculate the level positions to avoid partial state and per-frame
  // layout recalculations before saving the newly arrived timeline_data.
  UpdateLevelPositions(data);
  timeline_data_ = std::move(data);
  // Resolved event indices do not apply to the new data. The names are
  // resolved again below if the same op stays selected.
  EventDependencies old_dependencies = std::exchange(dependencies_, {});
  RebuildEntryColors();
  if (active_event_filter_.has_value()) {
    filtered_event_indices_ = EvaluateEventFilter(*active_event_filter_);
    has_active_event_filter_ = true;
  } else {
    has_active_event_filter_ = false;
  }
  ++filter_epoch_;
  if (is_compacted_ &&
      (!incremental_fetch_enabled_ ||
       (timeline_data_.entry_start_times.size() <=
            max_uncompacted_overview_entries_ &&
        EstimateRetainedEntryBytes() <= max_uncompacted_overview_bytes_))) {
    is_compacted_ = false;
    window_at_full_resolution_ = true;
  }
  BuildLevelMipPyramids();

  // Execute 4-tier despawn fallback to restore compensated scroll offset.
  if (has_anchor) {
    int new_group_index = -1;

    // Tier 1: Exact GroupKey match.
    for (size_t i = 0; i < timeline_data_.groups.size(); ++i) {
      const Group& g = timeline_data_.groups[i];
      const absl::string_view parent =
          (g.parent_index >= 0 &&
           g.parent_index < static_cast<int>(timeline_data_.groups.size()))
              ? absl::string_view(timeline_data_.groups[g.parent_index].name)
              : "";
      if (g.nesting_level == anchor_group_key.nesting_level &&
          g.name == anchor_group_key.name &&
          parent == anchor_group_key.parent_name) {
        new_group_index = static_cast<int>(i);
        break;
      }
    }

    // Tier 2: Match Parent Track if child track despawned.
    if (new_group_index == -1 && !anchor_group_key.parent_name.empty()) {
      for (size_t i = 0; i < timeline_data_.groups.size(); ++i) {
        if (timeline_data_.groups[i].nesting_level <
                anchor_group_key.nesting_level &&
            timeline_data_.groups[i].name == anchor_group_key.parent_name) {
          new_group_index = static_cast<int>(i);
          local_pixel_offset = 0.0f;
          break;
        }
      }
    }

    // Tier 3: Match Nearest Preceding Surviving Track.
    if (new_group_index == -1) {
      for (int i = static_cast<int>(timeline_data_.groups.size()) - 1; i >= 0;
           --i) {
        if (i < static_cast<int>(group_visible_.size()) && !group_visible_[i]) {
          continue;
        }
        if (GetGroupTop(&timeline_data_.groups[i]) <= last_scroll_y_) {
          new_group_index = i;
          local_pixel_offset = 0.0f;
          break;
        }
      }
    }

    // Tier 4: Apply Compensated Scroll Offset or Clamp to Bounds.
    if (timeline_data_.groups.empty()) {
      last_scroll_y_ = 0.0f;
    } else if (new_group_index != -1) {
      const Pixel track_height = group_heights_[new_group_index];
      local_pixel_offset = std::min(
          local_pixel_offset, std::max(0.0f, track_height - kEventHeight));
      last_scroll_y_ = GetGroupTop(&timeline_data_.groups[new_group_index]) +
                       local_pixel_offset;
    } else {
      last_scroll_y_ = std::max(0.0f, last_scroll_y_);
    }
  } else if (timeline_data_.groups.empty()) {
    last_scroll_y_ = 0.0f;
  } else {
    last_scroll_y_ = std::max(0.0f, last_scroll_y_);
  }

  // Reconcile loaded_index for all search results against the new
  // timeline_data_.
  const int num_loaded =
      static_cast<int>(timeline_data_.entry_event_ids.size());
  absl::flat_hash_map<EventId, int> event_id_to_loaded_index;
  if (!search_results_.empty() ||
      (had_selected_event && selected_event_id != 0)) {
    event_id_to_loaded_index.reserve(num_loaded);
    for (int i = 0; i < num_loaded; ++i) {
      event_id_to_loaded_index.try_emplace(timeline_data_.entry_event_ids[i],
                                           i);
    }
  }

  std::optional<absl::flat_hash_map<FallbackKey, std::vector<int>>>
      lazy_loaded_events;
  auto ensure_lazy_loaded_events = [&]() {
    if (lazy_loaded_events.has_value()) {
      return;
    }
    absl::flat_hash_set<FallbackKey> wanted_keys;
    wanted_keys.reserve(search_results_.size() + (had_selected_event ? 1 : 0));
    for (const SearchResult& res : search_results_) {
      if (!event_id_to_loaded_index.contains(res.event_id)) {
        wanted_keys.emplace(res.pid, res.tid, res.name);
      }
    }
    if (had_selected_event) {
      wanted_keys.emplace(selected_pid, selected_tid, selected_name);
    }
    lazy_loaded_events = BuildFallbackMap(timeline_data_, wanted_keys);
  };

  matching_event_indices_.clear();
  if (recompute_local_search) {
    RecomputeSearchResults();
  } else {
    for (SearchResult& result : search_results_) {
      if (const auto it = event_id_to_loaded_index.find(result.event_id);
          it != event_id_to_loaded_index.end()) {
        result.loaded_index = it->second;
        if (static_cast<size_t>(result.loaded_index) <
            timeline_data_.entry_levels.size()) {
          result.level = timeline_data_.entry_levels[result.loaded_index];
        }
      } else {
        ensure_lazy_loaded_events();
        result.loaded_index = FindFallbackMatchedIndex(
            *lazy_loaded_events, timeline_data_, result.pid, result.tid,
            result.name, result.start_time, result.duration);
        if (result.loaded_index != -1) {
          if (static_cast<size_t>(result.loaded_index) <
              timeline_data_.entry_levels.size()) {
            result.level = timeline_data_.entry_levels[result.loaded_index];
          }
          result.event_id =
              GetEventIdFromData(timeline_data_, result.loaded_index);
        }
      }
      if (result.loaded_index != -1 &&
          IsEventVisibleUnderFilter(result.loaded_index)) {
        matching_event_indices_.insert(result.loaded_index);
      }
    }
  }

  const int total_search_results = get_search_results_count();
  if (pending_navigation_event_id_.has_value()) {
    int matched_index = -1;
    if (current_search_result_index_ >= 0 &&
        current_search_result_index_ < total_search_results) {
      matched_index =
          !search_results_.empty()
              ? search_results_.at(current_search_result_index_).loaded_index
              : GetNthMatchingCsrEventIndex(current_search_result_index_);
    }

    if (matched_index != -1) {
      ZoomEvent(matched_index);
      pending_navigation_event_id_.reset();
    }
  } else if (current_search_result_index_ >= 0 &&
             current_search_result_index_ < total_search_results) {
    selected_event_index_ =
        !search_results_.empty()
            ? search_results_.at(current_search_result_index_).loaded_index
            : GetNthMatchingCsrEventIndex(current_search_result_index_);
  } else if (current_search_result_index_ >= 0) {
    selected_event_index_ = -1;
  } else if (had_selected_event) {
    // Decouple selection re-indexing without mutating viewport scroll.
    int new_selected_index = -1;
    if (selected_event_id != 0) {
      if (const auto it = event_id_to_loaded_index.find(selected_event_id);
          it != event_id_to_loaded_index.end()) {
        new_selected_index = it->second;
      }
    }
    if (new_selected_index != -1 && static_cast<size_t>(new_selected_index) <
                                        timeline_data_.entry_levels.size()) {
      if (const Group* grp = timeline_data_.FindGroupForLevel(
              timeline_data_.entry_levels[new_selected_index])) {
        if (grp->pid != selected_pid || grp->tid != selected_tid ||
            (!selected_group_name.empty() &&
             grp->name != selected_group_name)) {
          new_selected_index = -1;
        }
      }
    }
    if (new_selected_index == -1) {
      ensure_lazy_loaded_events();
      new_selected_index = FindFallbackMatchedIndex(
          *lazy_loaded_events, timeline_data_, selected_pid, selected_tid,
          selected_name, selected_start, selected_dur, selected_group_name);
    }
    selected_event_index_ = new_selected_index;
  } else if (current_search_result_index_ < 0) {
    selected_event_index_ = -1;
  }

  CompactTimelineDataToMipOverview();
  UploadGpuFlameInstances();

  // Resolves the dependencies again if the same op is still selected, unless a
  // selection handler already set them for the new selection during
  // `ZoomEvent`. This runs after the compaction above, so the resolved event
  // indices refer to the events that are kept.
  if (had_selected_event_dependencies && dependencies_.event_index == -1 &&
      selected_event_index_ >= 0 &&
      selected_event_index_ < static_cast<int>(GetEntryCount()) &&
      GetEventName(selected_event_index_) == selected_name) {
    SetSelectedEventDependencies(std::move(old_dependencies.operand_names),
                                 std::move(old_dependencies.consumer_names));
  }

  if (is_incremental_loading_) {
    should_restore_scroll_ = true;
  }

  minimap_schedule_dirty_ = true;
  SyncMinimapState();

  if (redraw_callback_) redraw_callback_();
}

void Timeline::RebuildEntryColors() {
  const auto& trace_colors = palette_.GetTraceColors();
  cached_trace_colors_version_ = palette_.GetTraceVersion();
  if (!timeline_data_.interned_string_pool.empty()) {
    const size_t n = timeline_data_.interned_string_pool.size();
    entry_colors_.resize(n);
    for (size_t i = 0; i < n; ++i) {
      entry_colors_[i] =
          GetColorForId(timeline_data_.interned_string_pool[i], trace_colors);
    }
    RebuildGroupAccentColors();
    return;
  }
  const size_t n = timeline_data_.entry_names.size();
  entry_colors_.resize(n);
  absl::string_view last_name;
  ImU32 last_color = 0;
  bool has_last = false;
  for (size_t i = 0; i < n; ++i) {
    const std::string& name = timeline_data_.entry_names[i];
    if (has_last && name == last_name) {
      entry_colors_[i] = last_color;
    } else {
      last_name = name;
      last_color = GetColorForId(name, trace_colors);
      has_last = true;
      entry_colors_[i] = last_color;
    }
  }
  RebuildGroupAccentColors();
}

void Timeline::RebuildGroupAccentColors() {
  // Bounds the work per track on huge traces: the dominant color is a visual
  // hint, so a uniform sample of a track's events is accurate enough.
  constexpr size_t kMaxSampledEventsPerGroup = 8192;

  const size_t group_count = timeline_data_.groups.size();
  group_accent_colors_.assign(group_count, 0);
  const auto& total_times = timeline_data_.entry_total_times;
  absl::InlinedVector<std::pair<ImU32, double>, kMaxColors> color_weights;
  for (size_t group_index = 0; group_index < group_count; ++group_index) {
    const Group& group = timeline_data_.groups[group_index];
    if (group.type != Group::Type::kFlame ||
        group.nesting_level == kProcessNestingLevel) {
      continue;
    }
    color_weights.clear();
    const int start_level = std::max(0, group.start_level);
    const int end_level =
        std::min(GetNextGroupStartLevel(timeline_data_, group_index),
                 timeline_data_.total_levels());
    size_t group_event_count = 0;
    for (int level = start_level; level < end_level; ++level) {
      group_event_count += timeline_data_.level_events(level).size();
    }
    // A single stride across all levels keeps the sample uniform, so levels
    // keep their relative weights.
    const size_t stride =
        std::max<size_t>(1, group_event_count / kMaxSampledEventsPerGroup);
    for (int level = start_level; level < end_level; ++level) {
      const absl::Span<const int> indices = timeline_data_.level_events(level);
      for (size_t i = 0; i < indices.size(); i += stride) {
        const int event_index = indices[i];
        if (event_index < 0 ||
            static_cast<size_t>(event_index) >= total_times.size()) {
          continue;
        }
        const ImU32 color = GetEventColor(event_index);
        // Weight by duration so the accent matches the color that covers the
        // most area on the canvas.
        const double weight = std::max(total_times[event_index], 1e-3);
        auto it = absl::c_find_if(color_weights, [color](const auto& entry) {
          return entry.first == color;
        });
        if (it != color_weights.end()) {
          it->second += weight;
        } else if (color_weights.size() < kMaxColors) {
          // Trace colors come from a palette of at most `kMaxColors` colors,
          // so this bound only guards the linear search above.
          color_weights.push_back({color, weight});
        }
      }
    }
    if (!color_weights.empty()) {
      group_accent_colors_[group_index] =
          absl::c_max_element(color_weights, [](const auto& a, const auto& b) {
            return a.second < b.second;
          })->first;
    }
  }
}

void Timeline::BuildLevelMipPyramids() {
  const int num_levels = timeline_data_.total_levels();
  level_mip_pyramids_.clear();
  level_mip_pyramids_.resize(num_levels);
  process_utilization_pyramids_.clear();
  process_utilization_pyramids_dirty_ = true;
  if (num_levels == 0 || timeline_data_.entry_start_times.empty()) {
    return;
  }
  for (int lvl = 0; lvl < num_levels; ++lvl) {
    BuildLevelMipPyramidForLevel(lvl);
  }
}

void Timeline::EnsureLevelMipPyramid(int lvl) {
  if (lvl < 0 || static_cast<size_t>(lvl) >= level_mip_pyramids_.size()) {
    return;
  }
  // Compacted levels keep only MIP representatives; keep the pre-compaction
  // pyramid. TODO(b/568464847): fetch full resolution instead.
  if (level_mip_pyramids_[lvl].raw_events_compacted) {
    return;
  }
  if (level_mip_pyramids_[lvl].built_filter_epoch != filter_epoch_) {
    BuildLevelMipPyramidForLevel(lvl);
  }
}

void Timeline::BuildLevelMipPyramidForLevel(int lvl) {
  if (lvl < 0 || static_cast<size_t>(lvl) >= level_mip_pyramids_.size()) {
    return;
  }
  LevelMipPyramid& pyramid = level_mip_pyramids_[lvl];
  pyramid = LevelMipPyramid{};
  pyramid.built_filter_epoch = filter_epoch_;
  if (timeline_data_.entry_start_times.empty()) {
    return;
  }

  const auto& start_times = timeline_data_.entry_start_times;
  const auto& total_times = timeline_data_.entry_total_times;

  const absl::Span<const int> indices = timeline_data_.level_events(lvl);
  size_t visible_event_count = 0;
  int first_idx = -1;
  int last_idx = -1;
  if (!has_active_event_filter_) {
    visible_event_count = indices.size();
    if (!indices.empty()) {
      first_idx = indices.front();
      last_idx = indices.back();
    }
  } else if (!filtered_event_indices_.empty()) {
    for (const int idx : indices) {
      if (filtered_event_indices_.contains(idx)) {
        if (first_idx == -1) {
          first_idx = idx;
        }
        last_idx = idx;
        ++visible_event_count;
      }
    }
  }
  if (visible_event_count < kMinEventsForMipPyramid || first_idx < 0 ||
      static_cast<size_t>(first_idx) >= start_times.size() || last_idx < 0 ||
      static_cast<size_t>(last_idx) >= start_times.size()) {
    return;
  }
  const Microseconds level_min_ts = start_times[first_idx];
  const Microseconds level_max_ts =
      std::max(start_times[first_idx] + total_times[first_idx],
               start_times[last_idx] + total_times[last_idx]);
  const Microseconds span = level_max_ts - level_min_ts;
  if (span <= 0.0) {
    return;
  }

  struct MipBuilderState {
    bool active = false;
    int num_bins = 0;
    double inv_bin_width = 0.0;
    int current_bin = -1;
    Microseconds dom_dur = -1.0;
    double occ_dur = 0.0;
    double weight_sum = 0.0;
    double r_sum = 0.0;
    double g_sum = 0.0;
    double b_sum = 0.0;
  };

  MipBuilderState states[LevelMipPyramid::kNumMipLevels];
  int first_active_m = -1;
  for (int m = 0; m < LevelMipPyramid::kNumMipLevels; ++m) {
    int num_bins = kMipBinCounts[m];
    if (is_compacted_ && m == LevelMipPyramid::kNumMipLevels - 1 &&
        static_cast<size_t>(num_bins) >= visible_event_count) {
      num_bins = std::max<int>(32, static_cast<int>(visible_event_count / 2));
    }
    if (static_cast<size_t>(num_bins) >= visible_event_count) {
      continue;
    }
    const Microseconds bin_width = span / static_cast<double>(num_bins);
    if (bin_width <= 0.0) continue;
    states[m].active = true;
    states[m].num_bins = num_bins;
    states[m].inv_bin_width = 1.0 / bin_width;
    pyramid.levels[m].bin_width_us = bin_width;
    pyramid.levels[m].tiles.reserve(num_bins);
    if (first_active_m == -1) {
      first_active_m = m;
    }
  }
  if (first_active_m == -1) {
    return;
  }
  pyramid.has_pyramid = true;
  pyramid.level_min_ts = level_min_ts;
  pyramid.level_max_ts = level_max_ts;
  pyramid.visible_event_count = visible_event_count;

  auto finalize_single_tile = [&](int m) {
    LevelMipLevel& mip = pyramid.levels[m];
    if (mip.tiles.empty()) return;
    LevelMipTile& tile = mip.tiles.back();
    const MipBuilderState& st = states[m];
    tile.total_occupied_dur = static_cast<float>(st.occ_dur);
    if (st.weight_sum > 0.0) {
      const uint32_t r = static_cast<uint32_t>(
          std::clamp(st.r_sum / st.weight_sum, 0.0, 255.0));
      const uint32_t g = static_cast<uint32_t>(
          std::clamp(st.g_sum / st.weight_sum, 0.0, 255.0));
      const uint32_t b = static_cast<uint32_t>(
          std::clamp(st.b_sum / st.weight_sum, 0.0, 255.0));
      tile.blended_color = IM_COL32(r, g, b, 255);
    } else if (tile.dominant_event_index >= 0) {
      tile.blended_color = GetEventColor(tile.dominant_event_index);
    }
  };

  size_t actual_visible_count = 0;
  for (int idx : indices) {
    if (idx < 0 || static_cast<size_t>(idx) >= start_times.size()) {
      continue;
    }
    if (has_active_event_filter_ && !filtered_event_indices_.contains(idx)) {
      continue;
    }
    ++actual_visible_count;
    const Microseconds s = start_times[idx];
    const Microseconds d = total_times[idx];
    const Microseconds e = s + d;
    pyramid.level_min_ts = std::min(pyramid.level_min_ts, s);
    pyramid.level_max_ts = std::max(pyramid.level_max_ts, e);
    const ImU32 c = GetEventColor(idx);
    const double w = std::max(d, 1e-3);
    const double cr = static_cast<double>((c >> IM_COL32_R_SHIFT) & 0xFF) * w;
    const double cg = static_cast<double>((c >> IM_COL32_G_SHIFT) & 0xFF) * w;
    const double cb = static_cast<double>((c >> IM_COL32_B_SHIFT) & 0xFF) * w;
    const Microseconds rel_s = s - level_min_ts;

    for (int m = first_active_m; m < LevelMipPyramid::kNumMipLevels; ++m) {
      MipBuilderState& st = states[m];
      if (!st.active) continue;
      LevelMipLevel& mip = pyramid.levels[m];
      const int bin = std::clamp(static_cast<int>(rel_s * st.inv_bin_width), 0,
                                 st.num_bins - 1);
      if (bin != st.current_bin) {
        finalize_single_tile(m);
        st.current_bin = bin;
        st.dom_dur = d;
        st.occ_dur = std::max(0.0, d);
        st.weight_sum = w;
        st.r_sum = cr;
        st.g_sum = cg;
        st.b_sum = cb;
        mip.tiles.push_back(LevelMipTile{
            .min_start = s,
            .max_end = e,
            .dominant_event_index = idx,
            .total_occupied_dur = static_cast<float>(st.occ_dur),
            .count = 1,
            .blended_color = c,
        });
      } else {
        LevelMipTile& tile = mip.tiles.back();
        const Microseconds prev_end = tile.max_end;
        tile.max_end = std::max(tile.max_end, e);
        if (e > prev_end) {
          st.occ_dur += (e - std::max(s, prev_end));
        }
        ++tile.count;
        st.weight_sum += w;
        st.r_sum += cr;
        st.g_sum += cg;
        st.b_sum += cb;
        if (d > st.dom_dur) {
          st.dom_dur = d;
          tile.dominant_event_index = idx;
        }
      }
    }
  }
  pyramid.visible_event_count = actual_visible_count;
  for (int m = first_active_m; m < LevelMipPyramid::kNumMipLevels; ++m) {
    if (states[m].active) {
      finalize_single_tile(m);
      pyramid.levels[m].tiles.shrink_to_fit();
    }
  }
}

void Timeline::UploadGpuFlameInstances() {
  gpu_flame_instances_.clear();
  const int num_levels = timeline_data_.total_levels();
  if (num_levels == 0 ||
      level_mip_pyramids_.size() != static_cast<size_t>(num_levels)) {
    return;
  }

  const auto& start_times = timeline_data_.entry_start_times;
  const auto& total_times = timeline_data_.entry_total_times;
  const bool include_raw_events =
      !start_times.empty() && start_times.size() <= 400000;

  size_t total_instances = 0;
  for (int lvl = 0; lvl < num_levels; ++lvl) {
    const auto& pyr = level_mip_pyramids_[lvl];
    for (int m = 0; m < LevelMipPyramid::kNumMipLevels; ++m) {
      total_instances += pyr.levels[m].tiles.size();
    }
    if (include_raw_events) {
      total_instances += timeline_data_.level_events(lvl).size();
    }
  }
  if (total_instances == 0) {
    return;
  }

  gpu_flame_instances_.reserve(total_instances);
  for (int lvl = 0; lvl < num_levels; ++lvl) {
    LevelMipPyramid& pyr = level_mip_pyramids_[lvl];
    for (int m = 0; m < LevelMipPyramid::kNumMipLevels; ++m) {
      LevelMipLevel& mip = pyr.levels[m];
      mip.gpu_instance_offset =
          static_cast<uint32_t>(gpu_flame_instances_.size());
      for (const auto& tile : mip.tiles) {
        const float start_hi = static_cast<float>(tile.min_start);
        const float start_lo =
            static_cast<float>(tile.min_start - static_cast<double>(start_hi));
        const float dur =
            static_cast<float>(std::max(0.0, tile.max_end - tile.min_start));
        const uint32_t col =
            (tile.blended_color != 0)
                ? tile.blended_color
                : (tile.dominant_event_index >= 0
                       ? GetEventColor(tile.dominant_event_index)
                       : 0u);
        gpu_flame_instances_.push_back(CachedGpuFlameInstance{
            .start_hi = start_hi,
            .start_lo = start_lo,
            .duration = dur,
            .color = col,
        });
      }
    }
    if (include_raw_events) {
      const absl::Span<const int> indices = timeline_data_.level_events(lvl);
      pyr.raw_gpu_instance_offset =
          static_cast<uint32_t>(gpu_flame_instances_.size());
      for (int idx : indices) {
        if (idx >= 0 && static_cast<size_t>(idx) < start_times.size()) {
          const double s = start_times[idx];
          const float start_hi = static_cast<float>(s);
          const float start_lo =
              static_cast<float>(s - static_cast<double>(start_hi));
          const double raw_dur = total_times[idx];
          const float dur = static_cast<float>(std::max(0.0, raw_dur));
          const uint32_t col = (raw_dur <= 1e-06) ? 0u : GetEventColor(idx);
          gpu_flame_instances_.push_back(CachedGpuFlameInstance{
              .start_hi = start_hi,
              .start_lo = start_lo,
              .duration = dur,
              .color = col,
          });
        } else {
          gpu_flame_instances_.push_back(CachedGpuFlameInstance{});
        }
      }
    } else {
      pyr.raw_gpu_instance_offset = UINT32_MAX;
    }
  }

#ifdef __EMSCRIPTEN__
  static_assert(sizeof(CachedGpuFlameInstance) ==
                sizeof(ImGui_ImplWGPU_FlameInstance));
  ImGui_ImplWGPU_UploadFlameInstances(
      reinterpret_cast<const ImGui_ImplWGPU_FlameInstance*>(
          gpu_flame_instances_.data()),
      gpu_flame_instances_.size());
#endif
}

bool Timeline::HasGpuFlameInstanceBuffer() const {
  if (gpu_flame_instances_.empty()) {
    return false;
  }
  if (force_gpu_flame_batch_mode_for_test_) {
    return true;
  }
#ifdef __EMSCRIPTEN__
  return ImGui_ImplWGPU_HasFlameInstanceBuffer();
#else
  return false;
#endif
}

void Timeline::AddGpuFlameBatch(ImDrawList* draw_list,
                                const GpuFlameBatch& batch) {
  if (!draw_list || batch.instance_count == 0) {
    return;
  }
  emitted_gpu_flame_batches_.push_back(batch);
#ifdef __EMSCRIPTEN__
  ImGui_ImplWGPU_AddFlameBatch(draw_list,
                               ImGui_ImplWGPU_FlameBatchParams{
                                   .first_instance = batch.first_instance,
                                   .instance_count = batch.instance_count,
                                   .visible_start_us = batch.visible_start_us,
                                   .px_per_us = batch.px_per_us,
                                   .screen_x_offset = batch.screen_x_offset,
                                   .timeline_width = batch.timeline_width,
                                   .y_top = batch.y_top,
                                   .y_bottom = batch.y_bottom,
                                   .min_width_px = batch.min_width_px,
                                   .padding_right_px = batch.padding_right_px,
                                   .alpha_multiplier = batch.alpha_multiplier,
                               });
#endif
}

void Timeline::ReleaseEntryMemory(FlameChartTimelineData& d) {
  d.ForEachEntryColumn(
      [](auto& col) { std::decay_t<decltype(col)>().swap(col); });
  d.ForEachSparseEntryMap([](auto& m) { std::decay_t<decltype(m)>().swap(m); });
  d.sparse_entry_group_ids.release();
  d.sparse_entry_hlo_module_indices.release();
  d.sparse_entry_hlo_op_indices.release();
  std::vector<std::vector<ModuleInterval>>().swap(d.hlo_module_intervals);
  std::vector<FlameChartTimelineData::HloTrackRule>().swap(d.hlo_track_rules);
  d.sparse_self_times.release();
  std::vector<int>().swap(d.level_event_indices);
  std::vector<size_t>().swap(d.level_offsets);
  std::vector<FlowLine>().swap(d.flow_lines);
  decltype(d.flow_ids_by_event_id)().swap(d.flow_ids_by_event_id);
  decltype(d.flow_lines_by_flow_id)().swap(d.flow_lines_by_flow_id);
  decltype(d.flow_line_ranges_by_flow_id)().swap(d.flow_line_ranges_by_flow_id);
  decltype(d.counter_data_by_group_index)().swap(d.counter_data_by_group_index);
  std::vector<std::string>().swap(d.interned_string_pool);
  std::vector<std::string>().swap(d.hlo_module_table);
  std::vector<std::string>().swap(d.hlo_op_table);
}

size_t Timeline::EstimateRetainedEntryBytes() const {
  size_t bytes = 0;
  const FlameChartTimelineData& d = timeline_data_;
  d.ForEachEntryColumn([&bytes](const auto& col) {
    using ValueType = typename std::decay_t<decltype(col)>::value_type;
    bytes += col.capacity() * sizeof(ValueType);
  });
  for (const std::string& s : d.entry_names) {
    bytes += s.capacity();
  }
  const auto estimate_string_vec =
      [](const std::vector<std::string>& vec) -> size_t {
    size_t total = vec.capacity() * sizeof(std::string);
    for (const std::string& s : vec) {
      total += s.capacity();
    }
    return total;
  };
  const auto estimate_arg_map =
      [](const absl::flat_hash_map<std::string, std::string>& m) -> size_t {
    using Map = absl::flat_hash_map<std::string, std::string>;
    size_t total = m.capacity() * (sizeof(Map::value_type) + 1);
    for (const auto& [k, v] : m) {
      total += k.capacity() + v.capacity();
    }
    return total;
  };
  for (const auto& m : d.entry_args) {
    bytes += estimate_arg_map(m);
  }
  bytes += entry_colors_.capacity() * sizeof(ImU32);
  bytes += d.level_event_indices.capacity() * sizeof(int);
  bytes += d.level_offsets.capacity() * sizeof(size_t);
  bytes += estimate_string_vec(d.interned_string_pool);
  bytes += estimate_string_vec(d.hlo_module_table);
  bytes += estimate_string_vec(d.hlo_op_table);
  bytes += d.sparse_entry_group_ids.AllocatedBytes();
  bytes += d.sparse_entry_hlo_module_indices.AllocatedBytes();
  bytes += d.sparse_entry_hlo_op_indices.AllocatedBytes();
  bytes += d.sparse_self_times.AllocatedBytes();
  bytes += d.non_zero_uids_by_event_index.capacity() *
           (sizeof(decltype(d.non_zero_uids_by_event_index)::value_type) + 1);
  bytes += d.flow_ids_by_event_index.capacity() *
           (sizeof(decltype(d.flow_ids_by_event_index)::value_type) + 1);
  bytes += d.flow_line_ranges_by_flow_id.capacity() *
           (sizeof(decltype(d.flow_line_ranges_by_flow_id)::value_type) + 1);
  using SparseArgsMap = decltype(d.sparse_entry_args);
  bytes +=
      d.sparse_entry_args.capacity() * (sizeof(SparseArgsMap::value_type) + 1);
  for (const auto& [idx, m] : d.sparse_entry_args) {
    bytes += estimate_arg_map(m);
  }
  bytes += d.flow_lines.capacity() * sizeof(FlowLine);
  bytes += d.hlo_track_rules.capacity() *
           sizeof(FlameChartTimelineData::HloTrackRule);
  bytes +=
      d.hlo_module_intervals.capacity() * sizeof(std::vector<ModuleInterval>);
  for (const std::vector<ModuleInterval>& intervals : d.hlo_module_intervals) {
    bytes += intervals.capacity() * sizeof(ModuleInterval);
  }
  return bytes;
}

void Timeline::CompactTimelineDataToMipOverview() {
  // Only compact downsampled overviews (`is_compacted_`) when incremental
  // refetching is enabled. Full-resolution loads (`!is_compacted_`) retain all
  // raw events in memory so zoom-in does not lose non-dominant events.
  if (!incremental_fetch_enabled_ || !is_compacted_) {
    return;
  }
  const size_t old_n = timeline_data_.entry_start_times.size();
  // Retain all raw events when both entry count and estimated retained entry
  // bytes fit within the uncompacted memory budget.
  if (old_n <= max_uncompacted_overview_entries_ &&
      EstimateRetainedEntryBytes() <= max_uncompacted_overview_bytes_) {
    is_compacted_ = false;
    window_at_full_resolution_ = true;
    return;
  }

  std::vector<int> old_to_new(old_n, -1);
  if (selected_event_index_ >= 0 &&
      static_cast<size_t>(selected_event_index_) < old_n) {
    old_to_new[selected_event_index_] = 0;
  }
  matching_event_indices_.ForEachSetBit([&](int idx) {
    if (idx >= 0 && static_cast<size_t>(idx) < old_n) {
      old_to_new[idx] = 0;
    }
  });

  const int total_lvls = timeline_data_.total_levels();
  std::vector<uint8_t> level_compacted(total_lvls, 0);
  for (int lvl = 0; lvl < total_lvls; ++lvl) {
    EnsureLevelMipPyramid(lvl);
    const absl::Span<const int> indices = timeline_data_.level_events(lvl);
    const bool has_mip =
        static_cast<size_t>(lvl) < level_mip_pyramids_.size() &&
        level_mip_pyramids_[lvl].has_pyramid;
    if (!has_mip || indices.size() < 64) {
      for (int idx : indices) {
        if (idx >= 0 && static_cast<size_t>(idx) < old_n) {
          old_to_new[idx] = 0;
        }
      }
    } else {
      level_compacted[lvl] = 1;
      for (int m = 0; m < LevelMipPyramid::kNumMipLevels; ++m) {
        for (const auto& tile : level_mip_pyramids_[lvl].levels[m].tiles) {
          if (tile.dominant_event_index >= 0 &&
              static_cast<size_t>(tile.dominant_event_index) < old_n) {
            old_to_new[tile.dominant_event_index] = 0;
          }
        }
      }
      if (!indices.empty()) {
        if (indices.front() >= 0 &&
            static_cast<size_t>(indices.front()) < old_n) {
          old_to_new[indices.front()] = 0;
        }
        if (indices.back() >= 0 &&
            static_cast<size_t>(indices.back()) < old_n) {
          old_to_new[indices.back()] = 0;
        }
      }
    }
  }

  // Encode kept entries as `new_idx >= 0` and unkept entries as
  // `~static_cast<int>(kept_before)` (`< 0`), allowing `kept_before(idx)` to
  // be queried in O(1) without allocating a separate `keep` or `kept_before`
  // vector.
  size_t new_n = 0;
  for (size_t i = 0; i < old_n; ++i) {
    if (old_to_new[i] >= 0) {
      old_to_new[i] = static_cast<int>(new_n++);
    } else {
      old_to_new[i] = ~static_cast<int>(new_n);
    }
  }
  const auto kept_before = [&old_to_new, old_n, new_n](size_t idx) -> size_t {
    if (idx >= old_n) return new_n;
    const int v = old_to_new[idx];
    return static_cast<size_t>(v >= 0 ? v : ~v);
  };
  if (new_n >= old_n) {
    is_compacted_ = false;
    window_at_full_resolution_ = true;
    return;
  }
  if (process_utilization_pyramids_dirty_) {
    BuildProcessUtilizationPyramids();
  }
  for (int lvl = 0; lvl < total_lvls; ++lvl) {
    if (level_compacted[lvl]) {
      level_mip_pyramids_[lvl].raw_events_compacted = true;
    }
  }

  auto compact_vec = [&](auto& vec) {
    if (vec.size() != old_n) return;
    using T = typename std::decay_t<decltype(vec)>::value_type;
    std::vector<T> out;
    out.reserve(new_n);
    for (size_t i = 0; i < old_n; ++i) {
      if (old_to_new[i] >= 0) {
        out.push_back(std::move(vec[i]));
      }
    }
    vec.swap(out);
  };

  timeline_data_.ForEachEntryColumn(compact_vec);
  if (timeline_data_.interned_string_pool.empty()) {
    compact_vec(entry_colors_);
  }

  timeline_data_.ForEachSparseEntryMap([&](auto& m) {
    if (m.empty()) return;
    using Map = std::decay_t<decltype(m)>;
    Map remapped;
    remapped.reserve(m.size());
    for (auto& [old_idx, value] : m) {
      const size_t i = static_cast<size_t>(old_idx);
      if (i < old_n && old_to_new[i] >= 0) {
        remapped.try_emplace(static_cast<typename Map::key_type>(old_to_new[i]),
                             std::move(value));
      }
    }
    m = std::move(remapped);
  });

  if (!timeline_data_.sparse_entry_group_ids.empty()) {
    GroupIdRuns remapped;
    for (const auto& run : timeline_data_.sparse_entry_group_ids.runs()) {
      const size_t new_begin = kept_before(run.begin);
      const size_t new_end = kept_before(run.end);
      for (size_t new_idx = new_begin; new_idx < new_end; ++new_idx) {
        remapped.Append(new_idx, run.group_id);
      }
    }
    timeline_data_.sparse_entry_group_ids = std::move(remapped);
  }

  auto remap_index_runs = [&](IndexRuns& runs) {
    if (runs.empty()) return;
    IndexRuns remapped;
    for (const auto& run : runs.runs()) {
      const size_t new_begin = kept_before(run.begin);
      const size_t new_end = kept_before(run.end);
      for (size_t new_idx = new_begin; new_idx < new_end; ++new_idx) {
        remapped.Append(new_idx, run.value);
      }
    }
    runs = std::move(remapped);
  };
  remap_index_runs(timeline_data_.sparse_entry_hlo_module_indices);
  remap_index_runs(timeline_data_.sparse_entry_hlo_op_indices);

  if (!timeline_data_.sparse_self_times.empty()) {
    SparseSelfTimes remapped;
    for (size_t i = 0; i < old_n; ++i) {
      if (old_to_new[i] < 0) continue;
      if (const std::optional<Microseconds> self_time =
              timeline_data_.sparse_self_times.Find(i);
          self_time.has_value()) {
        remapped.Append(static_cast<size_t>(old_to_new[i]), *self_time);
      }
    }
    timeline_data_.sparse_self_times = std::move(remapped);
  }

  for (FlameChartTimelineData::HloTrackRule& rule :
       timeline_data_.hlo_track_rules) {
    rule.begin_entry = kept_before(rule.begin_entry);
    rule.end_entry = kept_before(rule.end_entry);
  }

  std::vector<size_t> new_level_offsets(total_lvls + 1, 0);
  std::vector<int> new_level_event_indices;
  new_level_event_indices.reserve(new_n);
  for (int lvl = 0; lvl < total_lvls; ++lvl) {
    new_level_offsets[lvl] = new_level_event_indices.size();
    for (int idx : timeline_data_.level_events(lvl)) {
      if (idx >= 0 && static_cast<size_t>(idx) < old_n &&
          old_to_new[idx] >= 0) {
        new_level_event_indices.push_back(old_to_new[idx]);
      }
    }
  }
  new_level_offsets[total_lvls] = new_level_event_indices.size();
  timeline_data_.level_offsets.swap(new_level_offsets);
  timeline_data_.level_event_indices.swap(new_level_event_indices);

  for (auto& pyramid : level_mip_pyramids_) {
    for (int m = 0; m < LevelMipPyramid::kNumMipLevels; ++m) {
      for (auto& tile : pyramid.levels[m].tiles) {
        if (tile.dominant_event_index >= 0 &&
            static_cast<size_t>(tile.dominant_event_index) < old_n) {
          const int mapped = old_to_new[tile.dominant_event_index];
          tile.dominant_event_index = mapped >= 0 ? mapped : -1;
        }
      }
    }
  }

  if (selected_event_index_ >= 0 &&
      static_cast<size_t>(selected_event_index_) < old_n &&
      old_to_new[selected_event_index_] >= 0) {
    selected_event_index_ = old_to_new[selected_event_index_];
  } else {
    selected_event_index_ = -1;
  }

  if (event_index_to_scroll_to_ >= 0 &&
      static_cast<size_t>(event_index_to_scroll_to_) < old_n &&
      old_to_new[event_index_to_scroll_to_] >= 0) {
    event_index_to_scroll_to_ = old_to_new[event_index_to_scroll_to_];
  } else {
    event_index_to_scroll_to_ = -1;
  }

  auto remap_bitset = [&](EntryBitset& bitset) {
    if (bitset.empty()) return;
    EntryBitset remapped;
    remapped.Reserve(new_n);
    bitset.ForEachSetBit([&](int idx) {
      if (idx >= 0 && static_cast<size_t>(idx) < old_n &&
          old_to_new[idx] >= 0) {
        remapped.insert(old_to_new[idx]);
      }
    });
    bitset = std::move(remapped);
  };
  remap_bitset(selected_event_indices_);
  remap_bitset(matching_event_indices_);
  remap_bitset(filtered_event_indices_);

  for (auto& res : search_results_) {
    if (res.loaded_index >= 0 &&
        static_cast<size_t>(res.loaded_index) < old_n) {
      const int mapped = old_to_new[res.loaded_index];
      res.loaded_index = mapped >= 0 ? mapped : -1;
    }
  }
  is_compacted_ = true;
  full_trace_loaded_ = false;
  window_at_full_resolution_ = false;
}

bool Timeline::RestoreSavedFullOverview() {
  if (!saved_full_overview_.has_value()) {
    return false;
  }

  absl::flat_hash_map<GroupKey, bool> current_expanded;
  for (size_t i = 0; i < timeline_data_.groups.size(); ++i) {
    const Group& g = timeline_data_.groups[i];
    std::string parent_name =
        (g.parent_index >= 0 &&
         g.parent_index < static_cast<int>(timeline_data_.groups.size()))
            ? timeline_data_.groups[g.parent_index].name
            : "";
    current_expanded[GroupKey{g.nesting_level, g.name, parent_name}] =
        g.expanded;
  }
  auto& restored_groups = saved_full_overview_->data.groups;
  for (size_t i = 0; i < restored_groups.size(); ++i) {
    Group& g = restored_groups[i];
    std::string parent_name =
        (g.parent_index >= 0 &&
         g.parent_index < static_cast<int>(restored_groups.size()))
            ? restored_groups[g.parent_index].name
            : "";
    if (auto it = current_expanded.find(
            GroupKey{g.nesting_level, g.name, parent_name});
        it != current_expanded.end()) {
      g.expanded = it->second;
    }
  }

  UpdateLevelPositions(saved_full_overview_->data);
  ReleaseEntryMemory(timeline_data_);
  timeline_data_ = std::move(saved_full_overview_->data);
  entry_colors_ = std::move(saved_full_overview_->entry_colors);
  cached_trace_colors_version_ =
      saved_full_overview_->cached_trace_colors_version;
  level_mip_pyramids_ = std::move(saved_full_overview_->level_mip_pyramids);
  process_utilization_pyramids_ =
      std::move(saved_full_overview_->process_utilization_pyramids);
  gpu_flame_instances_ = std::move(saved_full_overview_->gpu_flame_instances);
  process_utilization_pyramids_dirty_ =
      saved_full_overview_->process_utilization_pyramids_dirty;
  fetched_data_time_range_ = saved_full_overview_->fetched_data_time_range;
  saved_full_overview_.reset();
  is_full_overview_data_ = true;
  stash_full_overview_on_next_update_ = false;
  selected_event_index_ = -1;
  selected_event_indices_.clear();
  matching_event_indices_.clear();
  if (active_event_filter_.has_value()) {
    filtered_event_indices_ = EvaluateEventFilter(*active_event_filter_);
    has_active_event_filter_ = true;
    ++filter_epoch_;
    process_utilization_pyramids_dirty_ = true;
  } else {
    filtered_event_indices_.release();
    has_active_event_filter_ = false;
  }
  if (!search_query_lower_.empty() && search_results_.empty()) {
    RecomputeSearchResults();
  }
  if (data_time_range_.duration() > 0) {
    last_fetch_request_range_ = data_time_range_;
  }
#ifdef __EMSCRIPTEN__
  if (!gpu_flame_instances_.empty()) {
    ImGui_ImplWGPU_UploadFlameInstances(
        reinterpret_cast<const ImGui_ImplWGPU_FlameInstance*>(
            gpu_flame_instances_.data()),
        gpu_flame_instances_.size());
  }
#endif
  if (redraw_callback_) redraw_callback_();
  return true;
}

ImU32 Timeline::GetEventColor(int event_index) const {
  if (cached_trace_colors_version_ == palette_.GetTraceVersion() &&
      event_index >= 0) {
    if (!timeline_data_.interned_string_pool.empty() &&
        static_cast<size_t>(event_index) <
            timeline_data_.entry_name_indices.size()) {
      const uint32_t name_idx = timeline_data_.entry_name_indices[event_index];
      if (name_idx < entry_colors_.size()) {
        return entry_colors_[name_idx];
      }
    } else if (static_cast<size_t>(event_index) < entry_colors_.size()) {
      return entry_colors_[event_index];
    }
  }
  return GetColorForId(GetEventName(event_index), palette_.GetTraceColors());
}

void Timeline::BuildProcessUtilizationPyramids() {
  process_utilization_pyramids_dirty_ = false;
  process_utilization_pyramids_.clear();
  const int num_levels = timeline_data_.total_levels();
  if (num_levels == 0 || timeline_data_.entry_start_times.empty()) {
    return;
  }

  for (size_t g_idx = 0; g_idx < timeline_data_.groups.size(); ++g_idx) {
    const Group& g = timeline_data_.groups[g_idx];
    if (g.nesting_level != kProcessNestingLevel ||
        g.type != Group::Type::kFlame) {
      continue;
    }

    int proc_end = num_levels;
    for (size_t j = g_idx + 1; j < timeline_data_.groups.size(); ++j) {
      if (timeline_data_.groups[j].nesting_level <= g.nesting_level) {
        proc_end = std::min(timeline_data_.groups[j].start_level, num_levels);
        break;
      }
    }

    std::optional<ProcessUtilizationPyramid> proc_pyr =
        BuildProcessUtilizationPyramidForRange(g.start_level, proc_end);
    if (proc_pyr.has_value()) {
      process_utilization_pyramids_.push_back(*std::move(proc_pyr));
    }
  }
}

std::optional<Timeline::ProcessUtilizationPyramid>
Timeline::BuildProcessUtilizationPyramidForRange(int proc_start,
                                                 int proc_end) const {
  if (proc_end <= proc_start) return std::nullopt;

  const auto& start_times = timeline_data_.entry_start_times;
  const auto& total_times = timeline_data_.entry_total_times;
  Microseconds proc_min_ts = std::numeric_limits<Microseconds>::max();
  Microseconds proc_max_ts = std::numeric_limits<Microseconds>::lowest();
  size_t proc_event_count = 0;

  for (int lvl = proc_start; lvl < proc_end; ++lvl) {
    if (static_cast<size_t>(lvl) < level_mip_pyramids_.size()) {
      const LevelMipPyramid& lpyr = level_mip_pyramids_[lvl];
      if (lpyr.has_pyramid && !lpyr.raw_events_compacted &&
          lpyr.built_filter_epoch == filter_epoch_ &&
          lpyr.visible_event_count > 0) {
        proc_event_count += lpyr.visible_event_count;
        proc_min_ts = std::min(proc_min_ts, lpyr.level_min_ts);
        proc_max_ts = std::max(proc_max_ts, lpyr.level_max_ts);
        continue;
      }
    }
    const absl::Span<const int> indices = timeline_data_.level_events(lvl);
    for (int idx : indices) {
      if (idx < 0 || static_cast<size_t>(idx) >= start_times.size()) continue;
      if (has_active_event_filter_ && !filtered_event_indices_.contains(idx)) {
        continue;
      }
      ++proc_event_count;
      const Microseconds s = start_times[idx];
      proc_min_ts = std::min(proc_min_ts, s);
      proc_max_ts = std::max(proc_max_ts, s + total_times[idx]);
    }
  }

  if (proc_event_count < kMinEventsForProcessUtilizationPyramid ||
      proc_max_ts <= proc_min_ts) {
    return std::nullopt;
  }

  ProcessUtilizationPyramid proc_pyr;
  proc_pyr.start_level = proc_start;
  proc_pyr.end_level = proc_end;
  const Microseconds proc_span = proc_max_ts - proc_min_ts;

  double bin_w[LevelMipPyramid::kNumMipLevels] = {};
  double inv_bin_w[LevelMipPyramid::kNumMipLevels] = {};
  for (int m = 0; m < LevelMipPyramid::kNumMipLevels; ++m) {
    const int num_bins = kMipBinCounts[m];
    if (num_bins <= 0) return std::nullopt;
    bin_w[m] = proc_span / static_cast<double>(num_bins);
    if (bin_w[m] <= 0.0) return std::nullopt;
    inv_bin_w[m] = 1.0 / bin_w[m];
    ProcessUtilizationMip& pmip = proc_pyr.levels[m];
    pmip.min_ts = proc_min_ts;
    pmip.bin_width_us = bin_w[m];
    pmip.occupied_us_bins.assign(num_bins, 0.0f);
  }

  const double bw0 = bin_w[0];
  const double inv_bw0 = inv_bin_w[0];
  const int max_b0 = kMipBinCounts[0] - 1;
  std::vector<float>& bins0 = proc_pyr.levels[0].occupied_us_bins;

  for (int lvl = proc_start; lvl < proc_end; ++lvl) {
    for (int idx : timeline_data_.level_events(lvl)) {
      if (idx < 0 || static_cast<size_t>(idx) >= start_times.size()) continue;
      if (has_active_event_filter_ && !filtered_event_indices_.contains(idx)) {
        continue;
      }
      const Microseconds s = start_times[idx];
      const Microseconds d = total_times[idx];
      if (d <= 0.0) continue;
      const Microseconds e = s + d;
      const Microseconds rel_s = s - proc_min_ts;
      const int b0 = std::clamp(static_cast<int>(rel_s * inv_bw0), 0, max_b0);
      const Microseconds b0_end = proc_min_ts + (b0 + 1) * bw0;
      if (e <= b0_end || b0 == max_b0) {
        bins0[b0] += static_cast<float>(d);
        continue;
      }
      const Microseconds rel_e = e - proc_min_ts;
      const int b1 = std::clamp(static_cast<int>(rel_e * inv_bw0), b0, max_b0);
      for (int b = b0; b <= b1; ++b) {
        const Microseconds bs = proc_min_ts + b * bw0;
        const Microseconds ov = std::min(e, bs + bw0) - std::max(s, bs);
        if (ov > 0.0) {
          bins0[b] += static_cast<float>(ov);
        }
      }
    }
  }

  static_assert(kMipBinCounts[0] == 4 * kMipBinCounts[1]);
  static_assert(kMipBinCounts[1] == 4 * kMipBinCounts[2]);
  for (int m = 1; m < LevelMipPyramid::kNumMipLevels; ++m) {
    const std::vector<float>& prev = proc_pyr.levels[m - 1].occupied_us_bins;
    std::vector<float>& cur = proc_pyr.levels[m].occupied_us_bins;
    const int cur_n = kMipBinCounts[m];
    for (int i = 0; i < cur_n; ++i) {
      const int base = i * 4;
      cur[i] =
          (prev[base] + prev[base + 1]) + (prev[base + 2] + prev[base + 3]);
    }
  }
  return proc_pyr;
}

void Timeline::ResetForNewDataset() {
  // Keep MaybeRequestData() silent until the next parse clears this flag.
  is_incremental_loading_ = true;

  // Selection and hover.
  selected_event_index_ = -1;
  selected_group_index_ = -1;
  selected_counter_index_ = -1;
  selected_event_indices_.clear();
  selected_counter_points_.clear();
  hovered_event_index_ = -1;
  last_reported_hovered_event_index_ = -1;
  event_index_to_scroll_to_ = -1;

  // Search.
  search_query_lower_.clear();
  search_results_.clear();
  matching_event_indices_.clear();
  current_search_result_index_ = -1;
  pending_navigation_event_id_.reset();

  // Time-range selection, bookmarks and in-progress interaction.
  selected_time_ranges_.clear();
  current_selected_time_range_.reset();
  time_range_resizing_state_.reset();
  bookmarks_.clear();
  selection_start_pos_.reset();
  selection_end_pos_.reset();
  is_dragging_ = false;
  is_selecting_ = false;

  // Timeline player.
  is_playing_ = false;
  current_play_time_ = -1.0;

  // Data and layout caches.
  active_event_filter_.reset();
  has_active_event_filter_ = false;
  filtered_event_indices_.release();
  ++filter_epoch_;
  ClearSavedFullOverview();
  SetTimelineData({});
  should_restore_scroll_ = false;
  last_scroll_y_ = 0.0f;
  // Draw() only rebuilds colors when the entry count changes.
  entry_colors_.clear();
  group_accent_colors_.clear();
  copied_track_group_index_ = -1;
  layout_transition_ = LayoutTransition();
  animate_next_layout_ = false;
  chevron_openness_.clear();
  hover_fades_.clear();

  // A zero visible range makes the next ProcessTraceEvents fit the new
  // dataset to width.
  fetched_data_time_range_ = TimeRange::Zero();
  data_time_range_ = TimeRange::Zero();
  last_fetch_request_range_ = TimeRange::Zero();
  SetVisibleRange(TimeRange::Zero());
}

void Timeline::Draw() {
  emitted_gpu_flame_batches_.clear();
  const size_t expected_colors_size =
      !timeline_data_.interned_string_pool.empty()
          ? timeline_data_.interned_string_pool.size()
          : timeline_data_.entry_names.size();
  if (cached_trace_colors_version_ != palette_.GetTraceVersion() ||
      entry_colors_.size() != expected_colors_size) {
    RebuildEntryColors();
    BuildLevelMipPyramids();
    UploadGpuFlameInstances();
  }
#ifdef __EMSCRIPTEN__
  if (!gpu_flame_instances_.empty() &&
      !ImGui_ImplWGPU_HasFlameInstanceBuffer()) {
    ImGui_ImplWGPU_UploadFlameInstances(
        reinterpret_cast<const ImGui_ImplWGPU_FlameInstance*>(
            gpu_flame_instances_.data()),
        gpu_flame_instances_.size());
  }
#endif
  hovered_event_index_ = -1;
  hovered_group_index_ = -1;
  hovered_bundle_ = -1;
  hovered_counter_value_ = 0.0;
  event_clicked_this_frame_ = false;
  bool is_resizer_hovered = false;
  bool needs_layout_update = false;
  motion_in_progress_ = false;
  AdvanceLayoutTransition(ImGui::GetIO().DeltaTime);

  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->Pos);
  ImGui::SetNextWindowSize(viewport->Size);

  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_CellPadding, ImVec2(0.0f, 0.0f));
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

  ImGui::Begin("Timeline viewer", nullptr, kImGuiWindowFlags);

  // Calculate the available width for the timeline before entering the table.
  // This ensures we get the correct width even if the table layout hasn't
  // finished or if GetContentRegionAvail behaves differently inside the table.
  // Reserve space for the vertical scrollbar so that the timeline layout
  // doesn't shift horizontally when the scrollbar appears/disappears due to
  // drawer resizing. We calculate the fixed table width based on the parent
  // window's available width, minus the scrollbar size.
  const Pixel content_region_avail_width =
      ImGui::GetWindowWidth() - ImGui::GetStyle().ScrollbarSize;

  current_timeline_width_ =
      content_region_avail_width - label_width_ - kTimelinePaddingRight;

  const double px_per_time_unit_val = px_per_time_unit(current_timeline_width_);
  const TickInfo tick_info = CalculateTickInfo(px_per_time_unit_val);

  const ImVec2 ruler_start_pos = ImGui::GetCursorPos();
  const ImVec2 ruler_start_screen_pos = ImGui::GetCursorScreenPos();
  ruler_screen_y_ = ruler_start_screen_pos.y;

  // Draw Ruler background anchored to the top (outside the scrollable child
  // window). The background covers the entire row.
  ImGui::GetWindowDrawList()->AddRectFilled(
      ImVec2(ruler_start_screen_pos.x, ruler_start_screen_pos.y),
      ImVec2(ruler_start_screen_pos.x + content_region_avail_width,
             ruler_start_screen_pos.y + kRulerHeight),
      palette_.GetColor(ColorPalette::Key::kMidtone).value_or(kLightGrayColor));

  ImGui::SetCursorPos(ruler_start_pos);
  DrawRulerUI(tick_info, current_timeline_width_);

  if (track_management_enabled_) {
    // Draw "Process" header text in the top-left empty label area of
    // the ruler row.
    ImFont* const header_font = traceviewer::fonts::label_medium_strong;
    // Vertically center "Process" within ruler height
    const Pixel vertical_offset =
        (kRulerHeight - FontLineHeight(header_font)) * 0.5f;
    DrawSpacedUppercaseText(
        ImGui::GetWindowDrawList(), header_font, kProcessHeaderLabel,
        ImVec2(ruler_start_screen_pos.x + kTrackLabelPaddingLeft +
                   kTrackChevronSlotWidth * 0.5f,
               ruler_start_screen_pos.y + std::max(0.0f, vertical_offset)),
        WithOpacity(palette_.GetColor(ColorPalette::Key::kForeground)
                        .value_or(kBlackColor),
                    kTrackMutedTextOpacity));
  }

  // Now move the cursor below the Ruler to start the Tracks child
  ImGui::SetCursorPos(
      ImVec2(ruler_start_pos.x, ruler_start_pos.y + kRulerHeight));

  // The tracks are in a child window to allow scrolling independently of the
  // ruler.
  // Keep the NoScrollWithMouse flag to disable the default scroll behavior
  // of ImGui, and use the custom scroll handler defined in `HandleWheel`
  // instead.
  ImGui::BeginChild("Tracks", ImVec2(0, 0), 0,
                    ImGuiWindowFlags_NoScrollWithMouse);

  if (should_restore_scroll_) {
    ImGui::SetScrollY(last_scroll_y_);
    should_restore_scroll_ = false;
  }

  // We set cursor to 0,0 locally
  const ImVec2 tracks_start_pos = ImGui::GetCursorPos();
  const ImVec2 tracks_start_screen_pos = ImGui::GetCursorScreenPos();
  tracks_start_screen_pos_ = tracks_start_screen_pos;

  if (show_grid_) {
    DrawVerticalGridLines(tick_info, current_timeline_width_,
                          viewport->Pos.y + viewport->Size.y);
  }

  const Pixel scroll_y = ImGui::GetScrollY();
  const Pixel window_height = ImGui::GetWindowHeight();

  // Find first visible item or item intersecting scroll_y using binary search.
  // Rows tucked away by an accordion transition break the ordering the search
  // relies on, so transitions (which last a few frames) scan from the start.
  auto it =
      layout_transition_.active
          ? flattened_groups_.begin()
          : std::lower_bound(flattened_groups_.begin(), flattened_groups_.end(),
                             scroll_y, [this](const Group* group, Pixel y) {
                               return this->GetGroupBottom(group) < y;
                             });

  // Draw visible groups.
  for (; it != flattened_groups_.end(); ++it) {
    const Group* group_ptr = *it;
    const bool is_header = IsVirtualHeader(group_ptr);
    int group_index = -1;
    if (!is_header) {
      group_index = group_ptr - &timeline_data_.groups[0];
      if (!IsGroupDrawn(group_index)) {
        continue;
      }
    }

    const Pixel group_top = GetGroupTop(group_ptr);
    if (group_top > scroll_y + window_height) {
      continue;
    }
    const Pixel group_bottom = GetGroupBottom(group_ptr);
    if (group_bottom < scroll_y) {
      // Rows above the viewport (only reachable while scanning linearly during
      // a transition; the binary search skips them).
      continue;
    }

    if (is_header) {
      if (DrawHeaderRow(group_ptr, tracks_start_screen_pos, group_top,
                        group_bottom)) {
        needs_layout_update = true;
      }
      continue;
    }

    if (DrawTrackRow(group_index, tracks_start_pos, tracks_start_screen_pos,
                     content_region_avail_width, px_per_time_unit_val, scroll_y,
                     window_height)) {
      needs_layout_update = true;
    }
  }

  // Create a dummy at the end to ensure the Tracks child has the right
  // scrolling height. We position the top of a 1px high dummy at
  // `group_offsets_.back() - 1.0f`. This means the bottom of the dummy is
  // exactly at `group_offsets_.back()`, ensuring the content size of the
  // child window matches the total calculated height from group offsets.
  // Using a 1px dummy instead of a 0-height dummy helps avoid potential issues
  // with ImGui's `ItemSpacing` adding extra space around zero-sized items.
  const Pixel total_tracks_height =
      group_offsets_.empty() ? 0.0f : group_offsets_.back();
  if (total_tracks_height > 0.0f) {
    if (!group_offsets_.empty() && group_offsets_.back() > 0.0f) {
    ImGui::SetCursorPos(
        ImVec2(0, tracks_start_pos.y + total_tracks_height - 1.0f));
    ImGui::Dummy(ImVec2(content_region_avail_width, 1.0f));
  }

    // Handle label resizing manually since we removed the table
    ImGui::SetCursorPos(ImVec2(
        tracks_start_pos.x + label_width_ - kSplitterOffset,
        tracks_start_pos.y));
    ImGui::InvisibleButton("##LabelResizer",
                           ImVec2(kSplitterWidth, group_offsets_.back()));
    if (ImGui::IsItemActive()) {
      label_width_ += ImGui::GetIO().MouseDelta.x;
      label_width_ = std::max(10.0f, label_width_);
      is_resizing_label_column_ = true;
    } else {
      is_resizing_label_column_ = false;
    }
    is_resizer_hovered = ImGui::IsItemHovered();
    if (is_resizer_hovered) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
      if (!is_resizing_label_column_) {
        ImGui::SetTooltip("%s", kDragToResizeTooltip);
      }
    } else if (is_resizing_label_column_) {
      ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
    }
  } else {
    is_resizing_label_column_ = false;
  }

  HandleEventDeselection();

  // Handle continuous keyboard and mouse wheel input for timeline navigation.
  // These functions are called every frame to ensure smooth and responsive
  // interaction.
  // The performance impact is fine because HandleKeyboard/HandleWheel() only
  // performs lightweight checks and calculations.
  bool is_interacting = is_resizing_label_column_;
  if (!is_resizing_label_column_) {
    is_interacting |= HandleKeyboard();
    is_interacting |= HandleWheel();
    is_interacting |= HandleMouse();
  }

  // We call MaybeRequestData() to check if we need to fetch more data.
  // We do this here instead of in SetVisibleRange because we want to debounce
  // the request during continuous interaction (like panning/zooming).
  // The check is skipped if the user is currently interacting with
  // the timeline to avoid sending requests during interaction.
  if (!is_interacting) {
    MaybeRequestData();
  }

  ProcessPendingScroll();

  last_scroll_y_ = ImGui::GetScrollY();
  ImGui::EndChild();

  // Draw the selected time range in a separate overlay child window.
  // This ensures it is drawn on top of the "Tracks" child window (because it's
  // declared after) but below tooltips (because it's a child window, not
  // in the foreground draw list).
  ImGui::SetCursorPos(ImVec2(0, 0));
  ImGui::BeginChild("SelectionOverlay", ImVec2(0, 0), 0,
                    ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                        ImGuiWindowFlags_NoSavedSettings |
                        ImGuiWindowFlags_NoInputs |
                        ImGuiWindowFlags_NoBackground);
  // Like a sticky CSS header, the ruler casts a soft shadow over the tracks
  // once they are scrolled underneath it.
  if (last_scroll_y_ > 0.0f) {
    DrawDropShadow(ImGui::GetWindowDrawList(), ruler_start_screen_pos.x,
                   ruler_start_screen_pos.x + content_region_avail_width,
                   ruler_start_screen_pos.y + kRulerHeight,
                   kStickyHeaderShadowHeight, kStickyHeaderShadowOpacity);
  }
  // `DrawFlows` and `DrawSelectedTimeRanges` should be called after all other
  // timeline content (events, ruler, etc.) has been drawn. This ensures that
  // flow lines and selected time ranges are rendered on top of everything
  // else within the current ImGui window, without affecting global foreground
  // elements like tooltips.
  DrawFlows(current_timeline_width_, tracks_start_screen_pos.y);
  DrawDependencyArrows(current_timeline_width_, tracks_start_screen_pos.y);
  DrawSelectedTimeRanges(current_timeline_width_, px_per_time_unit_val);
  DrawBookmarks(current_timeline_width_, px_per_time_unit_val);
  DrawSelectionRectangle();

  // =========================================================================
  // TIMELINE PLAYER SYNCHRONIZATION LOGIC
  // =========================================================================
  DrawTimelinePlayerSync();

  // Render the red vertical playhead (scrubber line) slicing across all tracks.
  // Only rendered if the feature flag is on and playhead is on-screen.
  if (timeline_player_enabled_) {
    if (current_play_time_ >= visible_range().start() &&
        current_play_time_ <= visible_range().end()) {
      const ImRect timeline_area = GetTimelineArea();
      const Pixel screen_x_offset = timeline_area.Min.x;
      const Pixel x = TimeToScreenX(current_play_time_, screen_x_offset,
                                    px_per_time_unit_val);
      ImGui::GetForegroundDrawList()->AddLine(ImVec2(x, timeline_area.Min.y),
                                              ImVec2(x, timeline_area.Max.y),
                                              IM_COL32(255, 0, 0, 255), 2.0f);
    }
  }

  // Draw vertical split line between sidebar and tracks
  // Drawn last inside SelectionOverlay so it sits on top of other elements,
  // and extends upwards to the beginning of the ruler.
  Pixel split_x = std::floor(ruler_start_screen_pos.x + label_width_) + 0.5f;
  ImU32 splitter_col = ImGui::GetColorU32(ImGuiCol_TableBorderLight);
  Pixel splitter_thickness = kSplitterDefaultThickness;
  if (is_resizing_label_column_) {
    splitter_col = kSplitterActiveColor;
    splitter_thickness = kSplitterHoverThickness;
  } else if (is_resizer_hovered) {
    splitter_col = kSplitterHoverColor;
    splitter_thickness = kSplitterHoverThickness;
  }

  ImGui::GetWindowDrawList()->AddLine(
      ImVec2(split_x, ruler_start_screen_pos.y),
      ImVec2(split_x, ruler_start_screen_pos.y + ImGui::GetWindowHeight()),
      splitter_col, splitter_thickness);

  ImGui::EndChild();

  DrawToast(absl::StrCat("Copied track name: ", copied_track_name_),
            copy_notification_timer_, 32.0f);

  float bounds_toast_offset = 32.0f;
  if (copy_notification_timer_ > 0.0f) {
    bounds_toast_offset += 48.0f;
  }
  DrawToast(bounds_notification_message_, bounds_notification_timer_,
            bounds_toast_offset);

  if (hovered_event_index_ != -1) {
    hovered_group_index_ = -1;
    hovered_bundle_ = -1;
  }
  if (hovered_event_index_ != last_reported_hovered_event_index_ ||
      hovered_group_index_ != last_reported_hovered_group_index_ ||
      hovered_bundle_ != last_reported_hovered_bundle_) {
    last_reported_hovered_event_index_ = hovered_event_index_;
    last_reported_hovered_group_index_ = hovered_group_index_;
    last_reported_hovered_bundle_ = hovered_bundle_;
    if (event_callback_) {
      ImVec2 mouse_pos = ImGui::GetMousePos();
      EmitEventHovered(hovered_event_index_, mouse_pos.x, mouse_pos.y);
    }
  }

  SyncMinimapState();

  ImGui::PopStyleVar();  // ItemSpacing
  ImGui::PopStyleVar();  // CellPadding
  ImGui::PopStyleVar();  // WindowPadding
  ImGui::PopStyleVar();  // WindowBorderSize
  ImGui::PopStyleVar();  // WindowRounding
  ImGui::End();          // Timeline viewer

  if (needs_layout_update) {
    UpdateLevelPositions(timeline_data_);
    if (redraw_callback_) redraw_callback_();
  } else if (motion_in_progress_ && redraw_callback_) {
    // Keep rendering frames until all motion has settled.
    redraw_callback_();
  }
}

bool Timeline::DrawHeaderRow(const Group* group_ptr,
                             const ImVec2& tracks_start_screen_pos,
                             Pixel group_top, Pixel group_bottom) {
  int header_id = kAllHeaderId;
  bool* expanded = &header_all_expanded_;
  int process_count = all_processes_count_;
  if (group_ptr->name == kHiddenHeaderName) {
    header_id = kHiddenHeaderId;
    expanded = &header_hidden_expanded_;
    process_count = hidden_processes_count_;
  } else if (group_ptr->name == kPinnedHeaderName) {
    header_id = kPinnedHeaderId;
    expanded = &header_pinned_expanded_;
    process_count = pinned_processes_count_;
  }
  ImGui::PushID(header_id);

  const ImVec2 row_min(tracks_start_screen_pos.x,
                       tracks_start_screen_pos.y + group_top);
  const ImVec2 row_max(tracks_start_screen_pos.x + label_width_,
                       tracks_start_screen_pos.y + group_bottom);
  const ImU32 foreground_color =
      palette_.GetColor(ColorPalette::Key::kForeground).value_or(kBlackColor);
  ImDrawList* const draw_list = ImGui::GetWindowDrawList();

  // The whole header row is a single hit target that toggles the section. It
  // stops short of the label resizer, which is submitted later and would
  // otherwise lose the overlapping hover.
  const Pixel hit_right = row_max.x - kSplitterOffset;
  bool toggled = false;
  bool hovered = false;
  if (hit_right > row_min.x && row_max.y > row_min.y) {
    ImGui::SetCursorScreenPos(row_min);
    // Rows are pointer targets only; keyboard arrows navigate the timeline.
    ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
    toggled = ImGui::InvisibleButton(
        "##section_header",
        ImVec2(hit_right - row_min.x, row_max.y - row_min.y));
    ImGui::PopItemFlag();
    hovered = ImGui::IsItemHovered();
  }
  if (toggled) {
    BeginExpandToggleMotion(header_id, *expanded);
    *expanded = !*expanded;
    event_clicked_this_frame_ = true;
  }
  const float hover_emphasis =
      HoverFade(ImGui::GetID("##section_hover"), hovered);
  if (hovered) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
  }
  if (hover_emphasis > 0.0f) {
    DrawRowPill(
        draw_list, row_min, row_max,
        WithOpacity(foreground_color, kTrackHoverOpacity * hover_emphasis));
  }

  // Limit drawing to the label column.
  ImGui::PushClipRect(row_min, row_max, true);

  const Pixel center_y = (row_min.y + row_max.y) * 0.5f;
  const ImU32 muted_color = WithOpacity(
      foreground_color, kTrackMutedTextOpacity + (kTrackSecondaryTextOpacity -
                                                  kTrackMutedTextOpacity) *
                                                     hover_emphasis);
  DrawExpandChevron(
      draw_list,
      ImVec2(row_min.x + kTrackLabelPaddingLeft + kTrackChevronSlotWidth * 0.5f,
             center_y),
      ChevronOpenness(header_id, *expanded), hover_emphasis, foreground_color);

  ImFont* const name_font = traceviewer::fonts::label_medium_strong;
  const ImVec2 name_pos(
      row_min.x + kTrackLabelPaddingLeft + kTrackChevronSlotWidth +
          kTrackLabelGap * 0.5f,
      std::round(center_y - FontLineHeight(name_font) * 0.5f));
  const Pixel name_width = DrawSpacedUppercaseText(
      draw_list, name_font, group_ptr->name, name_pos, muted_color);

  // AlphaNum formats into an inline buffer, avoiding a per-frame allocation.
  const absl::AlphaNum count_text(process_count);
  DrawCountChip(draw_list, count_text.Piece(),
                name_pos.x + name_width + kTrackLabelGap +
                    CountChipWidth(count_text.Piece()),
                center_y, WithOpacity(foreground_color, kTrackCountChipOpacity),
                muted_color);

  ImGui::PopClipRect();
  ImGui::PopID();

  return toggled;
}

void Timeline::CopyTrackName(int group_index) {
  const std::string& name = timeline_data_.groups[group_index].name;
  ImGui::SetClipboardText(name.c_str());
  traceviewer::CopyToClipboard(name);
  copied_track_name_ = name;
  copied_track_group_index_ = group_index;
  copy_notification_timer_ = 2.0f;
}

ImU32 Timeline::GetGroupAccentColor(int group_index) const {
  const Group& group = timeline_data_.groups[group_index];
  if (group.type == Group::Type::kCounter) {
    return kCounterTrackColor;
  }
  if (group.nesting_level == kProcessNestingLevel) {
    // Matches the color of the process utilization chart.
    return palette_.GetColor(ColorPalette::Key::kFlameHeader).value_or(kBlue70);
  }
  if (static_cast<size_t>(group_index) < group_accent_colors_.size() &&
      group_accent_colors_[group_index] != 0) {
    return group_accent_colors_[group_index];
  }
  return WithOpacity(
      palette_.GetColor(ColorPalette::Key::kForeground).value_or(kBlackColor),
      kTrackNeutralAccentOpacity);
}

bool Timeline::DrawTrackRow(int group_index, const ImVec2& tracks_start_pos,
                            const ImVec2& tracks_start_screen_pos,
                            Pixel content_region_avail_width,
                            double px_per_time_unit_val, Pixel scroll_y,
                            Pixel window_height) {
  // Rows that are fully folded away during a layout transition have nothing to
  // draw (and a zero-height child window would fill the remaining space).
  if (layout_transition_.active &&
      (static_cast<size_t>(group_index) >= group_heights_.size() ||
       group_heights_[group_index] < 0.5f)) {
    return false;
  }
  bool needs_layout_update = false;
  Group& group = timeline_data_.groups[group_index];
  // Rows sliding in or out during an accordion transition are clipped at a
  // moving edge; rows entirely above it are not drawn.
  const std::optional<Pixel> accordion_clip_top = AccordionClipTop(group_index);
  if (accordion_clip_top.has_value() && tracks_start_screen_pos.y +
                                                group_offsets_[group_index] +
                                                group_heights_[group_index] <=
                                            *accordion_clip_top) {
    return false;
  }
  ImGui::PushID(group_index);
  if (accordion_clip_top.has_value()) {
    ImGui::PushClipRect(
        ImVec2(tracks_start_screen_pos.x, *accordion_clip_top),
        ImVec2(tracks_start_screen_pos.x + content_region_avail_width +
                   ImGui::GetStyle().ScrollbarSize,
               std::numeric_limits<float>::max()),
        /*intersect_with_current_clip_rect=*/true);
  }

  const bool is_process = group.nesting_level == kProcessNestingLevel;
  const int own_level_count =
      GetNextGroupStartLevel(timeline_data_, group_index) - group.start_level;
  const bool expandable = group.type == Group::Type::kFlame &&
                          (group.has_children || own_level_count > 1);
  const bool is_collapsed = expandable && !group.expanded;

  // Row geometry in screen space. The label column spans [row_left,
  // label_right); the timeline canvas spans [label_right, row_right).
  const Pixel row_left = tracks_start_screen_pos.x;
  const Pixel row_right = row_left + content_region_avail_width;
  const Pixel row_top = tracks_start_screen_pos.y + group_offsets_[group_index];
  const Pixel row_height = group_heights_[group_index];
  const Pixel row_bottom = row_top + row_height;
  const Pixel label_right = row_left + label_width_ - kSplitterOffset;
  // The height of the first line of the row, used to vertically align the
  // chevron and the track name. Multi-level flame tracks align their label
  // with their first level.
  const Pixel first_line_height =
      is_process
          ? kProcessTrackHeight
          : (group.type == Group::Type::kFlame ? kEventHeight : row_height);

  const ImU32 foreground_color =
      palette_.GetColor(ColorPalette::Key::kForeground).value_or(kBlackColor);
  ImDrawList* const draw_list = ImGui::GetWindowDrawList();
  // How expanded the row looks, animated while its chevron rotates. A toggle
  // made this frame starts animating on the next one.
  const float openness =
      expandable ? ChevronOpenness(group_index, group.expanded) : 1.0f;

  // 1. Row background. Process headers get an elevated surface; child tracks
  // alternate a subtle stripe that spans both the label column and the canvas,
  // so a label is visually connected to its events. Splitting the gap between
  // consecutive child tracks keeps each row centered inside its visual band
  // while consecutive stripes still meet without a seam.
  const Pixel half_gap = kThreadTrackGap * 0.5f;
  const Pixel band_top =
      is_process ? row_top
                 : std::max(tracks_start_screen_pos.y, row_top - half_gap);
  const Pixel band_bottom = is_process ? row_bottom : (row_bottom + half_gap);
  if (is_process) {
    // Cross-fades between the collapsed and expanded surface colors while the
    // header's chevron rotates.
    const ImU32 bg_color =
        MixColors(palette_.GetColor(ColorPalette::Key::kCollapsedHeader)
                      .value_or(kProcessTrackCollapsedColor),
                  palette_.GetColor(ColorPalette::Key::kExpandedHeader)
                      .value_or(kProcessTrackExpandedColor),
                  openness);
    // A subtle top-to-bottom gradient, a 1px inner top highlight and a soft
    // drop shadow give the header an elevated, card-like surface (the
    // ImGui equivalent of CSS `linear-gradient` + `box-shadow`).
    const ImU32 top_color =
        MixColors(bg_color, foreground_color, kTrackHeaderGradientStrength);
    draw_list->AddRectFilledMultiColor(ImVec2(row_left, row_top),
                                       ImVec2(row_right, row_bottom), top_color,
                                       top_color, bg_color, bg_color);
    draw_list->AddLine(
        ImVec2(row_left, row_top + 0.5f), ImVec2(row_right, row_top + 0.5f),
        WithOpacity(foreground_color, kTrackHeaderHighlightOpacity), 1.0f);
    DrawDropShadow(draw_list, row_left, row_right, row_bottom,
                   kTrackHeaderShadowHeight, kTrackHeaderShadowOpacity);
  } else if (static_cast<size_t>(group_index) < group_striped_.size() &&
             group_striped_[group_index]) {
    draw_list->AddRectFilled(
        ImVec2(row_left, band_top), ImVec2(row_right, band_bottom),
        WithOpacity(foreground_color, kTrackZebraStripeOpacity));
  }

  // 2. The whole label row is a single hit target: clicking an expandable
  // track toggles it, clicking any other track copies its name, and
  // right-clicking any track copies its name. The track management buttons
  // (pin / hide) are excluded so they keep their own hit targets.
  const Pixel management_width = track_management_enabled_ && is_process
                                     ? CalculateTrackManagementButtonWidth()
                                     : 0.0f;
  const Pixel hit_right = label_right - management_width;
  bool row_clicked = false;
  bool row_hovered = false;
  bool row_hovered_for_tooltip = false;
  ImGui::SetCursorScreenPos(ImVec2(row_left, row_top));
  bool row_right_clicked = false;
  if (hit_right > row_left && row_height > 0.0f) {
    // Rows are pointer targets only; keyboard arrows navigate the timeline.
    ImGui::PushItemFlag(ImGuiItemFlags_NoNav, true);
    row_clicked = ImGui::InvisibleButton(
        "##track_label", ImVec2(hit_right - row_left, row_height));
    ImGui::PopItemFlag();
    row_hovered = ImGui::IsItemHovered();
    row_right_clicked = ImGui::IsItemClicked(ImGuiMouseButton_Right);
    row_hovered_for_tooltip =
        ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip);
  }
  // Include the management buttons area so the row stays highlighted while
  // the pointer moves onto the pin / hide buttons.
  const bool tracks_window_hovered =
      ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) && !is_dragging_ &&
      !is_resizing_label_column_;
  const bool label_area_hovered =
      row_hovered ||
      (management_width > 0.0f && tracks_window_hovered &&
       ImGui::IsMouseHoveringRect(ImVec2(hit_right, row_top),
                                  ImVec2(label_right, row_bottom)));
  // Whether the pointer is over this track's events on the canvas. The label
  // resizer's hit area (which straddles the splitter) is excluded.
  const bool canvas_row_hovered =
      !label_area_hovered && tracks_window_hovered &&
      ImGui::IsMouseHoveringRect(ImVec2(label_right + kSplitterWidth, row_top),
                                 ImVec2(row_right, row_bottom));
  // Evaluate full-row hover before pushing the sidebar clip rect (which clips
  // ImGui::IsMouseHoveringRect to the label column), so hovering the canvas row
  // also reserves space for the revealed pin / hide buttons.
  const bool management_buttons_visible =
      management_width > 0.0f &&
      (pinned_track_names_.contains(group.name) ||
       hidden_track_names_.contains(group.name) ||
       IsTrackRowHovered(group_index));

  if (row_clicked) {
    // Clicking a label must not clear the current event selection.
    event_clicked_this_frame_ = true;
    if (expandable) {
      BeginExpandToggleMotion(group_index, group.expanded);
      group.expanded = !group.expanded;
      needs_layout_update = true;
    } else {
      CopyTrackName(group_index);
    }
  }
  if (row_right_clicked) {
    CopyTrackName(group_index);
  }
  if (row_hovered) {
    ImGui::SetMouseCursor(expandable ? ImGuiMouseCursor_Hand
                                     : ImGuiMouseCursor_TextInput);
  }

  // Hover feedback highlights the entire row (both the track's label pill and
  // its events' row on the canvas) whether the pointer is over the label or
  // over the canvas row.
  const Pixel splitter_x = row_left + label_width_;
  const float label_emphasis =
      HoverFade(ImGui::GetID("##label_hover"), label_area_hovered);
  const float row_emphasis = HoverFade(
      ImGui::GetID("##row_hover"), label_area_hovered || canvas_row_hovered);
  if (row_emphasis > 0.0f) {
    draw_list->AddRectFilled(
        ImVec2(splitter_x, band_top), ImVec2(row_right, band_bottom),
        WithOpacity(foreground_color,
                    kTrackCanvasRowFocusOpacity * row_emphasis));
  }

  // Tree guides: one vertical line per ancestor level, centered under the
  // ancestor's chevron. Each row draws its own segment (including the gap
  // above it), so consecutive children form a continuous line.
  const int depth = std::max(0, group.nesting_level - kProcessNestingLevel);
  const ImU32 guide_color =
      WithOpacity(foreground_color, kTrackTreeGuideOpacity);
  ImGui::PushClipRect(ImVec2(row_left, row_top - kThreadTrackGap),
                      ImVec2(splitter_x, row_bottom), true);
  for (int level = 0; level < depth; ++level) {
    const Pixel guide_x = std::floor(row_left + kTrackLabelPaddingLeft +
                                     level * kTrackChevronSlotWidth +
                                     kTrackChevronSlotWidth * 0.5f) +
                          0.5f;
    draw_list->AddLine(ImVec2(guide_x, row_top - kThreadTrackGap),
                       ImVec2(guide_x, row_bottom), guide_color, 1.0f);
  }
  ImGui::PopClipRect();

  // 3. Label column content, clipped so it never bleeds into the canvas.
  ImGui::PushClipRect(ImVec2(row_left, band_top),
                      ImVec2(splitter_x, band_bottom), true);

  // Frame the track's title line (rather than the full multi-level row height)
  // and add 1px symmetric vertical breathing room on child tracks so the pill
  // is centered on `first_line_center_y`.
  const Pixel pill_top = is_process ? row_top : (row_top - 1.0f);
  const Pixel pill_bottom =
      is_process ? row_bottom : (row_top + first_line_height + 1.0f);
  if (row_emphasis > 0.0f) {
    DrawRowPill(
        draw_list, ImVec2(row_left, pill_top), ImVec2(label_right, pill_bottom),
        WithOpacity(foreground_color, kTrackHoverOpacity * row_emphasis));
  }
  if (copied_track_group_index_ == group_index) {
    // The copy flash fades out (or, without motion, shows briefly).
    const float flash = animations_enabled_
                            ? std::clamp((copy_notification_timer_ -
                                          (2.0f - kMotionCopyFlashSeconds)) /
                                             kMotionCopyFlashSeconds,
                                         0.0f, 1.0f)
                            : (copy_notification_timer_ > 1.8f ? 1.0f : 0.0f);
    if (flash > 0.0f) {
      DrawRowPill(draw_list, ImVec2(row_left, pill_top),
                  ImVec2(label_right, pill_bottom),
                  WithOpacity(kBlue60, kTrackCopyFlashOpacity * flash));
    }
  }

  // Chevron.
  const Pixel slot_left =
      row_left + kTrackLabelPaddingLeft + depth * kTrackChevronSlotWidth;
  const Pixel first_line_center_y = row_top + first_line_height * 0.5f;
  if (expandable) {
    DrawExpandChevron(
        draw_list,
        ImVec2(slot_left + kTrackChevronSlotWidth * 0.5f, first_line_center_y),
        openness, label_emphasis, foreground_color);
  }

  // Count chip on collapsed tracks: the number of hidden child tracks (or
  // levels, for a multi-level track without child tracks). The space of the
  // track management buttons is only reserved while they are visible (see
  // DrawPinButton and DrawHideButton), so names use the full width otherwise.
  const Pixel content_left = slot_left + kTrackChevronSlotWidth;
  const Pixel text_left = content_left + kTrackLabelGap;
  Pixel text_right = label_right - kTrackLabelPaddingRight -
                     (management_buttons_visible ? management_width : 0.0f);
  // The collapsed cues (count chip, stack hint) fade in as the row collapses
  // and out as it expands.
  const float collapsed_cue_alpha = expandable ? 1.0f - openness : 0.0f;
  const ImU32 accent_color =
      collapsed_cue_alpha > 0.0f ? GetGroupAccentColor(group_index) : 0;
  if (collapsed_cue_alpha > 0.0f) {
    const int hidden_count = group.has_children
                                 ? static_cast<int>(group.child_indices.size())
                                 : own_level_count;
    // "+N" reads as "N more inside". Formatted into a stack buffer to avoid a
    // per-frame allocation.
    char count_buffer[16];
    const int count_length =
        std::snprintf(count_buffer, sizeof(count_buffer), "+%d", hidden_count);
    const absl::string_view count_text(
        count_buffer,
        std::clamp<int>(count_length, 0, sizeof(count_buffer) - 1));
    if (hidden_count > 0 &&
        text_right - CountChipWidth(count_text) >= text_left) {
      // The chip is tinted with the track's accent color so it reads as part
      // of the track rather than generic chrome.
      const Pixel chip_left = DrawCountChip(
          draw_list, count_text, text_right, first_line_center_y,
          WithOpacity(accent_color,
                      kTrackCollapsedChipAccentOpacity * collapsed_cue_alpha),
          WithOpacity(foreground_color,
                      kTrackSecondaryTextOpacity * collapsed_cue_alpha));
      text_right = chip_left - kTrackLabelGap;
    }
  }

  // Track name and optional subtitle.
  absl::string_view display_name = group.name;
  const bool has_subtitle = !group.subtitle.empty();
  if (has_subtitle && display_name.size() > group.subtitle.size()) {
    display_name.remove_prefix(group.subtitle.size() + 1);
    if (!display_name.empty() && display_name.front() == '/') {
      display_name.remove_prefix(1);
    }
  }
  ImFont* const name_font = is_process ? traceviewer::fonts::title_small_strong
                                       : traceviewer::fonts::title_small;
  const ImU32 name_color =
      is_process ? foreground_color
                 : WithOpacity(foreground_color, kTrackSecondaryTextOpacity);
  const Pixel name_height = FontLineHeight(name_font);
  const Pixel subtitle_height =
      has_subtitle ? FontLineHeight(traceviewer::fonts::label_small) : 0.0f;
  const Pixel text_block_height = name_height + subtitle_height;
  // Roboto's cap-height center sits ~1px below `name_height * 0.5f` due to
  // ascender/descender metrics, so nudge single-line titles up by 1px for
  // optical centering inside the row pill.
  const Pixel name_top =
      std::round(row_top + (first_line_height - text_block_height) * 0.5f) -
      (has_subtitle ? 0.0f : 1.0f);
  const bool name_truncated =
      DrawTextWithEllipsis(name_font, display_name, name_color,
                           ImVec2(text_left, name_top), text_right);
  if (has_subtitle) {
    DrawTextWithEllipsis(traceviewer::fonts::label_small, group.subtitle,
                         palette_.GetColor(ColorPalette::Key::kSubtitle)
                             .value_or(kOnSecondaryFixedVariantColor),
                         ImVec2(text_left, name_top + name_height), text_right);
  }
  if (name_truncated && row_hovered_for_tooltip) {
    ShowCardTooltip(group.name);
  }

  ImGui::PopClipRect();

  if (collapsed_cue_alpha > 0.0f) {
    // The hint lives in the gap below the row, so it gets its own clip rect.
    ImGui::PushClipRect(ImVec2(row_left, row_bottom),
                        ImVec2(splitter_x, row_bottom + kThreadTrackGap), true);
    DrawCollapsedStackHint(draw_list, content_left,
                           label_right - kTrackLabelPaddingRight, row_bottom,
                           WithOpacity(accent_color, collapsed_cue_alpha));
    ImGui::PopClipRect();
  }

  if (DrawTrackManagementButtons(group_index, group, tracks_start_pos,
                                 first_line_height)) {
    needs_layout_update = true;
  }

  ImGui::SetCursorPos(ImVec2(tracks_start_pos.x + label_width_,
                             tracks_start_pos.y + group_offsets_[group_index]));

  if (is_collapsed) {
    DrawGroupPreview(group_index, px_per_time_unit_val);
  } else {
    DrawGroup(group_index, px_per_time_unit_val, scroll_y, window_height);
  }

  if (HandleProcessTrackHeaderClick(group_index, group, tracks_start_screen_pos,
                                    content_region_avail_width)) {
    needs_layout_update = true;
  }
  if (accordion_clip_top.has_value()) {
    ImGui::PopClipRect();
  }
  ImGui::PopID();

  return needs_layout_update;
}

bool Timeline::HandleProcessTrackHeaderClick(
    int group_index, Group& group, const ImVec2& tracks_start_screen_pos,
    Pixel content_region_avail_width) {
  if (group.nesting_level != kProcessNestingLevel || !group.has_children) {
    return false;
  }

  const Pixel timeline_start_x = tracks_start_screen_pos.x + label_width_;
  const Pixel timeline_end_x =
      tracks_start_screen_pos.x + content_region_avail_width;
  if (timeline_start_x >= timeline_end_x) {
    return false;
  }

  const ImVec2 header_min(timeline_start_x, tracks_start_screen_pos.y +
                                                group_offsets_[group_index]);
  const ImVec2 header_max(timeline_end_x, tracks_start_screen_pos.y +
                                              group_offsets_[group_index] +
                                              kProcessTrackHeight);

  if (!ImGui::IsMouseHoveringRect(header_min, header_max)) {
    return false;
  }

  // Check for mouse click release. Ignore if Ctrl/Super is held (used for
  // bookmarks).
  if (ImGui::IsMouseReleased(0) && !ImGui::GetIO().KeyCtrl &&
      !ImGui::GetIO().KeySuper) {
    bool is_click = true;
    if (selection_start_pos_) {
      const float dx = ImGui::GetIO().MousePos.x - selection_start_pos_->x;
      const float dy = ImGui::GetIO().MousePos.y - selection_start_pos_->y;
      const float distance_squared = dx * dx + dy * dy;
      if (distance_squared > kClickDistanceThresholdSquared) {
        is_click = false;
      }
    }
    if (is_click) {
      BeginExpandToggleMotion(group_index, group.expanded);
      group.expanded = !group.expanded;
      event_clicked_this_frame_ = true;
      return true;
    }
  }

  return false;
}

EventRect Timeline::CalculateEventRect(
    Microseconds start, Microseconds end, Pixel screen_x_offset,
    Pixel screen_y_offset, double px_per_time_unit, int level_in_group,
    Pixel timeline_width, Pixel event_height, Pixel padding_bottom) const {
  const Pixel left = TimeToScreenX(start, screen_x_offset, px_per_time_unit);
  Pixel right = TimeToScreenX(end, screen_x_offset, px_per_time_unit);

  // Ensure minimum width for visibility.
  right = std::max(right, left + kEventMinimumDrawWidth);
  // Add a small gap to the right of the event for visual separation.
  // This is done here instead of in the Draw function to ensure the gap is
  // visible even if the event name overflows the right edge of the event. We
  // only adjust `right` to ensure the `left` boundary accurately reflects the
  // event's start time.
  right -= kEventPaddingRight;

  const Pixel top =
      screen_y_offset + level_in_group * (event_height + padding_bottom);
  const Pixel bottom = top + event_height;

  const Pixel timeline_right_boundary = screen_x_offset + timeline_width;

  // If the event ends before the visible area starts, return a zero-width
  // rectangle at the left boundary.
  if (right < screen_x_offset) {
    return {screen_x_offset, top, screen_x_offset, bottom};
  }
  // If the event starts after the visible area ends, return a zero-width
  // rectangle at the right boundary.
  if (left > timeline_right_boundary) {
    return {timeline_right_boundary, top, timeline_right_boundary, bottom};
  }

  // Clip the event rectangle to the visible window bounds.
  const Pixel clipped_left = std::max(left, screen_x_offset);
  const Pixel clipped_right = std::min(right, timeline_right_boundary);

  return {clipped_left, top, clipped_right, bottom};
}

ImVec2 Timeline::CalculateEventTextRect(absl::string_view event_name,
                                        const EventRect& event_rect) const {
  const ImVec2 text_size = GetTextSize(event_name);

  // Center the text within the clipped visible portion of the event.
  const Pixel clipped_width = event_rect.right - event_rect.left;
  const Pixel text_x = event_rect.left + (clipped_width - text_size.x) * 0.5f;
  const Pixel event_height = event_rect.bottom - event_rect.top;
  const Pixel text_y = event_rect.top + (event_height - text_size.y) * 0.5f;

  // Ensure the text starts at least at the left boundary of the event rect.
  // ImGui's PushClipRect in DrawEventName will handle the right boundary
  // clipping.
  const Pixel text_x_clipped = std::max(text_x, event_rect.left);

  return ImVec2(text_x_clipped, text_y);
}

std::string Timeline::GetTextForDisplay(absl::string_view event_name,
                                        Pixel available_width) const {
  const Pixel text_width = GetTextSize(event_name).x;

  if (text_width > available_width) {
    // Truncate text with "..." at the end
    const Pixel ellipsis_width = GetTextSize("...").x;
    if (available_width <= ellipsis_width) {
      return "";
    }

    // Binary search for the longest prefix that fits within the available
    // width.
    int low = 0, high = event_name.length(), fit_len = 0;
    while (low <= high) {
      const int mid = low + (high - low) / 2;
      if (GetTextSize(absl::string_view(event_name.data(), mid)).x +
              ellipsis_width <=
          available_width) {
        fit_len = mid;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }

    if (fit_len == 0) {
      return "";
    }

    return absl::StrCat(event_name.substr(0, fit_len), "...");
  }
  return std::string(event_name);
}

Microseconds Timeline::PixelToTime(Pixel pixel_offset,
                                   double px_per_time_unit) const {
  if (px_per_time_unit <= 0) return visible_range().start();
  return visible_range().start() +
         (static_cast<Microseconds>(pixel_offset) / px_per_time_unit);
}

Pixel Timeline::TimeToPixel(Microseconds time, double px_per_time_unit) const {
  if (px_per_time_unit <= 0) return 0;
  return static_cast<Pixel>((time - visible_range_->start()) *
                            px_per_time_unit);
}

Pixel Timeline::TimeToScreenX(Microseconds time, Pixel screen_x_offset,
                              double px_per_time_unit) const {
  return screen_x_offset + TimeToPixel(time, px_per_time_unit);
}

void Timeline::ConstrainTimeRange(TimeRange& range) {
  if (range.duration() < kMinDurationMicros) {
    double center = range.center();
    range = {center - kMinDurationMicros / 2.0,
             center + kMinDurationMicros / 2.0};
  }
  if (range.start() < data_time_range_.start()) {
    // When shifting the start to data_time_range_.start(), ensure the
    // new end does not exceed data_time_range_.end().
    range = {data_time_range_.start(),
             std::min(range.end() + data_time_range_.start() - range.start(),
                      data_time_range_.end())};
  } else if (range.end() > data_time_range_.end()) {
    // When shifting the end to data_time_range_.end(), ensure the new
    // start does not go before data_time_range_.start() by taking the
    // maximum.
    range = {std::max(range.start() - range.end() + data_time_range_.end(),
                      data_time_range_.start()),
             data_time_range_.end()};
  }
}

absl::string_view Timeline::GetEventName(int event_index) const {
  return GetEventNameFromData(timeline_data_, event_index);
}

EventData Timeline::CreateBaseEventData(int event_index, bool is_hover) const {
  EventData event_data;
  event_data.try_emplace(kEventSelectedIndex, event_index);
  if (event_index < 0 ||
      (static_cast<size_t>(event_index) >= timeline_data_.entry_names.size() &&
       static_cast<size_t>(event_index) >=
           timeline_data_.entry_name_indices.size())) {
    return event_data;
  }
  event_data.try_emplace(kEventSelectedName,
                         std::string(GetEventName(event_index)));
  if (static_cast<size_t>(event_index) <
      timeline_data_.entry_start_times.size()) {
    const Microseconds start_ts = timeline_data_.entry_start_times[event_index];
    event_data.try_emplace(kEventSelectedStart, start_ts);
    event_data.try_emplace(
        kEventSelectedStartFormatted,
        time_axis_unit_ == TimeAxisUnit::kUnitless
            ? FormatRulerLabel(start_ts - data_time_range_.start())
            : FormatTime(start_ts));
  }
  if (static_cast<size_t>(event_index) <
      timeline_data_.entry_total_times.size()) {
    const Microseconds total_ts = timeline_data_.entry_total_times[event_index];
    event_data.try_emplace(kEventSelectedDuration, total_ts);
    event_data.try_emplace(kEventSelectedDurationFormatted,
                           FormatDuration(total_ts, time_axis_unit_));
  }
  if (static_cast<size_t>(event_index) < timeline_data_.entry_levels.size()) {
    ProcessId pid = 0;
    if (const Group* grp = timeline_data_.FindGroupForLevel(
            timeline_data_.entry_levels[event_index])) {
      pid = grp->pid;
    }
    event_data.try_emplace(kEventSelectedPid, static_cast<double>(pid));
  }
  if (timeline_data_.HasEntryArgs(event_index)) {
    const absl::flat_hash_map<std::string, std::string> args =
        timeline_data_.GetEntryArgs(event_index);
    if (const auto it = args.find("uid");
        it != args.end() && it->second != "0") {
      event_data.try_emplace(kEventSelectedUid, it->second);
    }
    if (const auto it = args.find(kHloModule); it != args.end()) {
      event_data.try_emplace(kEventSelectedHloModuleName, it->second);
    }
    if (const auto it = args.find(kHloOp); it != args.end()) {
      event_data.try_emplace(kEventSelectedHloOpName, it->second);
    }
    if (!is_hover) {
      EventData js_args;
      js_args.reserve(args.size());
      for (const auto& [key, value] : args) {
        js_args.try_emplace(key, value);
      }
      event_data.try_emplace(kEventSelectedArgs, std::move(js_args));
    }
  }
  return event_data;
}

void Timeline::EmitEventSelected(int event_index) {
  if (!event_callback_) return;
  EventData event_data = CreateBaseEventData(event_index, /*is_hover=*/false);
  if (!event_tooltip_enabled_) {
    PopulateEventSelectedScheduleDetails(event_index, event_data);
  }
  event_callback_(kEventSelected, event_data);
}

void Timeline::SetSelectedEventDependencies(
    std::vector<std::string> operand_names,
    std::vector<std::string> consumer_names) {
  if (!hlo_dependency_arrows_enabled_) return;
  dependencies_ = ResolveEventDependencies(
      timeline_data_, selected_event_index_, std::move(operand_names),
      std::move(consumer_names));
  // The names usually arrive after the frame that selected the event.
  if (redraw_callback_) redraw_callback_();
}

void Timeline::EmitEventHovered(int event_index, float mouse_x, float mouse_y) {
  if (!event_callback_) return;
  EventData event_data = CreateBaseEventData(event_index, /*is_hover=*/true);
  event_data.try_emplace("mouse_x", static_cast<double>(mouse_x));
  event_data.try_emplace("mouse_y", static_cast<double>(mouse_y));

  if (event_index >= 0 &&
      (static_cast<size_t>(event_index) < timeline_data_.entry_names.size() ||
       static_cast<size_t>(event_index) <
           timeline_data_.entry_name_indices.size()) &&
      static_cast<size_t>(event_index) < timeline_data_.entry_levels.size()) {
    // Find track name (group name)
    int level = timeline_data_.entry_levels[event_index];
    int group_index = -1;
    for (size_t i = 0; i < timeline_data_.groups.size(); ++i) {
      int next_group_start_level = GetNextGroupStartLevel(timeline_data_, i);
      if (level >= timeline_data_.groups[i].start_level &&
          level < next_group_start_level) {
        group_index = i;
        break;
      }
    }
    if (group_index != -1) {
      event_data.try_emplace("trackName",
                             timeline_data_.groups[group_index].name);
    }
  }
  if (!event_tooltip_enabled_) {
    PopulateEventHoveredScheduleDetails(event_index, event_data);
  }
  event_callback_(kEventHovered, event_data);
}

void Timeline::EmitViewportChanged(const TimeRange& range) {
  EventData range_obj;
  range_obj[kViewportChangedMin] = range.start();
  range_obj[kViewportChangedMax] = range.end();
  EventData detail_obj;
  detail_obj[kViewportChangedRange] = range_obj;
  event_callback_(kViewportChanged, detail_obj);
}

void Timeline::EmitMouseModeChanged() {
  if (!event_callback_) return;
  EventData event_data;
  event_data.try_emplace(kMouseModeKey, static_cast<int>(mouse_mode_));
  event_callback_(kMouseModeChanged, event_data);
}

void Timeline::ShowNavigationWarningNotification(absl::string_view message) {
  bounds_notification_message_ = std::string(message);
  bounds_notification_timer_ = 2.0f;
  if (redraw_callback_) redraw_callback_();
}

void Timeline::RevealEvent(int event_index) {
  if (event_index < 0 ||
      event_index >= timeline_data_.entry_start_times.size() ||
      event_index >= timeline_data_.entry_total_times.size()) {
    LOG(ERROR) << "Invalid event index: " << event_index;
    return;
  }

  selected_event_index_ = event_index;
  event_index_to_scroll_to_ = event_index;
  ExpandRelatedTracks(event_index);

  const Microseconds start = timeline_data_.entry_start_times[event_index];
  Microseconds event_duration = timeline_data_.entry_total_times[event_index];
  // Marker entries or zero duration events.
  if (std::isnan(event_duration) || event_duration <= 0) {
    event_duration = kMinVisibleEventDuration;
  }
  const Microseconds end = start + event_duration;
  const Microseconds time_left = visible_range().start();
  const Microseconds time_right = visible_range().end();
  const Microseconds current_duration = visible_range().duration();

  Microseconds min_entry_time_window =
      std::min(event_duration, current_duration);

  // Ensure at least 30 pixels are visible.
  double min_visible_width_px = 30.0;
  double px_per_time = px_per_time_unit();
  if (px_per_time > 0) {
    double time_per_px = 1.0 / px_per_time;
    min_entry_time_window =
        std::max(min_entry_time_window, time_per_px * min_visible_width_px);
  }

  if (time_left > end) {
    double delta = time_left - end + min_entry_time_window;
    SetVisibleRange({time_left - delta, time_right - delta}, /*animate=*/true);
  } else if (time_right < start) {
    double delta = start - time_right + min_entry_time_window;
    SetVisibleRange({time_left + delta, time_right + delta}, /*animate=*/true);
  }

  EmitEventSelected(event_index);
}

void Timeline::ZoomEvent(int event_index) {
  if (event_index < 0 ||
      event_index >= timeline_data_.entry_start_times.size() ||
      event_index >= timeline_data_.entry_total_times.size()) {
    LOG(ERROR) << "Invalid event index: " << event_index;
    return;
  }

  selected_event_index_ = event_index;
  event_index_to_scroll_to_ = event_index;
  ExpandRelatedTracks(event_index);

  const Microseconds start = timeline_data_.entry_start_times[event_index];
  const Microseconds event_duration =
      timeline_data_.entry_total_times[event_index];
  // When navigating to an event, set the visible duration to 2.5 times the
  // event's duration to provide context around the event. Clamp the
  // duration between 10us and 5s to prevent zooming in too far on
  // short events or zooming out too far on long events.
  const Microseconds duration =
      std::max(kEventNavigationMinDurationMicros,
               std::min(event_duration * kEventNavigationZoomFactor,
                        kEventNavigationMaxDurationMicros));
  const Microseconds center = start + event_duration / 2.0;
  TimeRange new_range = {center - duration / 2.0, center + duration / 2.0};
  ConstrainTimeRange(new_range);

  SetVisibleRange(new_range, /*animate=*/true);

  EmitEventSelected(event_index);
}

void Timeline::SelectPreviousEvent() {
  if (timeline_data_.entry_levels.empty() ||
      timeline_data_.total_levels() == 0 || selected_event_index_ < 0 ||
      selected_event_index_ >= timeline_data_.entry_levels.size()) {
    return;
  }

  int level = timeline_data_.entry_levels[selected_event_index_];
  if (level >= timeline_data_.total_levels()) {
    return;
  }

  const auto events_on_level = timeline_data_.level_events(level);
  auto it = std::find(events_on_level.begin(), events_on_level.end(),
                      selected_event_index_);
  if (it != events_on_level.end()) {
    while (it != events_on_level.begin()) {
      --it;
      if (IsEventVisibleUnderFilter(*it)) {
        RevealEvent(*it);
        break;
      }
    }
  }
}

void Timeline::SelectNextEvent() {
  if (timeline_data_.entry_levels.empty() ||
      timeline_data_.total_levels() == 0 || selected_event_index_ < 0 ||
      selected_event_index_ >= timeline_data_.entry_levels.size()) {
    return;
  }

  int level = timeline_data_.entry_levels[selected_event_index_];
  if (level >= timeline_data_.total_levels()) {
    return;
  }

  const auto events_on_level = timeline_data_.level_events(level);
  auto it = std::find(events_on_level.begin(), events_on_level.end(),
                      selected_event_index_);
  if (it != events_on_level.end()) {
    for (++it; it != events_on_level.end(); ++it) {
      if (IsEventVisibleUnderFilter(*it)) {
        RevealEvent(*it);
        break;
      }
    }
  }
}

// Computes the bounding time range of the active selection, handling either
// multiple selection (via selected_event_indices_) or single selection
// (via selected_event_index_). Returns std::nullopt if no event is selected.
std::optional<TimeRange> Timeline::GetSelectionTimeRange() const {
  if (!selected_event_indices_.empty()) {
    Microseconds min_start = std::numeric_limits<Microseconds>::infinity();
    Microseconds max_end = -std::numeric_limits<Microseconds>::infinity();

    selected_event_indices_.ForEachSetBit([&](int idx) {
      if (idx >= 0 &&
          static_cast<size_t>(idx) < timeline_data_.entry_start_times.size() &&
          static_cast<size_t>(idx) < timeline_data_.entry_total_times.size()) {
        const Microseconds start = timeline_data_.entry_start_times[idx];
        Microseconds duration = timeline_data_.entry_total_times[idx];
        if (std::isnan(duration) || duration <= 0) {
          duration = kMinVisibleEventDuration;
        }
        min_start = std::min(min_start, start);
        max_end = std::max(max_end, start + duration);
      }
    });
    if (min_start <= max_end) {
      return TimeRange(min_start, max_end);
    }
  } else if (selected_event_index_ != -1 &&
             selected_event_index_ < timeline_data_.entry_start_times.size() &&
             selected_event_index_ < timeline_data_.entry_total_times.size()) {
    const Microseconds start =
        timeline_data_.entry_start_times[selected_event_index_];
    Microseconds duration =
        timeline_data_.entry_total_times[selected_event_index_];
    if (std::isnan(duration) || duration <= 0) {
      duration = kMinVisibleEventDuration;
    }
    return TimeRange(start, start + duration);
  }
  return std::nullopt;
}

void Timeline::ExpandRelatedTracks(int event_index) {
  int level = timeline_data_.entry_levels[event_index];
  int group_index = -1;
  for (size_t i = 0; i < timeline_data_.groups.size(); ++i) {
    int next_group_start_level = GetNextGroupStartLevel(timeline_data_, i);
    if (level >= timeline_data_.groups[i].start_level &&
        level < next_group_start_level) {
      group_index = i;
      break;
    }
  }

  if (group_index != -1) {
    bool changed = false;
    if (!timeline_data_.groups[group_index].expanded) {
      timeline_data_.groups[group_index].expanded = true;
      changed = true;
    }
    int current_nesting = timeline_data_.groups[group_index].nesting_level;
    for (int i = group_index - 1; i >= 0 && current_nesting > 0; --i) {
      if (timeline_data_.groups[i].nesting_level < current_nesting) {
        if (!timeline_data_.groups[i].expanded) {
          timeline_data_.groups[i].expanded = true;
          changed = true;
        }
        current_nesting = timeline_data_.groups[i].nesting_level;
      }
    }
    if (changed) {
      UpdateLevelPositions(timeline_data_);
      if (redraw_callback_) redraw_callback_();
    }
  }
}

void Timeline::HideTrack(absl::string_view name) {
  hidden_track_names_.insert(std::string(name));
  UpdateLevelPositions(timeline_data_);
  if (redraw_callback_) redraw_callback_();
}

void Timeline::CalculateBezierControlPoints(float start_x, float start_y,
                                            float end_x, float end_y,
                                            ImVec2& cp0, ImVec2& cp1) {
  const Pixel dist = std::abs(end_x - start_x) * 0.5f;
  cp0 = ImVec2(start_x + dist, start_y);
  cp1 = ImVec2(end_x - dist, end_y);
}

void Timeline::Pan(Pixel pixel_amount) {
  // If the pixel amount is 0, we don't need to pan.
  if (pixel_amount == 0.0) return;

  const double px_per_time_unit_val = px_per_time_unit();
  // This should never happen, but we check it to avoid division by zero.
  if (px_per_time_unit_val <= 0.0) return;

  const double time_offset = pixel_amount / px_per_time_unit_val;
  TimeRange new_range = visible_range_.target() + time_offset;

  const bool showing_entire_trace =
      visible_range_.target().start() <= data_time_range_.start() &&
      visible_range_.target().end() >= data_time_range_.end();

  if (!showing_entire_trace) {
    if (pixel_amount < 0.0 && new_range.start() < data_time_range_.start()) {
      ShowNavigationWarningNotification(
          "Cannot pan further left: reached the beginning of the trace.");
    } else if (pixel_amount > 0.0 && new_range.end() > data_time_range_.end()) {
      ShowNavigationWarningNotification(
          "Cannot pan further right: reached the end of the trace.");
    }
  }

  ConstrainTimeRange(new_range);

  // Update the target of the animated visible range. The timeline will animate
  // towards this new time.
  SetVisibleRange(new_range, /*animate=*/true);
  EmitViewportChanged(new_range);
}

void Timeline::Scroll(Pixel pixel_amount) {
  // If the pixel amount is 0, we don't need to scroll.
  if (pixel_amount == 0.0) return;

  ImGui::SetScrollY(ImGui::GetScrollY() + pixel_amount);
}

void Timeline::Zoom(float zoom_factor) {
  ImGuiContext* context = ImGui::GetCurrentContext();
  // Use the latest mouse position as the zoom pivot if an ImGui window is
  // active and the mouse is within the timeline area.
  // We use IsMouseHoveringRect solely to check if the mouse position is within
  // the timeline bounds, as we don't rely on the window's hover state.
  // However, if the mouse is outside the timeline area (or outside an active
  // window context, such as when triggered via external toolbar actions or unit
  // tests), we fallback to using the current visible range center.
  if (context != nullptr && context->CurrentWindow != nullptr) {
    const ImRect timeline_area = GetTimelineArea();
    if (ImGui::IsMouseHoveringRect(timeline_area.Min, timeline_area.Max)) {
      const double px_per_time = px_per_time_unit();
      const Microseconds pivot = PixelToTime(
          ImGui::GetMousePos().x - timeline_area.Min.x, px_per_time);

      Zoom(zoom_factor, pivot);
      return;
    }
  }

  Zoom(zoom_factor, visible_range_->center());
}

void Timeline::Zoom(float zoom_factor, Microseconds pivot) {
  // If the zoom factor is 1, we don't need to zoom.
  if (zoom_factor == 1.0) return;

  // Clamp the zoom factor to the minimum value. This prevents the time
  // durations (mathmatically) become zero or negative.
  zoom_factor = std::max(zoom_factor, kMinZoomFactor);

  TimeRange new_range = visible_range_.target();
  new_range.Zoom(zoom_factor, pivot);

  if (zoom_factor < 1.0) {
    if (new_range.duration() < kMinDurationMicros) {
      ShowNavigationWarningNotification(
          "Cannot zoom in further: minimum zoom duration reached.");
    }
  } else if (zoom_factor > 1.0) {
    TimeRange constrained_range = new_range;
    ConstrainTimeRange(constrained_range);
    if (constrained_range.duration() < new_range.duration() ||
        (visible_range_.target().start() <= data_time_range_.start() &&
         visible_range_.target().end() >= data_time_range_.end())) {
      ShowNavigationWarningNotification(
          "Cannot zoom out further: showing the entire trace.");
    }
  }

  ConstrainTimeRange(new_range);

  // Update the target of the animated visible range. The timeline will animate
  // towards this new zoom level.
  SetVisibleRange(new_range, /*animate=*/true);
  EmitViewportChanged(new_range);
}

void Timeline::ApplySnapping(TimeRange& range) {
  // Snapping only applies when in measuring mode (kTiming) or when Shift+Drag
  // is used to select a time range.
  if (mouse_mode_ != MouseMode::kTiming && !ImGui::GetIO().KeyShift &&
      !time_range_resizing_state_.has_value()) {
    return;
  }
  const double px_per_time = px_per_time_unit();
  if (px_per_time <= 0) return;

  const Microseconds threshold = 16.0 / px_per_time;
  const Microseconds original_duration = visible_range_.target().duration();
  const bool is_pan = std::abs(range.duration() - original_duration) < 1e-6;

  Microseconds best_diff_start = threshold;
  Microseconds best_diff_end = threshold;
  Microseconds snapped_start_time = range.start();
  Microseconds snapped_end_time = range.end();
  bool snapped_start = false;
  bool snapped_end = false;

  bool is_resizing = time_range_resizing_state_.has_value();
  bool resize_start = is_resizing && time_range_resizing_state_->is_start_edge;
  bool resize_end = is_resizing && !time_range_resizing_state_->is_start_edge;

  // 1. Check selected time ranges (ONLY if we are NOT resizing!)
  if (!is_resizing) {
    for (const auto& sel_range : selected_time_ranges_) {
      if (!is_resizing || resize_start) {
        ApplySnappingToEdge(range.start(), threshold,
                            {sel_range.start(), sel_range.end()},
                            best_diff_start, snapped_start_time, snapped_start);
      }
      if (!is_resizing || resize_end) {
        ApplySnappingToEdge(range.end(), threshold,
                            {sel_range.start(), sel_range.end()}, best_diff_end,
                            snapped_end_time, snapped_end);
      }
    }
  }

  // 2. Check all visible events
  if (!is_resizing || resize_start) {
    FindNearestEventEdge(range.start(), threshold, best_diff_start,
                         snapped_start_time, snapped_start);
  }
  if (!is_resizing || resize_end) {
    FindNearestEventEdge(range.end(), threshold, best_diff_end,
                         snapped_end_time, snapped_end);
  }

  if (is_pan) {
    if (snapped_start) {
      range = {snapped_start_time, snapped_start_time + original_duration};
    } else if (snapped_end) {
      range = {snapped_end_time - original_duration, snapped_end_time};
    }
  } else {
    Microseconds final_start =
        snapped_start ? snapped_start_time : range.start();
    Microseconds final_end = snapped_end ? snapped_end_time : range.end();
    range = TimeRange(std::min(final_start, final_end),
                      std::max(final_start, final_end));
  }
}

void Timeline::FindNearestEventEdge(Microseconds time, Microseconds threshold,
                                    Microseconds& best_diff,
                                    Microseconds& snapped_time,
                                    bool& snapped) const {
  const double px_per_time = px_per_time_unit();
  if (px_per_time <= 0) return;

  Pixel current_scroll_y = ImGui::GetScrollY();
  Pixel window_height = ImGui::GetWindowHeight();
  ImVec2 mouse_pos = ImGui::GetMousePos();

  for (int group_index = 0; group_index < timeline_data_.groups.size();
       ++group_index) {
    if (group_index >= group_visible_.size() || !group_visible_[group_index]) {
      continue;
    }

    // Determine the bottom edge of this group by finding the next visible group
    Pixel group_bottom_offset = group_offsets_.back();
    for (int next_visible_group = group_index + 1;
         next_visible_group < group_visible_.size(); ++next_visible_group) {
      if (group_visible_[next_visible_group]) {
        group_bottom_offset = group_offsets_[next_visible_group];
        break;
      }
    }

    Pixel group_top_y =
        tracks_start_screen_pos_.y + group_offsets_[group_index];
    Pixel group_bottom_y = tracks_start_screen_pos_.y + group_bottom_offset;

    bool is_group_hovered =
        mouse_pos.y >= group_top_y && mouse_pos.y <= group_bottom_y;

    // If we are actively resizing, the mouse might drift vertically out of the
    // original track bounding box, but we STILL ONLY want to snap to the
    // hovered track. If the mouse vertically leaves the track but the user is
    // still dragging horizontally, they don't want snapping to suddenly jump to
    // the adjacent track their mouse fell into. However, since we don't save
    // the original track, finding the vertically hovered one is the best we can
    // do.
    if (!is_group_hovered) {
      continue;
    }

    const Group& group = timeline_data_.groups[group_index];
    int next_group_start_level =
        GetNextGroupStartLevel(timeline_data_, group_index);

    const bool has_children =
        group_index + 1 < timeline_data_.groups.size() &&
        timeline_data_.groups[group_index + 1].nesting_level >
            group.nesting_level;
    const bool has_multiple_levels =
        next_group_start_level - group.start_level > 1;
    const bool expandable = group.type == Group::Type::kFlame &&
                            (has_children || has_multiple_levels);
    const bool is_collapsed = expandable && !group.expanded;

    if (is_collapsed) {
      continue;
    }

    const int total_levels = timeline_data_.total_levels();
    for (int level = group.start_level; level < next_group_start_level;
         ++level) {
      if (level >= total_levels || level >= visible_level_offsets_.size()) {
        continue;
      }

      Pixel y_center = visible_level_offsets_[level];
      Pixel y_top = y_center - kEventHeight * 0.5f;
      Pixel y_bottom = y_center + kEventHeight * 0.5f;

      if (y_bottom < current_scroll_y ||
          y_top > current_scroll_y + window_height) {
        continue;
      }

      absl::Span<const int> indices = timeline_data_.level_events(level);
      if (indices.empty()) continue;

      auto it = std::lower_bound(
          indices.begin(), indices.end(), time - threshold,
          [this](int event_index, Microseconds t) {
            return timeline_data_.entry_start_times[event_index] +
                       timeline_data_.entry_total_times[event_index] <
                   t;
          });

      for (; it != indices.end(); ++it) {
        const int event_index = *it;
        if (event_index < 0 ||
            event_index >= timeline_data_.entry_start_times.size() ||
            event_index >= timeline_data_.entry_total_times.size() ||
            !IsEventVisibleUnderFilter(event_index)) {
          continue;
        }

        const Microseconds start =
            timeline_data_.entry_start_times[event_index];
        if (start > time + threshold) {
          break;
        }

        const Microseconds duration =
            timeline_data_.entry_total_times[event_index];
        if (duration * px_per_time >= 2.0) {
          ApplySnappingToEdge(time, threshold, {start, start + duration},
                              best_diff, snapped_time, snapped);
        }
      }
    }
  }
}

double Timeline::px_per_time_unit() const {
  return px_per_time_unit(current_timeline_width_);
}

double Timeline::px_per_time_unit(Pixel timeline_width) const {
  const Microseconds view_duration = visible_range_->duration();
  if (view_duration > 0 && timeline_width > 0) {
    return static_cast<double>(timeline_width) / view_duration;
  } else {
    return 0.0;
  }
}

// Calculates the timing and pixel spacing for timeline ticks.
// This function determines a "nice" interval (e.g., 1, 2, 5, 10...) such that
// ticks are neither too crowded nor too sparse on screen.
Timeline::TickInfo Timeline::CalculateTickInfo(
    double px_per_time_unit_val) const {
  const Microseconds min_time_interval =
      kMinTickDistancePx / px_per_time_unit_val;
  Microseconds tick_interval = CalculateNiceInterval(min_time_interval);
  const bool unitless = time_axis_unit_ == TimeAxisUnit::kUnitless;
  if (unitless) {
    // A unitless axis (e.g. scheduled bundle numbers) is discrete, so major
    // ticks are never placed at fractional positions.
    tick_interval = std::max(tick_interval, 1.0);
  }
  const Pixel major_tick_dist_px = tick_interval * px_per_time_unit_val;

  const Microseconds view_start = visible_range().start();
  const Microseconds trace_start = data_time_range_.start();

  if (unitless) {
    // Align ticks to absolute positions (rather than positions relative to the
    // trace start) so that ruler labels match the positions reported for
    // events, even if the trace does not start at 0.
    const Microseconds first_tick_time =
        std::floor(view_start / tick_interval) * tick_interval;
    return {tick_interval, major_tick_dist_px, first_tick_time - trace_start};
  }

  const Microseconds view_start_relative = view_start - trace_start;
  const Microseconds first_tick_time_relative =
      std::floor(view_start_relative / tick_interval) * tick_interval;

  return {tick_interval, major_tick_dist_px, first_tick_time_relative};
}

// Renders the ruler UI element at the top of the timeline.
// This is drawn as a table row and includes the background, the main horizontal
// line, major/minor tick marks, and time labels.
void Timeline::DrawRulerUI(const TickInfo& info, Pixel timeline_width) {
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const ImU32 ruler_text_color =
      palette_.GetColor(ColorPalette::Key::kRulerText)
          .value_or(kRulerTextColor);
  const ImU32 ruler_line_color =
      palette_.GetColor(ColorPalette::Key::kRulerLine)
          .value_or(kRulerLineColor);

  ImGui::SetCursorScreenPos(ImVec2(pos.x + label_width_, pos.y));

  ImDrawList* const draw_list = ImGui::GetWindowDrawList();

  const double px_per_time_unit_val = px_per_time_unit(timeline_width);
  if (px_per_time_unit_val > 0) {
    // Draw horizontal line
    const Pixel line_y = pos.y + kRulerHeight;
    draw_list->AddLine(
        ImVec2(pos.x + label_width_, line_y),
        ImVec2(pos.x + label_width_ + timeline_width + kTimelinePaddingRight,
               line_y),
        ruler_line_color);

    const Microseconds tick_interval = info.tick_interval;
    const Pixel major_tick_dist_px = info.major_tick_dist_px;
    const Microseconds first_tick_time_relative = info.first_tick_time_relative;
    const Microseconds trace_start = data_time_range_.start();

    const Pixel minor_tick_dist_px =
        major_tick_dist_px / static_cast<float>(kMinorTickDivisions);
    // On a unitless (discrete) axis, only draw minor ticks when they fall on
    // whole-number positions.
    const Microseconds minor_tick_interval =
        tick_interval / kMinorTickDivisions;
    const bool draw_minor_ticks =
        time_axis_unit_ != TimeAxisUnit::kUnitless ||
        (minor_tick_interval >= 1.0 &&
         std::floor(minor_tick_interval) == minor_tick_interval);

    Microseconds t_relative = first_tick_time_relative;
    Pixel x = TimeToScreenX(t_relative + trace_start, pos.x + label_width_,
                            px_per_time_unit_val);

    for (;; t_relative += tick_interval, x += major_tick_dist_px) {
      if (x > pos.x + label_width_ + timeline_width + kRulerScreenBuffer) {
        break;
      }

      if (x >= pos.x + label_width_ - kRulerScreenBuffer) {
        // Draw major tick marks on the ruler.
        draw_list->AddLine(ImVec2(x, pos.y), ImVec2(x, line_y),
                           ruler_line_color);

        const std::string time_label_text = FormatRulerLabel(t_relative);
        ImGui::PushFont(fonts::label_small);
        draw_list->AddText(ImVec2(x + kRulerTextPadding, pos.y),
                           ruler_text_color, time_label_text.c_str());
        ImGui::PopFont();
      }

      if (!draw_minor_ticks) continue;

      // Draw minor ticks for the current interval.
      for (int i = 1; i < kMinorTickDivisions; ++i) {
        const Pixel minor_x = x + i * minor_tick_dist_px;
        if (minor_x >
            pos.x + label_width_ + timeline_width + kRulerScreenBuffer) {
          break;
        }
        if (minor_x >= pos.x + label_width_ - kRulerScreenBuffer) {
          draw_list->AddLine(ImVec2(minor_x, line_y - kRulerMinorTickHeight),
                             ImVec2(minor_x, line_y), ruler_line_color);
        }
      }
    }
  }
}

std::string Timeline::FormatRulerLabel(Microseconds time_relative) const {
  switch (time_axis_unit_) {
    case TimeAxisUnit::kUnitless:
      // Show the absolute position as a plain integer (e.g. a bundle number),
      // without any time unit.
      return absl::StrCat(
          std::llround(time_relative + data_time_range_.start()));
    case TimeAxisUnit::kTime:
      break;
  }
  return FormatTime(time_relative);
}

// Draws vertical grid lines that extend from the ruler down across all tracks.
// These lines are typically drawn behind the tracks in the background layer
// to provide a visual time reference without obscuring track content.
void Timeline::DrawVerticalGridLines(const TickInfo& info, Pixel timeline_width,
                                     Pixel viewport_bottom) {
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  ImDrawList* const draw_list = ImGui::GetWindowDrawList();

  const double px_per_time_unit_val = px_per_time_unit(timeline_width);
  if (px_per_time_unit_val <= 0) return;

  const Pixel timeline_x_start = pos.x + label_width_;
  const Pixel line_y_top = pos.y;

  const Microseconds tick_interval = info.tick_interval;
  const Pixel major_tick_dist_px = info.major_tick_dist_px;
  const Microseconds first_tick_time_relative = info.first_tick_time_relative;

  const Microseconds trace_start = data_time_range_.start();

  Microseconds t_relative = first_tick_time_relative;
  Pixel x = TimeToScreenX(t_relative + trace_start, timeline_x_start,
                          px_per_time_unit_val);

  for (;; t_relative += tick_interval, x += major_tick_dist_px) {
    if (x > timeline_x_start + timeline_width + kRulerScreenBuffer) {
      break;
    }

    if (x >= timeline_x_start - kRulerScreenBuffer) {
      // Draw vertical line across the tracks.
      draw_list->AddLine(ImVec2(x, line_y_top), ImVec2(x, viewport_bottom),
                         palette_.GetColor(ColorPalette::Key::kMidtone)
                             .value_or(kTraceVerticalLineColor));
    }
  }
}

void Timeline::DrawEventName(absl::string_view event_name,
                             const EventRect& event_rect,
                             ImDrawList* absl_nonnull draw_list,
                             ImU32 text_color) const {
  // Cull text if the event overlaps with the ruler.
  if (event_rect.top < ruler_screen_y_ + kRulerHeight) {
    return;
  }

  const Pixel available_width = event_rect.right - event_rect.left;

  if (available_width >= kMinTextWidth) {
    const std::string text_display =
        GetTextForDisplay(event_name, available_width);

    if (!text_display.empty()) {
      const ImVec2 text_pos = CalculateEventTextRect(text_display, event_rect);

      // Push a clipping rectangle to ensure the text is only drawn within the
      // bounds of the event_rect. This prevents text from overflowing visually.
      draw_list->PushClipRect(ImVec2(event_rect.left, event_rect.top),
                              ImVec2(event_rect.right, event_rect.bottom));
      draw_list->AddText(text_pos, text_color, text_display.c_str());
      draw_list->PopClipRect();
    }
  }
}

void Timeline::DrawEventTextOnly(int event_index, const EventRect& rect,
                                 ImDrawList* absl_nonnull draw_list) const {
  if (rect.right <= rect.left || event_index < 0 ||
      static_cast<size_t>(event_index) >= GetEntryCount()) {
    return;
  }
  const ImU32 event_color = GetEventColor(event_index);
  const ImU32 on_surface_color =
      palette_.GetColor(ColorPalette::Key::kOnSurface)
          .value_or(kOnSurfaceColor);
  const ImU32 inverse_on_surface_color =
      palette_.GetColor(ColorPalette::Key::kInverseOnSurface)
          .value_or(kInverseOnSurfaceColor);
  const ImU32 text_color = GetTextColorForContrast(
      event_color, on_surface_color, inverse_on_surface_color);
  DrawEventName(GetEventName(event_index), rect, draw_list,
                text_color);
}

void Timeline::DrawEvent(int group_index, int event_index,
                         const EventRect& rect,
                         ImDrawList* absl_nonnull draw_list) {
  // Only draw the rectangle if it has a positive width after clipping.
  // TODO: b/453676716 - Add ImGUI test for this function, including condition
  // rect.right > rect.left.
  if (rect.right > rect.left) {
    const absl::string_view event_name = GetEventName(event_index);

    const bool is_instant =
        timeline_data_.entry_total_times[event_index] <= 1e-06;
    bool is_hovered = false;
    // As the shapes for instant events are triangles, we need to check for
    // hover in a different way than regular events.
    if (is_instant) {
      // Even if the instant event is triangular, to solve the "fat finger"
      // problem, we still want to check for hover in a rectangle around the
      // triangle.
      is_hovered = ImGui::IsMouseHoveringRect(
          ImVec2(rect.left - kInstantEventChevronHalfWidth, rect.top),
          ImVec2(rect.left + kInstantEventChevronHalfWidth,
                 rect.top + kInstantEventChevronHeight));
    } else {
      is_hovered = ImGui::IsMouseHoveringRect(
          ImVec2(rect.left, rect.top),
          ImVec2(rect.right + kEventPaddingRight,
                 rect.bottom + kEventPaddingBottom));
    }

    const Pixel corner_rounding =
        is_hovered ? kHoverCornerRounding : kCornerRounding;

    ImU32 event_color = GetEventColor(event_index);

    bool matches_search = false;
    if (!search_query_lower_.empty()) {
      matches_search = matching_event_indices_.contains(event_index);
      if (!matches_search) {
        // Gray out non-matching events with muted grayscale and lower opacity
        const uint32_t r = (event_color >> IM_COL32_R_SHIFT) & 0xFF;
        const uint32_t g = (event_color >> IM_COL32_G_SHIFT) & 0xFF;
        const uint32_t b = (event_color >> IM_COL32_B_SHIFT) & 0xFF;
        const uint32_t luminance = (r * 299 + g * 587 + b * 114) / 1000;
        event_color = IM_COL32(luminance, luminance, luminance, 102);
      }
    }

    ImVec2 top, left_bottom, right_bottom;
    if (is_instant) {
      const Pixel centerX = rect.left;
      const Pixel chevron_half_width = is_hovered
                                           ? kInstantEventHoverChevronHalfWidth
                                           : kInstantEventChevronHalfWidth;
      const Pixel chevron_height = is_hovered ? kInstantEventHoverChevronHeight
                                              : kInstantEventChevronHeight;

      top = ImVec2(centerX, rect.top);
      left_bottom =
          ImVec2(centerX - chevron_half_width, rect.top + chevron_height);
      right_bottom =
          ImVec2(centerX + chevron_half_width, rect.top + chevron_height);

      // Add transparency (0.6 opacity) to the event color of instant events
      // to help distinguish overlapping ones. Use full opacity when hovered.
      const ImU32 transparent_color =
          (event_color & ~IM_COL32_A_MASK) |
          (static_cast<ImU32>(0.6f * 255.0f) << IM_COL32_A_SHIFT);
      const ImU32 color =
          (is_hovered || matches_search) ? event_color : transparent_color;

      draw_list->AddTriangleFilled(top, left_bottom, right_bottom, color);
    } else {
      draw_list->AddRectFilled(ImVec2(rect.left, rect.top),
                               ImVec2(rect.right, rect.bottom), event_color,
                               corner_rounding, kImDrawFlags);
    }
    if (is_hovered) {
      hovered_event_index_ = event_index;
      // Draw a semi-transparent overlay when the event is hovered.
      if (is_instant) {
        draw_list->AddTriangleFilled(top, left_bottom, right_bottom,
                                     kHoverMaskColor);
        draw_list->AddTriangle(top, left_bottom, right_bottom, event_color,
                               2.0f);
      } else {
        draw_list->AddRectFilled(
            ImVec2(rect.left, rect.top), ImVec2(rect.right, rect.bottom),
            kHoverMaskColor, corner_rounding, kImDrawFlags);
      }

      if (event_tooltip_enabled_) {
        ImGui::SetTooltip(
            "%.*s (%s)", static_cast<int>(event_name.size()), event_name.data(),
            FormatDuration(timeline_data_.entry_total_times[event_index],
                           time_axis_unit_)
                .c_str());
      }

      // ImGui uses 0 to represent the left mouse button, as defined in the
      // ImGuiMouseButton enum. We check if the left mouse button was clicked.
      if (ImGui::IsMouseReleased(0)) {
        bool is_click = true;
        if (selection_start_pos_) {
          const float dx = ImGui::GetIO().MousePos.x - selection_start_pos_->x;
          const float dy = ImGui::GetIO().MousePos.y - selection_start_pos_->y;
          const float distance_squared = dx * dx + dy * dy;
          if (distance_squared > kClickDistanceThresholdSquared) {
            is_click = false;
          }
        }
        if (is_click) {
          event_clicked_this_frame_ = true;

          // If shift is held down, select/deselect the time range of the event.
          if (ImGui::GetIO().KeyShift) {
            const Microseconds start =
                timeline_data_.entry_start_times[event_index];
            const Microseconds end =
                start + timeline_data_.entry_total_times[event_index];
            TimeRange selected_time_range(start, end);
            auto it = absl::c_find(selected_time_ranges_, selected_time_range);
            // Click on the event to select, and click on the same event to
            // de-select.
            if (it != selected_time_ranges_.end()) {
              selected_time_ranges_.erase(it);
            } else {
              selected_time_ranges_.push_back(selected_time_range);
            }
          }

          const bool had_multi_selection = !selected_event_indices_.empty() ||
                                           !selected_counter_points_.empty();
          selected_event_indices_.clear();
          selected_counter_points_.clear();

          if (selected_event_index_ != event_index || had_multi_selection) {
            selected_group_index_ = group_index;
            selected_event_index_ = event_index;
            // Deselect any selected counter event.
            selected_counter_index_ = -1;
            selected_counter_bundle_ = -1;

            EmitEventSelected(event_index);
          }
        }  // close if (is_click)
      }  // close if (IsMouseReleased)
    }

    if (selected_event_index_ == event_index) {
      if (is_instant) {
        const Pixel centerX = rect.left;
        const Pixel chevron_half_width =
            is_hovered ? kInstantEventHoverChevronHalfWidth
                       : kInstantEventChevronHalfWidth;
        const Pixel chevron_height = is_hovered
                                         ? kInstantEventHoverChevronHeight
                                         : kInstantEventChevronHeight;

        ImVec2 top(centerX, rect.top);
        ImVec2 left_bottom(centerX - chevron_half_width,
                           rect.top + chevron_height);
        ImVec2 right_bottom(centerX + chevron_half_width,
                            rect.top + chevron_height);

        draw_list->AddTriangle(top, left_bottom, right_bottom,
                               kSelectedBorderColor, kSelectedBorderThickness);
      } else {
        // Draw a border around the selected event.
        draw_list->AddRect(ImVec2(rect.left, rect.top),
                           ImVec2(rect.right, rect.bottom),
                           kSelectedBorderColor, corner_rounding, kImDrawFlags,
                           kSelectedBorderThickness);
      }
    }

    const ImU32 on_surface_color =
        palette_.GetColor(ColorPalette::Key::kOnSurface)
            .value_or(kOnSurfaceColor);
    const ImU32 inverse_on_surface_color =
        palette_.GetColor(ColorPalette::Key::kInverseOnSurface)
            .value_or(kInverseOnSurfaceColor);
    const ImU32 text_color = GetTextColorForContrast(
        event_color, on_surface_color, inverse_on_surface_color);
    DrawEventName(event_name, rect, draw_list, text_color);
  }
}

void Timeline::DrawEventsForLevel(int group_index,
                                  absl::Span<const int> event_indices,
                                  double px_per_time_unit, int level_in_group,
                                  const ImVec2& pos, const ImVec2& max,
                                  Pixel event_height, Pixel padding_bottom) {
  ImDrawList* const draw_list = ImGui::GetWindowDrawList();
  if (!draw_list) {
    return;
  }

  const Microseconds visible_start_time = PixelToTime(0, px_per_time_unit);
  const Microseconds visible_end_time = PixelToTime(max.x, px_per_time_unit);

  auto first_visible_it = std::lower_bound(
      event_indices.begin(), event_indices.end(), visible_start_time,
      [&](int event_index, Microseconds time) {
        // Defensive bounds check for event_index. While event_indices is
        // expected to contain valid indices, this prevents crashes if an
        // invalid index somehow gets included.
        if (event_index < 0 ||
            event_index >= timeline_data_.entry_start_times.size() ||
            event_index >= timeline_data_.entry_total_times.size()) {
          return false;  // Treat out-of-bounds as not ending before 'time'.
        }
        return timeline_data_.entry_start_times[event_index] +
                   timeline_data_.entry_total_times[event_index] <
               time;
      });

  auto last_visible_it = std::upper_bound(
      first_visible_it, event_indices.end(), visible_end_time,
      [&](Microseconds time, int event_index) {
        // Defensive bounds check for event_index.
        if (event_index < 0 ||
            event_index >= timeline_data_.entry_start_times.size()) {
          return false;  // Treat out-of-bounds as not starting after 'time'.
        }
        return time < timeline_data_.entry_start_times[event_index];
      });

  const size_t visible_count =
      static_cast<size_t>(std::distance(first_visible_it, last_visible_it));
  if (search_query_lower_.empty() && px_per_time_unit > 0.0 &&
      !event_indices.empty()) {
    int global_level = -1;
    if (group_index >= 0 &&
        static_cast<size_t>(group_index) < timeline_data_.groups.size()) {
      const int candidate =
          timeline_data_.groups[group_index].start_level + level_in_group;
      if (candidate >= 0 && candidate < timeline_data_.total_levels() &&
          timeline_data_.level_events(candidate).data() ==
              event_indices.data() &&
          timeline_data_.level_events(candidate).size() ==
              event_indices.size()) {
        global_level = candidate;
      }
    }
    if (global_level < 0) {
      const int sample_idx = event_indices.front();
      if (sample_idx >= 0 &&
          static_cast<size_t>(sample_idx) <
              timeline_data_.entry_levels.size()) {
        global_level = timeline_data_.entry_levels[sample_idx];
      }
    }
    const bool has_level =
        global_level >= 0 &&
        static_cast<size_t>(global_level) < level_mip_pyramids_.size();
    if (has_level) {
      EnsureLevelMipPyramid(global_level);
    }
    if (has_level && level_mip_pyramids_[global_level].has_pyramid &&
        (level_mip_pyramids_[global_level].raw_events_compacted ||
         visible_count > kMinVisibleEventsForMipDraw)) {
      const LevelMipPyramid& pyramid = level_mip_pyramids_[global_level];
      const Microseconds us_per_px = 1.0 / px_per_time_unit;
      int selected_mip =
          pyramid.SelectMipLevel(kMaxMipBinWidthPx * us_per_px);
      if (selected_mip < 0 && pyramid.raw_events_compacted) {
        for (int m = 0; m < LevelMipPyramid::kNumMipLevels; ++m) {
          if (!pyramid.levels[m].tiles.empty()) {
            selected_mip = m;
            break;
          }
        }
      }
      if (selected_mip >= 0) {
        const auto& tiles = pyramid.levels[selected_mip].tiles;
        auto tile_first =
            std::lower_bound(tiles.begin(), tiles.end(), visible_start_time,
                             [](const LevelMipTile& tile, Microseconds time) {
                               return tile.max_end < time;
                             });
        auto tile_last =
            std::upper_bound(tile_first, tiles.end(), visible_end_time,
                             [](Microseconds time, const LevelMipTile& tile) {
                               return time < tile.min_start;
                             });
        const size_t visible_tiles =
            static_cast<size_t>(std::distance(tile_first, tile_last));
        if (pyramid.raw_events_compacted || visible_tiles * 2 < visible_count) {
          const Pixel y_top =
              pos.y + level_in_group * (event_height + padding_bottom);
          const Pixel y_bottom = y_top + event_height;
          const ImVec2 mouse_pos = ImGui::GetMousePos();
          const bool row_hovered_y =
              mouse_pos.y >= y_top && mouse_pos.y < y_bottom + padding_bottom;

          if (visible_tiles > 0 && HasGpuFlameInstanceBuffer()) {
            const uint32_t first_inst =
                pyramid.levels[selected_mip].gpu_instance_offset +
                static_cast<uint32_t>(std::distance(tiles.begin(), tile_first));
            AddGpuFlameBatch(
                draw_list,
                GpuFlameBatch{
                    .first_instance = first_inst,
                    .instance_count = static_cast<uint32_t>(visible_tiles),
                    .visible_start_us = visible_start_time,
                    .px_per_us = static_cast<float>(px_per_time_unit),
                    .screen_x_offset = pos.x,
                    .timeline_width = max.x,
                    .y_top = y_top,
                    .y_bottom = y_bottom,
                    .min_width_px = kEventMinimumDrawWidth,
                    .padding_right_px = 0.0f,
                    .alpha_multiplier = 1.0f,
                });
            int hovered_dom_idx = -1;
            Pixel hovered_tile_left = 0.0f;
            Pixel hovered_tile_right = 0.0f;
            for (auto it = tile_first; it != tile_last; ++it) {
              const int dom_idx = it->dominant_event_index;
              if (dom_idx < 0 || static_cast<size_t>(dom_idx) >=
                                     timeline_data_.entry_start_times.size()) {
                continue;
              }
              if (row_hovered_y) {
                Pixel left =
                    TimeToScreenX(it->min_start, pos.x, px_per_time_unit);
                Pixel right =
                    TimeToScreenX(it->max_end, pos.x, px_per_time_unit);
                if (right < left) std::swap(left, right);
                right = std::max(right, left + kEventMinimumDrawWidth);
                left = std::max(left, pos.x);
                right = std::min(right, pos.x + max.x);
                if (right > left && mouse_pos.x >= left &&
                    mouse_pos.x <= right) {
                  hovered_dom_idx = dom_idx;
                  hovered_tile_left = left;
                  hovered_tile_right = right;
                }
              }
              const Microseconds dom_start =
                  timeline_data_.entry_start_times[dom_idx];
              const Microseconds dom_dur =
                  timeline_data_.entry_total_times[dom_idx];
              const Microseconds dom_end = dom_start + dom_dur;
              if (dom_idx != selected_event_index_ &&
                  (dom_end - dom_start) * px_per_time_unit >= kMinTextWidth &&
                  hovered_dom_idx != dom_idx) {
                const EventRect rect = CalculateEventRect(
                    dom_start, dom_end, pos.x, pos.y, px_per_time_unit,
                    level_in_group, max.x, event_height, padding_bottom);
                DrawEventTextOnly(dom_idx, rect, draw_list);
              }
            }
            if (selected_event_index_ >= 0 &&
                static_cast<size_t>(selected_event_index_) <
                    timeline_data_.entry_levels.size() &&
                timeline_data_.entry_levels[selected_event_index_] ==
                    global_level &&
                static_cast<size_t>(selected_event_index_) <
                    timeline_data_.entry_start_times.size() &&
                static_cast<size_t>(selected_event_index_) <
                    timeline_data_.entry_total_times.size()) {
              const Microseconds sel_start =
                  timeline_data_.entry_start_times[selected_event_index_];
              const Microseconds sel_end =
                  sel_start +
                  timeline_data_.entry_total_times[selected_event_index_];
              if (sel_end >= visible_start_time &&
                  sel_start <= visible_end_time) {
                EventRect sel_rect = CalculateEventRect(
                    sel_start, sel_end, pos.x, pos.y, px_per_time_unit,
                    level_in_group, max.x, event_height, padding_bottom);
                if (selected_event_index_ == hovered_dom_idx) {
                  sel_rect.left = std::min(sel_rect.left, hovered_tile_left);
                  sel_rect.right = std::max(sel_rect.right, hovered_tile_right);
                }
                DrawEvent(group_index, selected_event_index_, sel_rect,
                          draw_list);
              }
            }
            if (hovered_dom_idx >= 0 &&
                hovered_dom_idx != selected_event_index_) {
              const Microseconds s =
                  timeline_data_.entry_start_times[hovered_dom_idx];
              const Microseconds e =
                  s + timeline_data_.entry_total_times[hovered_dom_idx];
              EventRect rect = CalculateEventRect(
                  s, e, pos.x, pos.y, px_per_time_unit, level_in_group, max.x,
                  event_height, padding_bottom);
              rect.left = std::min(rect.left, hovered_tile_left);
              rect.right = std::max(rect.right, hovered_tile_right);
              DrawEvent(group_index, hovered_dom_idx, rect, draw_list);
            }
            return;
          }

          Pixel last_mip_right = -std::numeric_limits<Pixel>::infinity();
          int hovered_dom_idx = -1;
          Pixel hovered_tile_left = 0.0f;
          Pixel hovered_tile_right = 0.0f;
          for (auto it = tile_first; it != tile_last; ++it) {
            const int dom_idx = it->dominant_event_index;
            if (dom_idx < 0 || static_cast<size_t>(dom_idx) >=
                                   timeline_data_.entry_start_times.size()) {
              continue;
            }
            const Microseconds dom_start =
                timeline_data_.entry_start_times[dom_idx];
            const Microseconds dom_dur =
                timeline_data_.entry_total_times[dom_idx];
            const Microseconds dom_end = dom_start + dom_dur;

            if (dom_idx != selected_event_index_ &&
                (it->count == 1 ||
                 (dom_end - dom_start) * px_per_time_unit >= kMinTextWidth)) {
              const EventRect rect = CalculateEventRect(
                  dom_start, dom_end, pos.x, pos.y, px_per_time_unit,
                  level_in_group, max.x, event_height, padding_bottom);
              if (rect.right > rect.left) {
                last_mip_right = std::max(last_mip_right, rect.right);
              }
              DrawEvent(group_index, dom_idx, rect, draw_list);
              continue;
            }

            Pixel left = TimeToScreenX(it->min_start, pos.x, px_per_time_unit);
            Pixel right = TimeToScreenX(it->max_end, pos.x, px_per_time_unit);
            if (right < left) std::swap(left, right);
            right = std::max(right, left + kEventMinimumDrawWidth);
            if (left < last_mip_right && right <= last_mip_right + 0.5f) {
              continue;
            }
            left = std::max(left, pos.x);
            right = std::min(right, pos.x + max.x);
            if (right <= left) continue;

            if (row_hovered_y && mouse_pos.x >= left && mouse_pos.x <= right) {
              hovered_dom_idx = dom_idx;
              hovered_tile_left = left;
              hovered_tile_right = right;
            }

            last_mip_right = std::max(last_mip_right, right);
            const ImU32 color = (it->blended_color != 0)
                                    ? it->blended_color
                                    : GetEventColor(dom_idx);
            draw_list->AddRectFilled(ImVec2(left, y_top),
                                     ImVec2(right, y_bottom), color,
                                     kCornerRounding, kImDrawFlags);
          }

          if (selected_event_index_ >= 0 &&
              IsEventVisibleUnderFilter(selected_event_index_) &&
              static_cast<size_t>(selected_event_index_) <
                  timeline_data_.entry_levels.size() &&
              timeline_data_.entry_levels[selected_event_index_] ==
                  global_level &&
              static_cast<size_t>(selected_event_index_) <
                  timeline_data_.entry_start_times.size() &&
              static_cast<size_t>(selected_event_index_) <
                  timeline_data_.entry_total_times.size()) {
            const Microseconds sel_start =
                timeline_data_.entry_start_times[selected_event_index_];
            const Microseconds sel_end =
                sel_start +
                timeline_data_.entry_total_times[selected_event_index_];
            if (sel_end >= visible_start_time &&
                sel_start <= visible_end_time) {
              EventRect sel_rect = CalculateEventRect(
                  sel_start, sel_end, pos.x, pos.y, px_per_time_unit,
                  level_in_group, max.x, event_height, padding_bottom);
              if (selected_event_index_ == hovered_dom_idx) {
                sel_rect.left = std::min(sel_rect.left, hovered_tile_left);
                sel_rect.right = std::max(sel_rect.right, hovered_tile_right);
              }
              DrawEvent(group_index, selected_event_index_, sel_rect,
                        draw_list);
            }
          }

          if (hovered_dom_idx >= 0 &&
              hovered_dom_idx != selected_event_index_) {
            const Microseconds s =
                timeline_data_.entry_start_times[hovered_dom_idx];
            const Microseconds e =
                s + timeline_data_.entry_total_times[hovered_dom_idx];
            EventRect rect = CalculateEventRect(
                s, e, pos.x, pos.y, px_per_time_unit, level_in_group, max.x,
                event_height, padding_bottom);
            rect.left = std::min(rect.left, hovered_tile_left);
            rect.right = std::max(rect.right, hovered_tile_right);
            DrawEvent(group_index, hovered_dom_idx, rect, draw_list);
          }
          return;
        }
      }
    }
  }

  if (search_query_lower_.empty() && visible_count > 0 &&
      px_per_time_unit > 0.0 && !event_indices.empty() &&
      HasGpuFlameInstanceBuffer()) {
    const int sample_idx = event_indices.front();
    const int global_level =
        (sample_idx >= 0 &&
         static_cast<size_t>(sample_idx) < timeline_data_.entry_levels.size())
            ? timeline_data_.entry_levels[sample_idx]
            : -1;
    if (global_level >= 0 &&
        static_cast<size_t>(global_level) < level_mip_pyramids_.size()) {
      const LevelMipPyramid& pyramid = level_mip_pyramids_[global_level];
      const absl::Span<const int> full_lvl_events =
          timeline_data_.level_events(global_level);
      if (pyramid.raw_gpu_instance_offset != UINT32_MAX &&
          event_indices.data() == full_lvl_events.data() &&
          event_indices.size() == full_lvl_events.size()) {
        const Pixel y_top =
            pos.y + level_in_group * (event_height + padding_bottom);
        const Pixel y_bottom = y_top + event_height;
        const uint32_t first_inst =
            pyramid.raw_gpu_instance_offset +
            static_cast<uint32_t>(
                std::distance(event_indices.begin(), first_visible_it));
        AddGpuFlameBatch(
            draw_list,
            GpuFlameBatch{
                .first_instance = first_inst,
                .instance_count = static_cast<uint32_t>(visible_count),
                .visible_start_us = visible_start_time,
                .px_per_us = static_cast<float>(px_per_time_unit),
                .screen_x_offset = pos.x,
                .timeline_width = max.x,
                .y_top = y_top,
                .y_bottom = y_bottom,
                .min_width_px = kEventMinimumDrawWidth,
                .padding_right_px = kEventPaddingRight,
                .alpha_multiplier = 1.0f,
            });
        const ImVec2 mouse_pos = ImGui::GetMousePos();
        const bool row_hovered_y =
            mouse_pos.y >= y_top && mouse_pos.y <= y_bottom;
        int hovered_idx = -1;
        for (auto it = first_visible_it; it != last_visible_it; ++it) {
          const int event_index = *it;
          if (event_index < 0 || static_cast<size_t>(event_index) >=
                                     timeline_data_.entry_start_times.size()) {
            continue;
          }
          const Microseconds start =
              timeline_data_.entry_start_times[event_index];
          const Microseconds dur =
              timeline_data_.entry_total_times[event_index];
          const Microseconds end = start + dur;
          const bool is_instant = dur <= 1e-06;
          if (is_instant || event_index == selected_event_index_) {
            const EventRect rect = CalculateEventRect(
                start, end, pos.x, pos.y, px_per_time_unit, level_in_group,
                max.x, event_height, padding_bottom);
            DrawEvent(group_index, event_index, rect, draw_list);
            continue;
          }
          if (row_hovered_y) {
            Pixel left = TimeToScreenX(start, pos.x, px_per_time_unit);
            Pixel right = TimeToScreenX(end, pos.x, px_per_time_unit);
            if (right < left) std::swap(left, right);
            right = std::max(right, left + kEventMinimumDrawWidth);
            if (mouse_pos.x >= left && mouse_pos.x <= right) {
              hovered_idx = event_index;
            }
          }
          if (dur * px_per_time_unit >= kMinTextWidth &&
              hovered_idx != event_index) {
            const EventRect rect = CalculateEventRect(
                start, end, pos.x, pos.y, px_per_time_unit, level_in_group,
                max.x, event_height, padding_bottom);
            DrawEventTextOnly(event_index, rect, draw_list);
          }
        }
        if (hovered_idx >= 0 && hovered_idx != selected_event_index_) {
          const Microseconds s = timeline_data_.entry_start_times[hovered_idx];
          const Microseconds e =
              s + timeline_data_.entry_total_times[hovered_idx];
          const EventRect rect = CalculateEventRect(
              s, e, pos.x, pos.y, px_per_time_unit, level_in_group, max.x,
              event_height, padding_bottom);
          DrawEvent(group_index, hovered_idx, rect, draw_list);
        }
        return;
      }
    }
  }

  Pixel last_drawn_right = -std::numeric_limits<Pixel>::infinity();
  for (auto it = first_visible_it; it != last_visible_it; ++it) {
    int event_index = *it;
    if (event_index < 0 ||
        event_index >= timeline_data_.entry_start_times.size() ||
        event_index >= timeline_data_.entry_total_times.size() ||
        !IsEventVisibleUnderFilter(event_index)) {
      // Should not happen if data is well-formed, but good to be safe.
      continue;
    }
    const Microseconds start = timeline_data_.entry_start_times[event_index];
    const Microseconds dur = timeline_data_.entry_total_times[event_index];
    const Microseconds end = start + dur;

    const bool is_instant = dur <= 1e-06;
    const bool is_selected = (selected_event_index_ == event_index);
    const bool matches_search =
        !search_query_lower_.empty() &&
        matching_event_indices_.contains(event_index);

    if (!is_instant && !is_selected && !matches_search) {
      const Pixel raw_left = TimeToScreenX(start, pos.x, px_per_time_unit);
      const Pixel raw_right = TimeToScreenX(end, pos.x, px_per_time_unit);
      if (raw_left < last_drawn_right &&
          raw_right <= last_drawn_right + 0.5f) {
        continue;
      }
    }

    const EventRect rect =
        CalculateEventRect(start, end, pos.x, pos.y, px_per_time_unit,
                           level_in_group, max.x, event_height, padding_bottom);

    if (!is_instant && rect.right > rect.left) {
      last_drawn_right = std::max(last_drawn_right, rect.right);
    }

    DrawEvent(group_index, event_index, rect, draw_list);
  }
}

void Timeline::DrawCounterTooltip(int group_index, const CounterData& data,
                                  double px_per_time_unit_val,
                                  const ImVec2& pos, Pixel height,
                                  float y_ratio, ImDrawList* draw_list) {
  const ImVec2 mouse_pos = ImGui::GetMousePos();
  const double mouse_time =
      PixelToTime(mouse_pos.x - pos.x, px_per_time_unit_val);

  // Find the interval [t_i, t_{i+1}) containing mouse_time for sample-and-hold
  // (step) interpolation.
  // We use upper_bound to find the first timestamp strictly greater than
  // mouse_time. This ensures that std::prev(it) always points to t_i (the
  // start of the interval), even if mouse_time exactly equals t_i.
  // Using lower_bound would be incorrect for exact matches, as it would return
  // t_i, causing std::prev(it) to point to t_{i-1}.
  auto it = std::upper_bound(data.timestamps.begin(), data.timestamps.end(),
                             mouse_time);

  // Ensure we are not before the first timestamp.
  if (it != data.timestamps.begin()) {
    size_t index = std::distance(data.timestamps.begin(), std::prev(it));
    double val = data.values[index];

    const Pixel y_base = pos.y + height;

    if (index + 1 < data.timestamps.size()) {
      const double t1 = data.timestamps[index];
      const double t2 = data.timestamps[index + 1];
      const double v1 = data.values[index];

      // Sample-and-hold interpolation to match the bars drawn in
      // DrawCounterTrack.
      val = v1;

      // Highlight the entire bar under hover.
      const Pixel x1 = TimeToScreenX(t1, pos.x, px_per_time_unit_val);
      const Pixel x2 = TimeToScreenX(t2, pos.x, px_per_time_unit_val);
      const Pixel y = y_base - (v1 - data.min_value) * y_ratio;
      draw_list->AddRect(ImVec2(x1, y), ImVec2(x2, y_base), kCounterHoverColor,
                         0.0f, 0, kCounterHoverThickness);
    }

    const Pixel x = mouse_pos.x;
    const Pixel y = y_base - (val - data.min_value) * y_ratio;

    // Draw circle
    draw_list->AddCircleFilled(ImVec2(x, y), kPointRadius, kWhiteColor);
    draw_list->AddCircle(ImVec2(x, y), kPointRadius, kBlackColor);

    // Draw tooltip for current counter point's value and timestamp
    ImGui::SetTooltip(
        "%s", FormatCounterTooltip(time_axis_unit_, mouse_time, val, data.unit)
                  .c_str());
    if (!event_tooltip_enabled_ && hovered_event_index_ == -1 &&
        mouse_time >= 0.0) {
      hovered_group_index_ = group_index;
      hovered_bundle_ = static_cast<int64_t>(std::floor(mouse_time));
      hovered_counter_value_ = val;
    }

    // ImGui uses 0 to represent the left mouse button, as defined in the
    // ImGuiMouseButton enum. We check if the left mouse button was clicked.
    if (ImGui::IsMouseReleased(0)) {
      bool is_click = true;
      if (selection_start_pos_) {
        const float dx = ImGui::GetIO().MousePos.x - selection_start_pos_->x;
        const float dy = ImGui::GetIO().MousePos.y - selection_start_pos_->y;
        const float distance_squared = dx * dx + dy * dy;
        if (distance_squared > kClickDistanceThresholdSquared) {
          is_click = false;
        }
      }
      if (is_click) {
        event_clicked_this_frame_ = true;
        const int64_t clicked_bundle =
            static_cast<int64_t>(std::floor(std::max(0.0, mouse_time)));
        const bool had_multi_selection = !selected_event_indices_.empty() ||
                                         !selected_counter_points_.empty();
        selected_event_indices_.clear();
        selected_counter_points_.clear();
        if (selected_group_index_ != group_index ||
            selected_counter_index_ != static_cast<int>(index) ||
            (!event_tooltip_enabled_ &&
             selected_counter_bundle_ != clicked_bundle) ||
            had_multi_selection) {
          selected_group_index_ = group_index;
          selected_counter_index_ = index;
          selected_counter_bundle_ = clicked_bundle;
          // Deselect any selected flame event.
          selected_event_index_ = -1;

          // Emit an event to notify the application that a counter event was
          // selected.
          const std::string& name = timeline_data_.groups[group_index].name;
          EventData event_data;
          // We pass -1 for the event index to indicate that no flame event is
          // selected.
          event_data.try_emplace(kEventSelectedIndex, -1);
          event_data.try_emplace(kEventSelectedName, name);
          if (!event_tooltip_enabled_) {
            PopulateCounterSelectedScheduleDetails(
                group_index, index, static_cast<Microseconds>(clicked_bundle),
                event_data);
          }

          event_callback_(kEventSelected, event_data);
        }
      }
    }
  } else if (!event_tooltip_enabled_ && hovered_event_index_ == -1 &&
             mouse_time >= 0.0) {
    hovered_group_index_ = group_index;
    hovered_bundle_ = static_cast<int64_t>(std::floor(mouse_time));
    hovered_counter_value_ = 0.0;
  }
}

void Timeline::DrawCounterTrack(int group_index, const CounterData& data,
                                double px_per_time_unit_val, const ImVec2& pos,
                                Pixel height) {
  // At least two timestamps are required to draw a line segment.
  if (data.timestamps.size() < 2) return;

  ImDrawList* const draw_list = ImGui::GetWindowDrawList();

  if (!draw_list) return;
  const double value_range = data.max_value - data.min_value;

  // This should not happen with valid data where max_value >= min_value.
  if (value_range < 0) {
    LOG(ERROR) << "Invalid counter data: max_value " << data.max_value
               << " is less than min_value " << data.min_value;
    return;
  }

  const Pixel y_base = pos.y + height;

  // If all counter values are the same, draw a 1px thin rectangle
  // at the base to avoid division by zero.
  if (value_range == 0) {
    const Pixel y = y_base - 1.0f;
    const Pixel x_start =
        TimeToScreenX(data.timestamps.front(), pos.x, px_per_time_unit_val);
    const Pixel x_end =
        TimeToScreenX(data.timestamps.back(), pos.x, px_per_time_unit_val);
    draw_list->AddRectFilled(ImVec2(x_start, y), ImVec2(x_end, y_base),
                             kCounterTrackColor);
    return;
  }

  const float y_ratio = height / value_range;

  Pixel pending_x1 = 0.0f;
  Pixel pending_x2 = 0.0f;
  Pixel pending_min_y = y_base;
  bool has_pending = false;

  for (size_t i = 0; i < data.timestamps.size() - 1; ++i) {
    Pixel x1 = TimeToScreenX(data.timestamps[i], pos.x, px_per_time_unit_val);
    Pixel x2 =
        TimeToScreenX(data.timestamps[i + 1], pos.x, px_per_time_unit_val);
    Pixel y = y_base - (data.values[i] - data.min_value) * y_ratio;

    // Add a minimum 1px height so that a value equal to min_value is still
    // visible as a thin line instead of completely disappearing.
    y = std::min(y, y_base - 1.0f);

    if (x2 - x1 < 0.5f) {
      if (!has_pending) {
        pending_x1 = x1;
        pending_x2 = std::max(x2, x1 + 1.0f);
        pending_min_y = y;
        has_pending = true;
      } else if (x1 <= pending_x2) {
        pending_x2 = std::max(pending_x2, x2);
        pending_min_y = std::min(pending_min_y, y);
      } else {
        draw_list->AddRectFilled(ImVec2(pending_x1, pending_min_y),
                                 ImVec2(pending_x2, y_base),
                                 kCounterTrackColor);
        pending_x1 = x1;
        pending_x2 = std::max(x2, x1 + 1.0f);
        pending_min_y = y;
      }
      continue;
    }

    if (has_pending) {
      draw_list->AddRectFilled(ImVec2(pending_x1, pending_min_y),
                               ImVec2(pending_x2, y_base), kCounterTrackColor);
      has_pending = false;
    }

    draw_list->AddRectFilled(ImVec2(x1, y), ImVec2(x2, y_base),
                             kCounterTrackColor);
  }
  if (has_pending) {
    draw_list->AddRectFilled(ImVec2(pending_x1, pending_min_y),
                             ImVec2(pending_x2, y_base), kCounterTrackColor);
  }

  // For the last point, draw a 1px wide bar to show its value.
  if (!data.timestamps.empty()) {
    Pixel x =
        TimeToScreenX(data.timestamps.back(), pos.x, px_per_time_unit_val);
    Pixel y = y_base - (data.values.back() - data.min_value) * y_ratio;
    draw_list->AddRectFilled(ImVec2(x, y), ImVec2(x + 1.0f, y_base),
                             kCounterTrackColor);
  }

  // Draw selected points from rectangle selection.
  auto it_pair = std::equal_range(
      selected_counter_points_.begin(), selected_counter_points_.end(),
      std::make_pair(group_index, 0),
      [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first < b.first;
      });

  for (auto it = it_pair.first; it != it_pair.second; ++it) {
    size_t p_idx = it->second;
    if (p_idx < data.timestamps.size()) {
      if (selected_group_index_ == group_index &&
          selected_counter_index_ == p_idx) {
        continue;  // Handled by single selection below.
      }
      Microseconds ts = data.timestamps[p_idx];
      double val = data.values[p_idx];
      Pixel x = TimeToScreenX(ts, pos.x, px_per_time_unit_val);
      Pixel y = pos.y + height - (val - data.min_value) * y_ratio;

      draw_list->AddCircleFilled(ImVec2(x, y), kSelectedDataPointRadius,
                                 kBlue60);
      draw_list->AddCircle(ImVec2(x, y), kSelectedDataPointRadius, kWhiteColor,
                           /*num_segments=*/0,
                           /*thickness=*/kSelectedBorderThickness);
    }
  }

  if (selected_group_index_ == group_index && selected_counter_index_ != -1 &&
      selected_counter_index_ < data.timestamps.size()) {
    Microseconds ts = data.timestamps[selected_counter_index_];
    if (!event_tooltip_enabled_ && selected_counter_bundle_ >= 0) {
      const Microseconds next_ts =
          (static_cast<size_t>(selected_counter_index_ + 1) <
           data.timestamps.size())
              ? data.timestamps[selected_counter_index_ + 1]
              : GetScheduleEndBundle();
      const Microseconds bundle_ts =
          static_cast<Microseconds>(selected_counter_bundle_);
      if (bundle_ts >= ts && bundle_ts < next_ts) {
        ts = bundle_ts;
      }
    }
    double val = data.values[selected_counter_index_];
    Pixel x = TimeToScreenX(ts, pos.x, px_per_time_unit_val);
    Pixel y = pos.y + height - (val - data.min_value) * y_ratio;

    draw_list->AddCircleFilled(ImVec2(x, y), kPointRadius, kWhiteColor);
    draw_list->AddCircle(ImVec2(x, y), kPointRadius, kSelectedBorderColor,
                         /*num_segments=*/0,
                         /*thickness=*/kSelectedBorderThickness);
  }

  if (ImGui::IsWindowHovered()) {
    DrawCounterTooltip(group_index, data, px_per_time_unit_val, pos, height,
                       y_ratio, draw_list);
  }
}

void Timeline::DrawGroup(int group_index, double px_per_time_unit_val,
                         Pixel scroll_y, Pixel window_height) {
  const Group& group = timeline_data_.groups[group_index];
  const int start_level = group.start_level;
  int end_level = GetNextGroupStartLevel(timeline_data_, group_index);
  if (group.type == Group::Type::kFlame && !group.expanded) {
    end_level = start_level;
  }
  // Ensure end_level is not less than start_level, to avoid negative height.
  end_level = std::max(start_level, end_level);

  // Calculate group height. Ensure a minimum height of one level to prevent
  // ImGui::BeginChild from auto-resizing, even if a group contains no levels.
  // This is important for parent groups (e.g., a process) that might not
  // contain any event levels directly.
  // TODO: b/453676716 - Add tests for group height calculation.
  const Pixel group_height =
      group.type == Group::Type::kCounter
          ? kCounterTrackHeight
          : (group.nesting_level == kProcessNestingLevel
                 ? kProcessTrackHeight
                 : std::max(1, end_level - start_level) *
                       (kEventHeight + kEventPaddingBottom));
  // Groups might have the same name. We add the index of the group to the ID
  // to ensure each ImGui::BeginChild call has a unique ID, otherwise ImGui
  // might ignore later calls with the same name.
  const std::string timeline_child_id =
      absl::StrCat("TimelineChild_", group.name, "_", group_index);

  const ImVec2 pos = ImGui::GetCursorScreenPos();

  // During a layout transition the row may be partially open; the child window
  // then clips the events like a drawer. A zero height would make ImGui fill
  // the remaining space, so fully folded rows are skipped.
  const Pixel drawn_height = AnimatedRowHeight(group_index, group_height);
  if (drawn_height < 1.0f) return;

  if (ImGui::BeginChild(timeline_child_id.c_str(), ImVec2(0, drawn_height), 0,
                        kTrackFlags)) {
    const ImVec2 max = ImGui::GetContentRegionMax();

    if (group.type == Group::Type::kCounter) {
      const auto it =
          timeline_data_.counter_data_by_group_index.find(group_index);
      if (it != timeline_data_.counter_data_by_group_index.end()) {
        DrawCounterTrack(group_index, it->second, px_per_time_unit_val, pos,
                         group_height);
      }
    } else if (group.type == Group::Type::kFlame) {
      if (group.nesting_level == kProcessNestingLevel) {
        ImDrawList* const draw_list = ImGui::GetWindowDrawList();
        if (draw_list) {
          // Find the next group that is NOT a child of the current group to
          // determine the end level for the utilization chart.
          int proc_end_level = timeline_data_.total_levels();
          for (size_t i = group_index + 1; i < timeline_data_.groups.size();
               ++i) {
            if (timeline_data_.groups[i].nesting_level <= group.nesting_level) {
              proc_end_level = timeline_data_.groups[i].start_level;
              break;
            }
          }
          DrawUtilizationAreaChart(group_index, start_level, proc_end_level,
                                   px_per_time_unit_val, pos, group_height,
                                   draw_list);
        }
      }
      const Pixel level_stride = kEventHeight + kEventPaddingBottom;
      const Pixel group_offset = group_offsets_[group_index];

      int first_visible_level = start_level;
      Pixel relative_scroll_y = scroll_y - group_offset;
      if (relative_scroll_y > 0) {
        first_visible_level =
            start_level + std::floor(relative_scroll_y / level_stride);
      }

      int last_visible_level = end_level;
      Pixel relative_scroll_end = scroll_y + window_height - group_offset;
      if (relative_scroll_end > 0) {
        last_visible_level =
            start_level + std::ceil(relative_scroll_end / level_stride);
      }

      first_visible_level = std::max(start_level, first_visible_level);
      last_visible_level = std::min(
          {end_level, last_visible_level, timeline_data_.total_levels()});

      for (int level = first_visible_level; level < last_visible_level;
           ++level) {
        // TODO: b/453676716 - Add boundary test cases for this function.
        absl::Span<const int> indices = timeline_data_.level_events(level);
        DrawEventsForLevel(group_index, indices, px_per_time_unit_val,
                           /*level_in_group=*/level - start_level, pos, max,
                           kEventHeight, kEventPaddingBottom);
      }
    }
  }
  ImGui::EndChild();
}

void Timeline::DrawGroupPreview(int group_index, double px_per_time_unit_val) {
  const Group& group = timeline_data_.groups[group_index];
  const std::string timeline_child_id =
      absl::StrCat("TimelineChildPreview_", group.name, "_", group_index);

  // Process tracks have a fixed height, other tracks use a single event height
  // for the preview.
  const Pixel group_height = group.nesting_level == kProcessNestingLevel
                                 ? kProcessTrackHeight
                                 : kEventHeight;

  // Calculate level Y positions for the preview.
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const int start_level = group.start_level;
  int end_level = timeline_data_.total_levels();
  // Find the next group that is NOT a child of the current group.
  for (size_t i = group_index + 1; i < timeline_data_.groups.size(); ++i) {
    if (timeline_data_.groups[i].nesting_level <= group.nesting_level) {
      end_level = timeline_data_.groups[i].start_level;
      break;
    }
  }
  end_level = std::max(start_level, end_level);

  const Pixel drawn_height = AnimatedRowHeight(group_index, group_height);
  if (drawn_height < 1.0f) return;

  if (ImGui::BeginChild(timeline_child_id.c_str(), ImVec2(0, drawn_height), 0,
                        kTrackFlags)) {
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    if (group.type == Group::Type::kCounter) {
      const auto it =
          timeline_data_.counter_data_by_group_index.find(group_index);
      if (it != timeline_data_.counter_data_by_group_index.end()) {
        DrawCounterTrack(group_index, it->second, px_per_time_unit_val, pos,
                         group_height);
      }
    } else if (group.type == Group::Type::kFlame) {
      if (group.nesting_level == kProcessNestingLevel) {
        DrawUtilizationAreaChart(group_index, start_level, end_level,
                                 px_per_time_unit_val, pos, group_height,
                                 draw_list);
      } else {
        DrawFlameGroupPreview(start_level, end_level, px_per_time_unit_val, pos,
                              group_height, draw_list);
      }
    }
  }
  ImGui::EndChild();
}

void Timeline::DrawFlameGroupPreview(int start_level, int end_level,
                                     double px_per_time_unit_val,
                                     const ImVec2& pos, Pixel group_height,
                                     ImDrawList* draw_list) {
  // Aggregated view: Flatten all levels into one track with reduced
  // opacity.
  const Microseconds visible_start = visible_range().start();
  const Microseconds visible_end = visible_range().end();

  for (int level = start_level; level < end_level; ++level) {
    if (px_per_time_unit_val > 0.0 && level >= 0 &&
        static_cast<size_t>(level) < level_mip_pyramids_.size()) {
      EnsureLevelMipPyramid(level);
      const LevelMipPyramid& pyramid = level_mip_pyramids_[level];
      const Microseconds us_per_px = 1.0 / px_per_time_unit_val;
      const int selected_mip =
          pyramid.SelectMipLevel(kMaxPreviewMipBinWidthPx * us_per_px);
      if (selected_mip >= 0) {
        const auto& tiles = pyramid.levels[selected_mip].tiles;
        auto tile_first =
            std::lower_bound(tiles.begin(), tiles.end(), visible_start,
                             [](const LevelMipTile& tile, Microseconds t) {
                               return tile.max_end <= t;
                             });
        if (HasGpuFlameInstanceBuffer()) {
          auto tile_last =
              std::upper_bound(tile_first, tiles.end(), visible_end,
                               [](Microseconds t, const LevelMipTile& tile) {
                                 return t < tile.min_start;
                               });
          const uint32_t count =
              static_cast<uint32_t>(std::distance(tile_first, tile_last));
          if (count > 0) {
            const uint32_t first_inst =
                pyramid.levels[selected_mip].gpu_instance_offset +
                static_cast<uint32_t>(std::distance(tiles.begin(), tile_first));
            AddGpuFlameBatch(
                draw_list,
                GpuFlameBatch{
                    .first_instance = first_inst,
                    .instance_count = count,
                    .visible_start_us = visible_start,
                    .px_per_us = static_cast<float>(px_per_time_unit_val),
                    .screen_x_offset = pos.x,
                    .timeline_width = current_timeline_width_,
                    .y_top = pos.y,
                    .y_bottom = pos.y + group_height,
                    .min_width_px = kEventMinimumDrawWidth,
                    .padding_right_px = 0.0f,
                    .alpha_multiplier = kGroupPreviewOpacity,
                });
            continue;
          }
        }
        Pixel last_drawn_right = -std::numeric_limits<Pixel>::infinity();
        for (auto it = tile_first; it != tiles.end(); ++it) {
          if (it->min_start >= visible_end) break;
          Pixel x_start =
              TimeToScreenX(it->min_start, pos.x, px_per_time_unit_val);
          Pixel x_end = TimeToScreenX(it->max_end, pos.x, px_per_time_unit_val);
          if (x_end < x_start) std::swap(x_start, x_end);
          if (x_start < last_drawn_right && x_end <= last_drawn_right + 0.5f) {
            continue;
          }
          ImU32 color = (it->blended_color != 0)
                            ? it->blended_color
                            : GetEventColor(it->dominant_event_index);
          color = (color & ~IM_COL32_A_MASK) |
                  (static_cast<ImU32>(kGroupPreviewOpacity * 255.0f)
                   << IM_COL32_A_SHIFT);
          x_end = std::max(x_end, x_start + kEventMinimumDrawWidth);
          last_drawn_right = std::max(last_drawn_right, x_end);
          draw_list->AddRectFilled(ImVec2(x_start, pos.y),
                                   ImVec2(x_end, pos.y + group_height), color);
        }
        continue;
      }
    }

    absl::Span<const int> indices = timeline_data_.level_events(level);

    // Find the first event that ends after the visible start.
    // Since events in the same level are non-overlapping and sorted by
    // start time, they are effectively sorted by end time as well.
    auto it = std::lower_bound(
        indices.begin(), indices.end(), visible_start,
        [&](int event_idx, Microseconds t) {
          const Microseconds end = timeline_data_.entry_start_times[event_idx] +
                                   timeline_data_.entry_total_times[event_idx];
          return end <= t;
        });

    if (px_per_time_unit_val > 0.0 && level >= 0 &&
        static_cast<size_t>(level) < level_mip_pyramids_.size() &&
        level_mip_pyramids_[level].raw_gpu_instance_offset != UINT32_MAX &&
        HasGpuFlameInstanceBuffer()) {
      auto last_it = std::upper_bound(
          it, indices.end(), visible_end, [&](Microseconds t, int event_idx) {
            return t < timeline_data_.entry_start_times[event_idx];
          });
      const uint32_t count = static_cast<uint32_t>(std::distance(it, last_it));
      if (count > 0) {
        const uint32_t first_inst =
            level_mip_pyramids_[level].raw_gpu_instance_offset +
            static_cast<uint32_t>(std::distance(indices.begin(), it));
        AddGpuFlameBatch(
            draw_list,
            GpuFlameBatch{
                .first_instance = first_inst,
                .instance_count = count,
                .visible_start_us = visible_start,
                .px_per_us = static_cast<float>(px_per_time_unit_val),
                .screen_x_offset = pos.x,
                .timeline_width = current_timeline_width_,
                .y_top = pos.y,
                .y_bottom = pos.y + group_height,
                .min_width_px = kEventMinimumDrawWidth,
                .padding_right_px = 0.0f,
                .alpha_multiplier = kGroupPreviewOpacity,
            });
        continue;
      }
    }

    Pixel last_drawn_right = -std::numeric_limits<Pixel>::infinity();
    for (; it != indices.end(); ++it) {
      int event_index = *it;
      if (!IsEventVisibleUnderFilter(event_index)) {
        continue;
      }
      const Microseconds start = timeline_data_.entry_start_times[event_index];
      if (start >= visible_end) break;

      const Microseconds end =
          start + timeline_data_.entry_total_times[event_index];

      Pixel x_start = TimeToScreenX(start, pos.x, px_per_time_unit_val);
      Pixel x_end = TimeToScreenX(end, pos.x, px_per_time_unit_val);
      if (x_end < x_start) std::swap(x_start, x_end);
      if (x_start < last_drawn_right && x_end <= last_drawn_right + 0.5f) {
        continue;
      }

      // Draw Logic
      ImU32 color = GetEventColor(event_index);
      // Render with reduced opacity to show density.
      color = (color & ~IM_COL32_A_MASK) |
              (static_cast<ImU32>(kGroupPreviewOpacity * 255.0f)
               << IM_COL32_A_SHIFT);

      x_end = std::max(x_end, x_start + kEventMinimumDrawWidth);
      last_drawn_right = std::max(last_drawn_right, x_end);

      draw_list->AddRectFilled(ImVec2(x_start, pos.y),
                               ImVec2(x_end, pos.y + group_height), color);
    }
  }
}

void Timeline::AccumulateClippedRangeOccupancy(Pixel x_start, Pixel x_end,
                                               float density, int num_bins) {
  if (num_bins <= 0 || density <= 0.0f || x_end <= x_start) return;
  const Pixel max_x = static_cast<Pixel>(num_bins);
  if (x_end <= 0.0f || x_start >= max_x) return;

  const Pixel clipped_start = std::max(0.0f, x_start);
  const Pixel clipped_end = std::min(max_x, x_end);
  const int bin_start =
      std::max(0, static_cast<int>(std::floor(clipped_start)));
  const int bin_end = std::min(
      num_bins - 1, static_cast<int>(std::ceil(clipped_end - kEpsilon)));

  for (int i = bin_start; i <= bin_end; ++i) {
    const Pixel overlap = std::min(clipped_end, static_cast<Pixel>(i + 1)) -
                          std::max(clipped_start, static_cast<Pixel>(i));
    if (overlap > 0.0f) {
      utilization_bins_[i] += overlap * density;
    }
  }
}

bool Timeline::TryAccumulateProcessUtilization(int start_level, int end_level,
                                               double px_per_time_unit_val,
                                               int num_bins) {
  if (px_per_time_unit_val <= 0.0) return false;
  if (process_utilization_pyramids_dirty_) {
    BuildProcessUtilizationPyramids();
  }
  const Microseconds us_per_px = 1.0 / px_per_time_unit_val;

  const ProcessUtilizationPyramid* matching_pyr = nullptr;
  for (const auto& proc_pyr : process_utilization_pyramids_) {
    if (proc_pyr.start_level == start_level &&
        proc_pyr.end_level == end_level) {
      matching_pyr = &proc_pyr;
      break;
    }
  }
  if (matching_pyr == nullptr) return false;

  const int selected_mip = matching_pyr->SelectMipLevel(
      kMaxProcessUtilizationMipBinWidthPx * us_per_px);
  if (selected_mip < 0) return false;

  const ProcessUtilizationMip& pmip = matching_pyr->levels[selected_mip];
  const int m_bins = static_cast<int>(pmip.occupied_us_bins.size());
  if (m_bins <= 0 || pmip.bin_width_us <= 0.0) return false;

  const Microseconds visible_start = visible_range().start();
  const Microseconds visible_end = visible_range().end();
  const Microseconds pmip_max_ts = pmip.min_ts + m_bins * pmip.bin_width_us;
  if (visible_end <= pmip.min_ts || visible_start >= pmip_max_ts) {
    return true;
  }

  const double inv_bw = 1.0 / pmip.bin_width_us;
  const double max_bin_idx = static_cast<double>(m_bins - 1);
  const int b_first = static_cast<int>(std::clamp(
      std::floor((visible_start - pmip.min_ts) * inv_bw), 0.0, max_bin_idx));
  const int b_last = static_cast<int>(std::clamp(
      std::ceil((visible_end - pmip.min_ts) * inv_bw), 0.0, max_bin_idx));

  for (int b = b_first; b <= b_last; ++b) {
    const float occ_us = pmip.occupied_us_bins[b];
    if (occ_us <= 0.0f) continue;
    const Microseconds ts_s = pmip.min_ts + b * pmip.bin_width_us;
    const Microseconds ts_e = ts_s + pmip.bin_width_us;
    const Pixel x_start = TimeToPixel(ts_s, px_per_time_unit_val);
    const Pixel x_end = TimeToPixel(ts_e, px_per_time_unit_val);
    const Pixel tile_px = x_end - x_start;
    if (tile_px <= 0.0f) continue;
    const float occ_px = static_cast<float>(occ_us * px_per_time_unit_val);
    AccumulateClippedRangeOccupancy(x_start, x_end, occ_px / tile_px, num_bins);
  }
  return true;
}

void Timeline::AccumulateLevelUtilization(int level,
                                          double px_per_time_unit_val,
                                          int num_bins) {
  const Microseconds visible_start = visible_range().start();
  const Microseconds visible_end = visible_range().end();

  if (px_per_time_unit_val > 0.0 && level >= 0 &&
      static_cast<size_t>(level) < level_mip_pyramids_.size()) {
    EnsureLevelMipPyramid(level);
    const LevelMipPyramid& pyramid = level_mip_pyramids_[level];
    const Microseconds us_per_px = 1.0 / px_per_time_unit_val;
    const int selected_mip =
        pyramid.SelectMipLevel(kMaxPreviewMipBinWidthPx * us_per_px);
    if (selected_mip >= 0) {
      const auto& tiles = pyramid.levels[selected_mip].tiles;
      auto tile_first =
          std::lower_bound(tiles.begin(), tiles.end(), visible_start,
                           [](const LevelMipTile& tile, Microseconds t) {
                             return tile.max_end <= t;
                           });
      for (auto it = tile_first; it != tiles.end(); ++it) {
        if (it->min_start >= visible_end) break;
        const Pixel x_start = TimeToPixel(it->min_start, px_per_time_unit_val);
        const Pixel x_end = TimeToPixel(it->max_end, px_per_time_unit_val);
        const Pixel tile_px_span = x_end - x_start;
        if (tile_px_span <= 0.0f) continue;

        const float occupied_px =
            static_cast<float>(it->total_occupied_dur * px_per_time_unit_val);
        const float density = std::min(1.0f, occupied_px / tile_px_span);
        AccumulateClippedRangeOccupancy(x_start, x_end, density, num_bins);
      }
      return;
    }
  }

  const absl::Span<const int> indices = timeline_data_.level_events(level);
  auto it = std::lower_bound(indices.begin(), indices.end(), visible_start,
                             [&](int event_idx, Microseconds t) {
                               const Microseconds end =
                                   timeline_data_.entry_start_times[event_idx] +
                                   timeline_data_.entry_total_times[event_idx];
                               return end <= t;
                             });

  for (; it != indices.end(); ++it) {
    const int event_index = *it;
    const Microseconds start = timeline_data_.entry_start_times[event_index];
    if (start >= visible_end) break;
    if (!IsEventVisibleUnderFilter(event_index)) continue;
    const Microseconds end =
        start + timeline_data_.entry_total_times[event_index];

    // Calculate pixel coordinates relative to the start of the visible range.
    const Pixel x_start = TimeToPixel(start, px_per_time_unit_val);
    const Pixel x_end = TimeToPixel(end, px_per_time_unit_val);

    // Clip events that are partially outside the visible range.
    // Offset by 0.5 to center the bins on pixels? No, ImGui uses screen
    // coords.
    AccumulateClippedRangeOccupancy(x_start, x_end, /*density=*/1.0f, num_bins);
  }
}

void Timeline::DrawUtilizationAreaChart(int group_index, int start_level,
                                        int end_level,
                                        double px_per_time_unit_val,
                                        const ImVec2& pos, Pixel group_height,
                                        ImDrawList* draw_list) {
  const Pixel timeline_width = current_timeline_width_;
  if (timeline_width <= 0) return;

  const int num_bins = static_cast<int>(std::ceil(timeline_width));
  if (num_bins <= 0) return;

  if (utilization_bins_.size() < static_cast<size_t>(num_bins)) {
    utilization_bins_.resize(num_bins);
  }
  std::fill(utilization_bins_.begin(), utilization_bins_.begin() + num_bins,
            0.0f);

  if (has_active_event_filter_ ||
      !TryAccumulateProcessUtilization(start_level, end_level,
                                       px_per_time_unit_val, num_bins)) {
    for (int level = start_level; level < end_level; ++level) {
      AccumulateLevelUtilization(level, px_per_time_unit_val, num_bins);
    }
  }

  Pixel max_util = 0.0f;
  for (int i = 0; i < num_bins; ++i) {
    max_util = std::max(max_util, utilization_bins_[i]);
  }

  // Normalize by at least one full track of activity.
  max_util = std::max(kMinUtilizationNormalization, max_util);

  const ImU32 bar_color =
      palette_.GetColor(ColorPalette::Key::kFlameHeader).value_or(kBlue70);

  if (timeline_data_.entry_start_times.size() <= 64) {
    for (int i = 0; i < num_bins; ++i) {
      if (utilization_bins_[i] > 0.0f) {
        const Pixel h = (utilization_bins_[i] / max_util) * group_height;
        draw_list->AddRectFilled(ImVec2(pos.x + i, pos.y + group_height - h),
                                 ImVec2(pos.x + i + 1, pos.y + group_height),
                                 bar_color);
      }
    }
  } else {
    // Coalesce adjacent bins with identical quantized heights (0.5px steps)
    // into a single AddRectFilled call.
    int i = 0;
    while (i < num_bins) {
      if (utilization_bins_[i] <= 0.0f) {
        ++i;
        continue;
      }
      const Pixel raw_h = (utilization_bins_[i] / max_util) * group_height;
      const Pixel q_h = std::round(raw_h * 2.0f) * 0.5f;
      if (q_h <= 0.0f) {
        ++i;
        continue;
      }
      int j = i + 1;
      while (j < num_bins && utilization_bins_[j] > 0.0f) {
        const Pixel next_raw_h =
            (utilization_bins_[j] / max_util) * group_height;
        const Pixel next_q_h = std::round(next_raw_h * 2.0f) * 0.5f;
        if (next_q_h != q_h) break;
        ++j;
      }
      draw_list->AddRectFilled(ImVec2(pos.x + i, pos.y + group_height - q_h),
                               ImVec2(pos.x + j, pos.y + group_height),
                               bar_color);
      i = j;
    }
  }

  // Draw tooltips when hovering over the chart.
  if (ImGui::IsWindowHovered()) {
    const ImVec2 mouse_pos = ImGui::GetMousePos();
    if (mouse_pos.x >= pos.x && mouse_pos.x < pos.x + timeline_width &&
        mouse_pos.y >= pos.y && mouse_pos.y < pos.y + group_height) {
      const int bin_idx = static_cast<int>(mouse_pos.x - pos.x);
      if (bin_idx >= 0 && bin_idx < num_bins) {
        float val = utilization_bins_[bin_idx];
        ImGui::SetTooltip(
            "Utilization: %.2f\n(Chart height represents event density)", val);
        if (!event_tooltip_enabled_ && hovered_event_index_ == -1) {
          const double mouse_time =
              PixelToTime(mouse_pos.x - pos.x, px_per_time_unit_val);
          if (mouse_time >= 0.0) {
            hovered_group_index_ = group_index;
            hovered_bundle_ = static_cast<int64_t>(std::floor(mouse_time));
            hovered_counter_value_ = val;
          }
        }
      }
    }
  }
}

int Timeline::FindEventAtLevelAndTime(int level, Microseconds ts) const {
  if (level < 0) {
    return -1;
  }
  const absl::Span<const int> indices = timeline_data_.level_events(level);
  if (!indices.empty()) {
    const auto it_idx = std::upper_bound(
        indices.begin(), indices.end(), ts,
        [this](Microseconds target_ts, int idx) {
          if (idx < 0 || static_cast<size_t>(idx) >=
                             timeline_data_.entry_start_times.size()) {
            return false;
          }
          return target_ts < timeline_data_.entry_start_times[idx];
        });
    if (it_idx != indices.begin()) {
      const int idx = *std::prev(it_idx);
      if (idx >= 0 &&
          static_cast<size_t>(idx) < timeline_data_.entry_start_times.size() &&
          static_cast<size_t>(idx) < timeline_data_.entry_total_times.size() &&
          timeline_data_.entry_start_times[idx] <= ts &&
          timeline_data_.entry_start_times[idx] +
                  timeline_data_.entry_total_times[idx] >=
              ts) {
        return idx;
      }
    }
    return -1;
  }
  const size_t n = std::min({timeline_data_.entry_levels.size(),
                             timeline_data_.entry_start_times.size(),
                             timeline_data_.entry_total_times.size()});
  for (size_t i = 0; i < n; ++i) {
    if (timeline_data_.entry_levels[i] == level &&
        timeline_data_.entry_start_times[i] <= ts &&
        timeline_data_.entry_start_times[i] +
                timeline_data_.entry_total_times[i] >=
            ts) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

bool Timeline::IsFlowVisibleUnderFilter(const FlowLine& flow) const {
  if (!has_active_event_filter_) {
    return true;
  }
  const int src_idx =
      FindEventAtLevelAndTime(flow.source_level, flow.source_ts);
  const int dst_idx =
      FindEventAtLevelAndTime(flow.target_level, flow.target_ts);
  return src_idx >= 0 && dst_idx >= 0 && IsEventVisibleUnderFilter(src_idx) &&
         IsEventVisibleUnderFilter(dst_idx);
}

void Timeline::DrawSingleFlow(const FlowLine& flow, Pixel timeline_x_start,
                              Pixel timeline_y_start, double px_per_time,
                              ImDrawList* draw_list) {
  if (!IsFlowVisibleUnderFilter(flow) ||
      flow.source_level >= visible_level_offsets_.size() ||
      flow.target_level >= visible_level_offsets_.size()) {
    return;
  }

  const Pixel start_y =
      timeline_y_start + visible_level_offsets_[flow.source_level];
  const Pixel end_y =
      timeline_y_start + visible_level_offsets_[flow.target_level];

  const Pixel start_x =
      TimeToScreenX(flow.source_ts, timeline_x_start, px_per_time);
  const Pixel end_x =
      TimeToScreenX(flow.target_ts, timeline_x_start, px_per_time);

  const ImVec2 p0(start_x, start_y);
  const ImVec2 p1(end_x, end_y);

  ImVec2 cp0, cp1;
  Timeline::CalculateBezierControlPoints(start_x, start_y, end_x, end_y, cp0,
                                         cp1);

  draw_list->AddBezierCubic(p0, cp0, cp1, p1, flow.color, 1.0f);
  draw_list->AddCircleFilled(p0, kPointRadius, flow.color);
  draw_list->AddCircleFilled(p1, kPointRadius, flow.color);
}

void Timeline::SetVisibleFlowCategories(const std::vector<int>& category_ids) {
  visible_flow_categories_.clear();
  visible_flow_categories_.insert(category_ids.begin(), category_ids.end());
  if (redraw_callback_) redraw_callback_();
}

void Timeline::DrawFlows(Pixel timeline_width, Pixel timeline_y_start) {
  const bool has_selected_event =
      selected_event_index_ >= 0 &&
      (selected_event_index_ <
           static_cast<int>(timeline_data_.entry_event_ids.size()) ||
       timeline_data_.flow_ids_by_event_index.contains(selected_event_index_) ||
       selected_event_index_ <
           static_cast<int>(timeline_data_.entry_name_indices.size()));

  if ((visible_flow_categories_.empty() && !has_selected_event) ||
      timeline_data_.flow_lines.empty()) {
    return;
  }

  const ImVec2 table_rect_min = ImGui::GetItemRectMin();
  const Pixel timeline_x_start = table_rect_min.x + label_width_;

  // Use ForegroundDrawList to draw on top of opaque child windows (groups).
  ImDrawList* const draw_list = ImGui::GetForegroundDrawList();

  // Clip to the current window ("Tracks") to avoid drawing outside the timeline
  // area.
  const ImVec2 window_pos = ImGui::GetWindowPos();
  const ImVec2 clip_min = ImVec2(window_pos.x + label_width_, window_pos.y);
  const ImVec2 clip_max = ImVec2(window_pos.x + ImGui::GetWindowWidth(),
                                 window_pos.y + ImGui::GetWindowHeight());
  draw_list->PushClipRect(clip_min, clip_max,
                          /*intersect_with_current_clip_rect=*/true);

  const double px_per_time = px_per_time_unit(timeline_width);
  // If px_per_time is non-positive, it indicates an invalid or zero-width
  // timeline or view duration, so we cannot draw flows.
  if (px_per_time <= 0) {
    draw_list->PopClipRect();
    return;
  }

  if (has_selected_event) {
    if (const auto it_idx =
            timeline_data_.flow_ids_by_event_index.find(selected_event_index_);
        it_idx != timeline_data_.flow_ids_by_event_index.end()) {
      if (const auto it_range =
              timeline_data_.flow_line_ranges_by_flow_id.find(it_idx->second);
          it_range != timeline_data_.flow_line_ranges_by_flow_id.end()) {
        const size_t end = std::min(static_cast<size_t>(it_range->second.end),
                                    timeline_data_.flow_lines.size());
        for (size_t i = it_range->second.begin; i < end; ++i) {
          DrawSingleFlow(timeline_data_.flow_lines[i], timeline_x_start,
                         timeline_y_start, px_per_time, draw_list);
        }
      } else {
        const std::string flow_id_str = std::to_string(it_idx->second);
        const auto it_lines =
            timeline_data_.flow_lines_by_flow_id.find(flow_id_str);
        if (it_lines != timeline_data_.flow_lines_by_flow_id.end()) {
          for (const FlowLine& flow : it_lines->second) {
            DrawSingleFlow(flow, timeline_x_start, timeline_y_start,
                           px_per_time, draw_list);
          }
        }
      }
    } else if (selected_event_index_ >= 0 &&
               selected_event_index_ <
                   static_cast<int>(timeline_data_.entry_start_times.size())) {
      const EventId selected_event_id =
          GetEventIdFromData(timeline_data_, selected_event_index_);
      const auto it_ids =
          timeline_data_.flow_ids_by_event_id.find(selected_event_id);
      if (it_ids != timeline_data_.flow_ids_by_event_id.end()) {
        for (const std::string& flow_id : it_ids->second) {
          const auto it_lines =
              timeline_data_.flow_lines_by_flow_id.find(flow_id);
          if (it_lines != timeline_data_.flow_lines_by_flow_id.end()) {
            for (const FlowLine& flow : it_lines->second) {
              DrawSingleFlow(flow, timeline_x_start, timeline_y_start,
                             px_per_time, draw_list);
            }
          }
        }
      }
    }
  } else {
    for (const auto& flow : timeline_data_.flow_lines) {
      if (visible_flow_categories_.contains(static_cast<int>(flow.category))) {
        DrawSingleFlow(flow, timeline_x_start, timeline_y_start, px_per_time,
                       draw_list);
      }
    }
  }

  draw_list->PopClipRect();
}

void Timeline::DrawDependencyArrow(int from_event, int to_event, ImU32 color,
                                   Pixel timeline_x_start,
                                   Pixel timeline_y_start, double px_per_time,
                                   ImDrawList* draw_list) {
  const uint16_t from_level = timeline_data_.entry_levels[from_event];
  const uint16_t to_level = timeline_data_.entry_levels[to_event];
  if (from_level >= visible_level_offsets_.size() ||
      to_level >= visible_level_offsets_.size()) {
    return;
  }
  const ImVec2 start(
      TimeToScreenX(timeline_data_.entry_start_times[from_event] +
                        timeline_data_.entry_total_times[from_event],
                    timeline_x_start, px_per_time),
      timeline_y_start + visible_level_offsets_[from_level]);
  const ImVec2 end(TimeToScreenX(timeline_data_.entry_start_times[to_event],
                                 timeline_x_start, px_per_time),
                   timeline_y_start + visible_level_offsets_[to_level]);

  ImVec2 cp0, cp1;
  CalculateBezierControlPoints(start.x, start.y, end.x, end.y, cp0, cp1);
  draw_list->AddBezierCubic(start, cp0, cp1, end, color,
                            kHloDependencyArrowThickness);
  draw_list->AddCircleFilled(start, kPointRadius, color);

  // The arrowhead points along the end of the curve. If the curve ends
  // vertically, its last control point coincides with the end, so fall back to
  // the overall direction.
  ImVec2 direction(end.x - cp1.x, end.y - cp1.y);
  if (direction.x == 0.0f && direction.y == 0.0f) {
    direction = ImVec2(end.x - start.x, end.y - start.y);
  }
  const float length = std::hypot(direction.x, direction.y);
  if (length == 0.0f) return;
  direction = ImVec2(direction.x / length, direction.y / length);
  const ImVec2 base(end.x - direction.x * kHloDependencyArrowHeadSize,
                    end.y - direction.y * kHloDependencyArrowHeadSize);
  const ImVec2 half_width(-direction.y * kHloDependencyArrowHeadSize * 0.5f,
                          direction.x * kHloDependencyArrowHeadSize * 0.5f);
  draw_list->AddTriangleFilled(
      end, ImVec2(base.x + half_width.x, base.y + half_width.y),
      ImVec2(base.x - half_width.x, base.y - half_width.y), color);
}

void Timeline::DrawDependencyArrows(Pixel timeline_width,
                                    Pixel timeline_y_start) {
  const absl::Span<const int> producers = producer_event_indices();
  const absl::Span<const int> consumers = consumer_event_indices();
  if (producers.empty() && consumers.empty()) return;
  if (selected_event_index_ < 0 ||
      selected_event_index_ >=
          static_cast<int>(timeline_data_.entry_levels.size())) {
    return;
  }
  const double px_per_time = px_per_time_unit(timeline_width);
  if (px_per_time <= 0) return;

  const Pixel timeline_x_start = ImGui::GetItemRectMin().x + label_width_;
  // Drawn like the flow lines: on the foreground draw list, clipped to the
  // timeline area of the current window.
  ImDrawList* const draw_list = ImGui::GetForegroundDrawList();
  const ImVec2 window_pos = ImGui::GetWindowPos();
  draw_list->PushClipRect(ImVec2(window_pos.x + label_width_, window_pos.y),
                          ImVec2(window_pos.x + ImGui::GetWindowWidth(),
                                 window_pos.y + ImGui::GetWindowHeight()),
                          /*intersect_with_current_clip_rect=*/true);

  const ImU32 producer_color =
      palette_.GetColor(ColorPalette::Key::kHloProducerArrow)
          .value_or(kHloProducerArrowColor);
  for (const int producer : producers.first(
           std::min(producers.size(), kMaxHloDependencyArrowsPerSide))) {
    DrawDependencyArrow(producer, selected_event_index_, producer_color,
                        timeline_x_start, timeline_y_start, px_per_time,
                        draw_list);
  }
  const ImU32 consumer_color =
      palette_.GetColor(ColorPalette::Key::kHloConsumerArrow)
          .value_or(kHloConsumerArrowColor);
  for (const int consumer : consumers.first(
           std::min(consumers.size(), kMaxHloDependencyArrowsPerSide))) {
    DrawDependencyArrow(selected_event_index_, consumer, consumer_color,
                        timeline_x_start, timeline_y_start, px_per_time,
                        draw_list);
  }
  draw_list->PopClipRect();
}

void Timeline::DrawSelectedTimeRange(const TimeRange& range,
                                     Pixel timeline_width,
                                     double px_per_time_unit_val,
                                     bool show_delete_button,
                                     std::optional<size_t> range_index) {
  const ImGuiViewport* viewport = ImGui::GetMainViewport();
  const Pixel timeline_x_start = viewport->Pos.x + label_width_;
  const ImU32 color = palette_.GetColor(ColorPalette::Key::kSelection)
                          .value_or(kSelectedTimeRangeColor);

  const Pixel time_range_x_start =
      TimeToScreenX(range.start(), timeline_x_start, px_per_time_unit_val);
  const Pixel time_range_x_end =
      TimeToScreenX(range.end(), timeline_x_start, px_per_time_unit_val) -
      kEventPaddingRight;
  // Clip the selection rectangle to the visible timeline bounds.
  // If the selection starts before the timeline's visible area,
  // clipped_x_start ensures we only start drawing from timeline_x_start.
  const Pixel clipped_x_start = std::max(time_range_x_start, timeline_x_start);
  // If the selection ends after the timeline's visible area, clipped_x_end
  // ensures we stop drawing at the right edge of the timeline.
  const Pixel clipped_x_end =
      std::min(time_range_x_end, timeline_x_start + timeline_width);

  if (clipped_x_end > clipped_x_start) {
    // Use the window draw list to render over all other timeline content.
    ImDrawList* const draw_list = ImGui::GetWindowDrawList();

    const Pixel rect_y_min = ruler_screen_y_ + kRulerHeight;
    const Pixel rect_y_max = viewport->Pos.y + viewport->Size.y;

    draw_list->AddRectFilledMultiColor(ImVec2(clipped_x_start, rect_y_min),
                                       ImVec2(clipped_x_end, rect_y_max),
                                       (color & 0x00FFFFFF) | (0x99 << 24),
                                       (color & 0x00FFFFFF) | (0x99 << 24),
                                       (color & 0x00FFFFFF) | (0x1A << 24),
                                       (color & 0x00FFFFFF) | (0x1A << 24));

    // Only draw the border if the edge of the time range is visible.
    if (time_range_x_start >= timeline_x_start) {
      bool is_hovering_start = false;
      if (!is_dragging_ &&
          std::abs(ImGui::GetMousePos().x - time_range_x_start) <=
              kSelectionEdgeThreshold &&
          ImGui::IsMouseHoveringRect(
              ImVec2(time_range_x_start - kSelectionEdgeThreshold, rect_y_min),
              ImVec2(time_range_x_start + kSelectionEdgeThreshold,
                     rect_y_max))) {
        is_hovering_start = true;
      }

      ImU32 start_edge_color = color;
      bool is_resizing_start = false;
      if (range_index.has_value()) {
        is_resizing_start =
            time_range_resizing_state_.has_value() &&
            time_range_resizing_state_->range_index == *range_index &&
            time_range_resizing_state_->is_start_edge;
      } else if (is_dragging_ && is_selecting_) {
        is_resizing_start =
            std::abs(ImGui::GetMousePos().x - time_range_x_start) <
            std::abs(ImGui::GetMousePos().x - time_range_x_end);
      }
      if (is_hovering_start || is_resizing_start) {
        start_edge_color = kSelectedTimeRangeResizeColor;
      }
      draw_list->AddLine(ImVec2(time_range_x_start, rect_y_min),
                         ImVec2(time_range_x_start, rect_y_max),
                         start_edge_color,
                         /*thickness=*/kSelectedTimeRangeThickness);

      if (is_hovering_start) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
      }
    }
    if (time_range_x_end <= timeline_x_start + timeline_width) {
      bool is_hovering_end = false;
      if (!is_dragging_ &&
          std::abs(ImGui::GetMousePos().x - time_range_x_end) <=
              kSelectionEdgeThreshold &&
          ImGui::IsMouseHoveringRect(
              ImVec2(time_range_x_end - kSelectionEdgeThreshold, rect_y_min),
              ImVec2(time_range_x_end + kSelectionEdgeThreshold, rect_y_max))) {
        is_hovering_end = true;
      }

      ImU32 end_edge_color = color;
      bool is_resizing_end = false;
      if (range_index.has_value()) {
        is_resizing_end =
            time_range_resizing_state_.has_value() &&
            time_range_resizing_state_->range_index == *range_index &&
            !time_range_resizing_state_->is_start_edge;
      } else if (is_dragging_ && is_selecting_) {
        is_resizing_end = std::abs(ImGui::GetMousePos().x - time_range_x_end) <=
                          std::abs(ImGui::GetMousePos().x - time_range_x_start);
      }
      if (is_hovering_end || is_resizing_end) {
        end_edge_color = kSelectedTimeRangeResizeColor;
      }
      draw_list->AddLine(ImVec2(time_range_x_end, rect_y_min),
                         ImVec2(time_range_x_end, rect_y_max), end_edge_color,
                         /*thickness=*/kSelectedTimeRangeThickness);

      if (is_hovering_end) {
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
      }
    }

    const std::string text = FormatDuration(range.duration(), time_axis_unit_);
    const ImVec2 text_size = ImGui::CalcTextSize(text.c_str());
    const ImU32 kTextColor =
        palette_.GetColor(ColorPalette::Key::kForeground).value_or(kBlackColor);

    // Move the text to the top, below the ruler.
    const Pixel text_y = rect_y_min + kSelectedTimeRangeTextTopPadding;
    const Pixel text_x = clipped_x_start +
                         (clipped_x_end - clipped_x_start - text_size.x) / 2.0f;
    const ImVec2 text_pos(text_x, text_y);

    const ImRect visible_range_rect(ImVec2(clipped_x_start, rect_y_min),
                                    ImVec2(clipped_x_end, rect_y_max));
    const ImRect full_range_rect(ImVec2(time_range_x_start, rect_y_min),
                                 ImVec2(time_range_x_end, rect_y_max));

    if (show_delete_button) {
      const DeleteButtonLayout layout = GetDeleteButtonLayout(
          text_size, text_pos, visible_range_rect, full_range_rect);

      if (layout.text_fits) {
        draw_list->AddText(text_pos, kTextColor, text.c_str());
      }

      DrawDeleteButton(draw_list, layout.button_pos, layout.hover_rect, range);
    } else {
      bool text_fits = visible_range_rect.GetWidth() > text_size.x;
      if (text_fits) {
        draw_list->AddText(text_pos, kTextColor, text.c_str());
      }
    }
  }
}

DeleteButtonLayout Timeline::GetDeleteButtonLayout(
    const ImVec2& text_size, const ImVec2& text_pos,
    const ImRect& visible_range_rect, const ImRect& full_range_rect) const {
  ImVec2 button_pos;
  ImRect hover_rect;
  bool text_fits = false;

  // Check if text fits in the visible range.
  if (visible_range_rect.GetWidth() > text_size.x) {
    text_fits = true;
    // Position the button to the right of the text.
    button_pos = ImVec2(text_pos.x + text_size.x + kCloseButtonPadding,
                        text_pos.y + (text_size.y - kCloseButtonSize) / 2.0f);

    // Expand the hover area to include both the text and the button, with a
    // small margin.
    ImVec2 hover_min(std::min(text_pos.x, button_pos.x),
                     std::min(text_pos.y, button_pos.y));
    ImVec2 hover_max(
        std::max(text_pos.x + text_size.x, button_pos.x + kCloseButtonSize),
        std::max(text_pos.y + text_size.y, button_pos.y + kCloseButtonSize));

    hover_min.x -= kHoverPadding;
    hover_min.y -= kHoverPadding;
    hover_max.x += kHoverPadding;
    hover_max.y += kHoverPadding;
    hover_rect = ImRect(hover_min, hover_max);
  } else {
    // If text doesn't fit, center the button in the visible range.
    const Pixel center_x = visible_range_rect.GetCenter().x;
    button_pos = ImVec2(center_x - kCloseButtonSize / 2.0f,
                        text_pos.y + (text_size.y - kCloseButtonSize) / 2.0f);

    // Set the hover area to the visible selected time range, expanded to
    // include the button.
    hover_rect = visible_range_rect;
    const ImRect button_rect(button_pos,
                             ImVec2(button_pos.x + kCloseButtonSize,
                                    button_pos.y + kCloseButtonSize));
    hover_rect.Add(button_rect);
  }

  return {button_pos, hover_rect, text_fits};
}

void Timeline::DrawDeleteButton(ImDrawList* draw_list, const ImVec2& button_pos,
                                const ImRect& hover_rect,
                                const TimeRange& range) {
  if (DrawCloseButton(draw_list, button_pos, hover_rect)) {
    auto it = absl::c_find(selected_time_ranges_, range);
    if (it != selected_time_ranges_.end()) {
      selected_time_ranges_.erase(it);
    }
    if (current_selected_time_range_ &&
        *current_selected_time_range_ == range) {
      current_selected_time_range_.reset();
    }
  }
}

bool Timeline::DrawCloseButton(ImDrawList* draw_list, const ImVec2& button_pos,
                               const ImRect& hover_rect) {
  const Pixel button_size = kCloseButtonSize;
  const ImVec2 button_min = button_pos;
  const ImVec2 button_max(button_pos.x + button_size,
                          button_pos.y + button_size);

  bool clicked = false;
  ImU32 button_color = kCloseButtonColor;

  if (ImGui::IsMouseHoveringRect(button_min, button_max)) {
    button_color = kCloseButtonHoverColor;
    ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    if (ImGui::IsMouseClicked(0)) {
      clicked = true;
    }
  }

  const ImVec2 center(button_min.x + button_size / 2.0f,
                      button_min.y + button_size / 2.0f);
  draw_list->AddCircleFilled(center, button_size / 2.0f, button_color);

  const Pixel x_radius = button_size * 0.25f;
  draw_list->AddLine(ImVec2(center.x - x_radius, center.y - x_radius),
                     ImVec2(center.x + x_radius, center.y + x_radius),
                     kWhiteColor);
  draw_list->AddLine(ImVec2(center.x - x_radius, center.y + x_radius),
                     ImVec2(center.x + x_radius, center.y - x_radius),
                     kWhiteColor);
  return clicked;
}

bool Timeline::DrawTrackManagementButtons(int group_index, const Group& group,
                                          const ImVec2& tracks_start_pos,
                                          Pixel centereable_height) {
  if (!track_management_enabled_) return false;
  if (group.nesting_level != kProcessNestingLevel) return false;

  bool needs_layout_update = false;

  // The buttons are right-aligned in the label column with a small margin
  // before the splitter: [pin] gap [hide] margin | splitter.
  const Pixel row_y = tracks_start_pos.y + group_offsets_[group_index];
  const Pixel hide_x = tracks_start_pos.x + label_width_ - kSplitterOffset -
                       kTrackIconButtonMarginRight - kTrackIconButtonSize;
  const Pixel pin_x = hide_x - kTrackIconButtonGap - kTrackIconButtonSize;

  // Position and draw Pin button
  ImGui::SetCursorPos(ImVec2(pin_x, row_y));
  const bool is_pinned = pinned_track_names_.contains(group.name);
  if (DrawPinButton(group_index, centereable_height, is_pinned)) {
    needs_layout_update = true;
    ShowNavigationWarningNotification(
        is_pinned
            ? absl::StrCat(kUnpinnedProcessNotificationPrefix, group.name)
            : absl::StrCat(kPinnedProcessNotificationPrefix, group.name));
  }

  // Position and draw Hide button
  ImGui::SetCursorPos(ImVec2(hide_x, row_y));
  const bool is_track_hidden = hidden_track_names_.contains(group.name);
  if (DrawHideButton(group_index, centereable_height, is_track_hidden)) {
    needs_layout_update = true;
    // The state was toggled inside DrawHideButton, so we should check the
    // new state or invert the old state. Since is_track_hidden holds the
    // old state:
    // If it WAS hidden (true), it is now UNHIDDEN.
    // If it WAS visible (false), it is now HIDDEN.
    ShowNavigationWarningNotification(
        is_track_hidden
            ? absl::StrCat(kUnhiddenProcessNotificationPrefix, group.name)
            : absl::StrCat(kHiddenProcessNotificationPrefix, group.name));
  }

  return needs_layout_update;
}

bool Timeline::IsTrackRowHovered(int group_index) const {
  const Pixel content_region_avail_width =
      ImGui::GetWindowWidth() - ImGui::GetStyle().ScrollbarSize;
  const Pixel row_top =
      tracks_start_screen_pos_.y + group_offsets_[group_index];
  return ImGui::IsMouseHoveringRect(
      ImVec2(tracks_start_screen_pos_.x, row_top),
      ImVec2(tracks_start_screen_pos_.x + content_region_avail_width,
             row_top + group_heights_[group_index]));
}

bool Timeline::DrawHideButton(int group_index, Pixel height,
                              bool is_track_hidden) {
  if (height <= 0.0f) return false;
  const Group& group = timeline_data_.groups[group_index];
  const ImU32 foreground_color =
      palette_.GetColor(ColorPalette::Key::kForeground).value_or(kBlackColor);

  const IconButtonState state = TrackIconButton(
      "##hide", height, is_track_hidden,
      is_track_hidden ? kUnhideTrackTooltip : kHideTrackTooltip,
      foreground_color,
      [this](ImGuiID id, bool hovered) { return HoverFade(id, hovered); });

  bool toggled = false;
  if (state.clicked) {
    auto it = hidden_track_names_.find(group.name);
    if (it != hidden_track_names_.end()) {
      hidden_track_names_.erase(it);
      toggled = true;
    } else {
      if (all_processes_count_ > 1) {
        hidden_track_names_.insert(group.name);
        toggled = true;
      } else {
        ShowNavigationWarningNotification(kCannotHideLastProcessNotification);
      }
    }
  }

  // The button is only revealed on row hover, unless the track is hidden.
  if (state.hovered || is_track_hidden || IsTrackRowHovered(group_index)) {
    DrawHideIcon(ImGui::GetWindowDrawList(), state.center.x, state.center.y,
                 kTrackIconButtonSize,
                 TrackIconColor(state, is_track_hidden, foreground_color),
                 is_track_hidden);
  }
  return toggled;
}

bool Timeline::DrawPinButton(int group_index, Pixel height, bool is_pinned) {
  if (height <= 0.0f) return false;
  const Group& group = timeline_data_.groups[group_index];
  const ImU32 foreground_color =
      palette_.GetColor(ColorPalette::Key::kForeground).value_or(kBlackColor);

  const IconButtonState state = TrackIconButton(
      "##pin", height, is_pinned,
      is_pinned ? kUnpinTrackTooltip : kPinTrackTooltip, foreground_color,
      [this](ImGuiID id, bool hovered) { return HoverFade(id, hovered); });

  bool toggled = false;
  if (state.clicked) {
    if (is_pinned) {
      pinned_track_names_.erase(group.name);
    } else {
      pinned_track_names_.insert(group.name);
    }
    toggled = true;
  }

  // The button is only revealed on row hover, unless the track is pinned.
  if (state.hovered || is_pinned || IsTrackRowHovered(group_index)) {
    DrawPinIcon(ImGui::GetWindowDrawList(), state.center.x, state.center.y,
                kTrackIconButtonSize,
                TrackIconColor(state, is_pinned, foreground_color), is_pinned);
  }
  return toggled;
}

void Timeline::DrawSelectedTimeRanges(Pixel timeline_width,
                                      double px_per_time_unit_val) {
  // We make a copy of the selected time ranges because DrawSelectedTimeRange
  // might modify `selected_time_ranges_` (e.g. by deleting a range), which
  // would invalidate iterators if we were iterating over the original vector.
  // We want to iterate over the ranges as they were at the start of the frame.
  // Note that `selected_time_ranges_` contains only user-created selections
  // (typically very few), not the trace events, so this copy is negligible in
  // terms of memory and performance.
  std::vector<TimeRange> ranges_to_draw = selected_time_ranges_;
  for (size_t i = 0; i < ranges_to_draw.size(); ++i) {
    DrawSelectedTimeRange(ranges_to_draw[i], timeline_width,
                          px_per_time_unit_val,
                          /*show_delete_button=*/true, /*range_index=*/i);
  }

  if (current_selected_time_range_) {
    DrawSelectedTimeRange(*current_selected_time_range_, timeline_width,
                          px_per_time_unit_val, /*show_delete_button=*/false);
  }
}

bool Timeline::HandleKeyboard() {
  const ImGuiIO& io = ImGui::GetIO();
  bool is_interacting = false;

  // Pan left ('a', or LeftArrow when no event is selected)
  if (ImGui::IsKeyDown(ImGuiKey_A) ||
      (selected_event_index_ == -1 && ImGui::IsKeyDown(ImGuiKey_LeftArrow))) {
    ImGuiKey key =
        ImGui::IsKeyDown(ImGuiKey_A) ? ImGuiKey_A : ImGuiKey_LeftArrow;
    float multiplier =
        GetSpeedMultiplier(io, key, kShiftPanAccelerateFactor);
    Pan(-panning_speed_ * io.DeltaTime * multiplier);
    is_interacting = true;
  }
  // Pan right ('d', 'e', or RightArrow when no event is selected)
  if (ImGui::IsKeyDown(ImGuiKey_D) || ImGui::IsKeyDown(ImGuiKey_E) ||
      (selected_event_index_ == -1 && ImGui::IsKeyDown(ImGuiKey_RightArrow))) {
    ImGuiKey key =
        ImGui::IsKeyDown(ImGuiKey_D)
            ? ImGuiKey_D
            : (ImGui::IsKeyDown(ImGuiKey_E) ? ImGuiKey_E : ImGuiKey_RightArrow);
    float multiplier = GetSpeedMultiplier(io, key, kShiftPanAccelerateFactor);
    Pan(panning_speed_ * io.DeltaTime * multiplier);
    is_interacting = true;
  }

  // Unlike panning and zooming, vertical scrolling does not affect the visible
  // time range, so it doesn't need to pause data requests by setting
  // is_interacting = true.
  // Scroll up
  if (ImGui::IsKeyDown(ImGuiKey_UpArrow)) {
    Scroll(-kScrollSpeed * io.DeltaTime);
  }
  // Scroll down
  if (ImGui::IsKeyDown(ImGuiKey_DownArrow)) {
    Scroll(kScrollSpeed * io.DeltaTime);
  }

  // Zoom in ('w' or ',')
  if (ImGui::IsKeyDown(ImGuiKey_W) || ImGui::IsKeyDown(ImGuiKey_Comma)) {
    ImGuiKey key = ImGui::IsKeyDown(ImGuiKey_W) ? ImGuiKey_W : ImGuiKey_Comma;
    float multiplier = GetSpeedMultiplier(io, key, kShiftZoomAccelerateFactor);
    Zoom(1.0f - zoom_speed_ * io.DeltaTime * multiplier);
    is_interacting = true;
  }
  // Zoom out ('s' or 'o')
  if (ImGui::IsKeyDown(ImGuiKey_S) || ImGui::IsKeyDown(ImGuiKey_O)) {
    ImGuiKey key = ImGui::IsKeyDown(ImGuiKey_S) ? ImGuiKey_S : ImGuiKey_O;
    float multiplier = GetSpeedMultiplier(io, key, kShiftZoomAccelerateFactor);
    Zoom(1.0f + zoom_speed_ * io.DeltaTime * multiplier);
    is_interacting = true;
  }

  // Cancel selection or resize
  if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    if (is_selecting_ || time_range_resizing_state_.has_value()) {
      if (is_selecting_) {
        is_selecting_ = false;
      }
      if (time_range_resizing_state_.has_value()) {
        time_range_resizing_state_.reset();
      }

      // Common cancel actions
      is_dragging_ = false;
      current_selected_time_range_.reset();
    }
  }

  // Toggle grid lines ('g')
  if (ImGui::IsKeyPressed(ImGuiKey_G)) {
    show_grid_ = !show_grid_;
    is_interacting = true;
  }

  // Zoom to selection ('f')
  if (ImGui::IsKeyPressed(ImGuiKey_F)) {
    if (selected_event_index_ != -1) {
      ZoomEvent(selected_event_index_);
      is_interacting = true;
    }
  }

  // Reset viewport zoom ('0')
  if (ImGui::IsKeyPressed(ImGuiKey_0)) {
    SetVisibleRange(data_time_range_, /*animate=*/true);
    is_interacting = true;
  }

  // Mark interest range ('m')
  if (ImGui::IsKeyPressed(ImGuiKey_M)) {
    const std::optional<TimeRange> target_range = GetSelectionTimeRange();
    if (target_range.has_value()) {
      auto it = absl::c_find(selected_time_ranges_, *target_range);
      if (it != selected_time_ranges_.end()) {
        selected_time_ranges_.erase(it);
      } else {
        selected_time_ranges_.push_back(*target_range);
      }
      is_interacting = true;
    } else if (!selected_time_ranges_.empty()) {
      // When no event is selected, pressing 'm' clears all marked interest
      // ranges.
      selected_time_ranges_.clear();
      is_interacting = true;
    }
  }

  // Select previous event ('<-' when event selected, or 'Shift+Tab')
  if ((selected_event_index_ != -1 &&
       ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) ||
      (ImGui::IsKeyPressed(ImGuiKey_Tab) && io.KeyShift)) {
    SelectPreviousEvent();
    is_interacting = true;
  }

  // Select next event ('->' when event selected, or 'Tab')
  if ((selected_event_index_ != -1 &&
       ImGui::IsKeyPressed(ImGuiKey_RightArrow)) ||
      (ImGui::IsKeyPressed(ImGuiKey_Tab) && !io.KeyShift)) {
    SelectNextEvent();
    is_interacting = true;
  }

  // Mouse Mode shortcuts
  if (ImGui::IsKeyPressed(ImGuiKey_1)) {
    mouse_mode_ = MouseMode::kSelect;
    EmitMouseModeChanged();
  }
  if (ImGui::IsKeyPressed(ImGuiKey_2)) {
    mouse_mode_ = MouseMode::kPan;
    EmitMouseModeChanged();
  }
  if (ImGui::IsKeyPressed(ImGuiKey_3)) {
    mouse_mode_ = MouseMode::kZoom;
    EmitMouseModeChanged();
  }
  if (ImGui::IsKeyPressed(ImGuiKey_4)) {
    mouse_mode_ = MouseMode::kTiming;
    EmitMouseModeChanged();
  }

  return is_interacting;
}

bool Timeline::HandleWheel() {
  const ImGuiIO& io = ImGui::GetIO();

  if (io.MouseWheel == 0.0f && io.MouseWheelH == 0.0f) {
    return false;
  }

  if (io.KeyCtrl || io.KeySuper) {
    // If the mouse wheel is being used with the control or command key, zoom
    // in or out.
    const float zoom_factor = 1.0f + io.MouseWheel * mouse_wheel_zoom_speed_;
    Zoom(zoom_factor);
    return true;
  }

  const Pixel horizontal_pan_delta =
      io.KeyShift ? io.MouseWheel : io.MouseWheelH;
  const Pixel vertical_scroll_delta =
      io.KeyShift ? io.MouseWheelH : io.MouseWheel;

  if (horizontal_pan_delta != 0.0f) Pan(horizontal_pan_delta);
  if (vertical_scroll_delta != 0.0f) Scroll(vertical_scroll_delta);

  return true;
}

// Checks if there is a pending request to vertically scroll to a specific
// event, and sets next window scroll to make it visible if it's out of view.
void Timeline::ProcessPendingScroll() {
  // Check if there is a pending request to scroll to an event.
  if (event_index_to_scroll_to_ < 0) return;

  int level = timeline_data_.entry_levels[event_index_to_scroll_to_];
  if (level < 0 || level >= visible_level_offsets_.size()) return;

  Pixel y_center = visible_level_offsets_[level];
  Pixel y_top = y_center - kEventHeight * 0.5f;
  Pixel y_bottom = y_center + kEventHeight * 0.5f;
  Pixel window_height = ImGui::GetWindowHeight();
  Pixel current_scroll_y = ImGui::GetScrollY();

  // Default target scroll position is current scroll position.
  Pixel target_scroll_y = current_scroll_y;

  // Check if the event is already fully visible.
  bool is_fully_visible = (y_top >= current_scroll_y) &&
                          (y_bottom <= current_scroll_y + window_height);

  if (is_fully_visible) {
    // Event is fully visible, no need to scroll.
  } else if (y_top < current_scroll_y) {
    // Case A: Event is above the current viewport, scroll up until top is
    // visible.
    target_scroll_y = y_top;
  } else if (y_bottom > current_scroll_y + window_height) {
    // Case B: Event is below the current viewport, scroll down until bottom
    // is visible.
    target_scroll_y = y_bottom - window_height;
  }

  // Ensure scroll value is not negative.
  target_scroll_y = std::max(0.0f, target_scroll_y);

  // Trigger scroll if target position changed.
  if (target_scroll_y != current_scroll_y) {
    ImGui::SetScrollY(target_scroll_y);
    // Request a redraw to ensure the UI updates immediately after scrolling.
    if (redraw_callback_) redraw_callback_();
  }

  // Reset request flag to prevent repeated scrolling.
  event_index_to_scroll_to_ = -1;
}

void Timeline::HandleEventDeselection() {
  const bool has_single_selection =
      (selected_event_index_ != -1 || selected_group_index_ != -1);
  const bool has_rectangle_selection =
      (!selected_event_indices_.empty() || !selected_counter_points_.empty());

  // If an event was selected, and the user clicks on an empty area
  // (i.e., not on any event), deselect the event.
  if ((has_single_selection || has_rectangle_selection) &&
      ImGui::IsMouseReleased(0) &&
      ImGui::IsWindowHovered(ImGuiHoveredFlags_ChildWindows) &&
      !event_clicked_this_frame_) {
    bool is_click = true;
    if (selection_start_pos_) {
      const float dx = ImGui::GetIO().MousePos.x - selection_start_pos_->x;
      const float dy = ImGui::GetIO().MousePos.y - selection_start_pos_->y;
      const float distance_squared = dx * dx + dy * dy;
      if (distance_squared > kClickDistanceThresholdSquared) {
        is_click = false;
      }
    }
    if (is_click) {
      if (has_single_selection) {
        selected_event_index_ = -1;
        selected_group_index_ = -1;
        selected_counter_index_ = -1;
        selected_counter_bundle_ = -1;

        EventData event_data;
        event_data.try_emplace(kEventSelectedIndex, -1);
        event_data.try_emplace(kEventSelectedName, std::string());
        event_data.try_emplace(kEventSelectedStart, 0.0);
        event_data.try_emplace(kEventSelectedDuration, 0.0);
        event_data.try_emplace(kEventSelectedStartFormatted, std::string());
        event_data.try_emplace(kEventSelectedDurationFormatted, std::string());

        event_callback_(kEventSelected, event_data);
      }

      if (has_rectangle_selection) {
        selected_event_indices_.clear();
        selected_counter_points_.clear();
        CalculateAndEmitMetrics();
      }

      if (redraw_callback_) redraw_callback_();
    }
  }
}

bool Timeline::HandleMouse() {
  const ImRect timeline_area = GetTimelineArea();
  const bool is_mouse_over_timeline =
      ImGui::IsMouseHoveringRect(timeline_area.Min, timeline_area.Max);

  if (time_range_resizing_state_.has_value()) {
    ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
  } else if (is_mouse_over_timeline) {
    switch (mouse_mode_) {
      case MouseMode::kSelect:
        ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
        break;
      case MouseMode::kTiming:
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeEW);
        break;
      case MouseMode::kPan:
        if (ImGui::IsMouseDown(0)) {
          ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeAll);
        } else {
          ImGui::SetMouseCursor(ImGuiMouseCursor_Arrow);
        }
        break;
      case MouseMode::kZoom:
        ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
        break;
      default:
        break;
    }
  }

  if (!is_mouse_over_timeline && !is_dragging_) {
    return false;
  }

  if (is_mouse_over_timeline) {
    HandleMouseDown(timeline_area.Min.x);
  }

  if (is_dragging_) {
    HandleMouseDrag(timeline_area.Min.x);
    HandleMouseRelease();
    return true;
  }

  return false;
}

void Timeline::HandleMouseDown(Pixel timeline_origin_x) {
  // ImGui uses 0 to represent the left mouse button, as defined in the
  // ImGuiMouseButton enum. We check if the left mouse button was clicked.
  if (ImGui::IsMouseClicked(0) && !event_clicked_this_frame_) {
    is_dragging_ = true;
    ImGuiIO& io = ImGui::GetIO();
    selection_start_pos_ = io.MousePos;

    // Check if we are starting a resize operation on an existing range.
    const double px_per_time = px_per_time_unit();
    if (px_per_time > 0) {
      for (size_t i = 0; i < selected_time_ranges_.size(); ++i) {
        const auto& range = selected_time_ranges_[i];
        const Pixel x_start =
            TimeToScreenX(range.start(), timeline_origin_x, px_per_time);
        const Pixel x_end =
            TimeToScreenX(range.end(), timeline_origin_x, px_per_time);

        if (std::abs(io.MousePos.x - x_start) <= kSelectionEdgeThreshold) {
          time_range_resizing_state_ = {i, /*is_start_edge=*/true};
          return;
        } else if (std::abs(io.MousePos.x - x_end) <= kSelectionEdgeThreshold) {
          time_range_resizing_state_ = {i, /*is_start_edge=*/false};
          return;
        }
      }
    }

    if (mouse_mode_ == MouseMode::kSelect && !is_selecting_) {
      is_selecting_ = true;
      selection_end_pos_ = io.MousePos;
      selected_event_indices_.clear();
      selected_counter_points_.clear();
    } else if (mouse_mode_ != MouseMode::kSelect) {
      is_selecting_ = io.KeyShift || mouse_mode_ == MouseMode::kTiming;
      if (is_selecting_) {
        const double px_per_time = px_per_time_unit();
        drag_start_time_ =
            PixelToTime(io.MousePos.x - timeline_origin_x, px_per_time);
        current_selected_time_range_ =
            TimeRange(drag_start_time_, drag_start_time_);
      }
    }
  }
}

void Timeline::HandleMouseDrag(Pixel timeline_origin_x) {
  // ImGui uses 0 to represent the left mouse button, as defined in the
  // ImGuiMouseButton enum. We check if the left mouse button was clicked.
  if (ImGui::IsMouseDown(0)) {
    ImGuiIO& io = ImGui::GetIO();
    if (time_range_resizing_state_.has_value() &&
        time_range_resizing_state_->range_index <
            selected_time_ranges_.size()) {
      const double px_per_time = px_per_time_unit();
      Microseconds current_time =
          PixelToTime(io.MousePos.x - timeline_origin_x, px_per_time);

      auto& range =
          selected_time_ranges_[time_range_resizing_state_->range_index];
      Microseconds start = range.start();
      Microseconds end = range.end();

      if (time_range_resizing_state_->is_start_edge) {
        start = current_time;
      } else {
        end = current_time;
      }

      if (start > end) {
        time_range_resizing_state_->is_start_edge =
            !time_range_resizing_state_->is_start_edge;
        std::swap(start, end);
      } else if (start == end) {
        // If they exactly cross over on this pixel, we nudge one by a
        // microscopic amount so that duration doesn't become zero and vanish
        // from rendering. It also breaks the exact symmetry so it doesn't get
        // stuck rendering 0 width.
        end += current_time * 0.000000001;
        if (start > end) {
          time_range_resizing_state_->is_start_edge =
              !time_range_resizing_state_->is_start_edge;
          std::swap(start, end);
        }
      }
      TimeRange new_range(start, end);
      ApplySnapping(new_range);
      range = new_range;
    } else if (is_selecting_) {
      if (mouse_mode_ == MouseMode::kSelect) {
        selection_end_pos_ = io.MousePos;
      } else {
        const double px_per_time = px_per_time_unit();
        Microseconds current_time =
            PixelToTime(io.MousePos.x - timeline_origin_x, px_per_time);
        current_selected_time_range_ =
            TimeRange(std::min(drag_start_time_, current_time),
                      std::max(drag_start_time_, current_time));
        ApplySnapping(*current_selected_time_range_);
      }
    } else if (mouse_mode_ == MouseMode::kZoom) {
      Zoom(1.0f + io.MouseDelta.y * 0.01f);
    } else {
      Pan(-io.MouseDelta.x);
      Scroll(-io.MouseDelta.y);
    }
  }
}

void Timeline::DrawToast(absl::string_view message, float& timer,
                         float base_y_offset) {
  if (timer <= 0.0f) return;

  timer -= ImGui::GetIO().DeltaTime;
  if (redraw_callback_) redraw_callback_();

  const ImGuiViewport* main_viewport = ImGui::GetMainViewport();
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();

  const ImVec2 text_size =
      ImGui::CalcTextSize(message.data(), message.data() + message.size());
  const ImVec2 padding(16.0f, 8.0f);
  const ImVec2 toast_size(text_size.x + padding.x * 2.0f,
                          text_size.y + padding.y * 2.0f);

  const ImVec2 toast_pos(
      main_viewport->Pos.x + (main_viewport->Size.x - toast_size.x) * 0.5f,
      main_viewport->Pos.y + main_viewport->Size.y - toast_size.y -
          base_y_offset);

  // Fade out at the end.
  const float alpha = std::max(0.0f, std::min(1.0f, timer * 2.0f));
  const ImU32 bg_color =
      IM_COL32(32, 33, 36, (int)(230.0f * alpha));  // Dark grey
  const ImU32 text_color = IM_COL32(255, 255, 255, (int)(255.0f * alpha));

  draw_list->AddRectFilled(
      toast_pos, ImVec2(toast_pos.x + toast_size.x, toast_pos.y + toast_size.y),
      bg_color, kToastCornerRounding);
  draw_list->AddText(ImVec2(toast_pos.x + padding.x, toast_pos.y + padding.y),
                     text_color, message.data(),
                     message.data() + message.size());
}

void Timeline::HandleMouseRelease() {
  if (ImGui::IsMouseReleased(0)) {
    is_dragging_ = false;
    is_selecting_ = false;
    time_range_resizing_state_.reset();

    bool is_click = false;
    if (selection_start_pos_) {
      const float dx = ImGui::GetIO().MousePos.x - selection_start_pos_->x;
      const float dy = ImGui::GetIO().MousePos.y - selection_start_pos_->y;
      const float distance_squared = dx * dx + dy * dy;
      if (distance_squared <= kClickDistanceThresholdSquared) {
        is_click = true;
      }
    }

    // If a bookmark was handled (e.g., added via Ctrl+Click), bypass
    // selection or time range additions. This prevents accidentally adding
    // tiny time ranges if the user slightly moved the mouse (within
    // threshold) during the click.
    if (!HandleBookmarkAddition(is_click)) {
      HandleSelectionOrTimeRangeAddition();
      if (redraw_callback_)
        redraw_callback_();  // Trigger redraw to show points
    }

    selection_start_pos_.reset();
    selection_end_pos_.reset();
    current_selected_time_range_.reset();
  }
}

bool Timeline::HandleBookmarkAddition(bool is_click) {
  if (bookmarks_enabled_ && is_click &&
      (ImGui::GetIO().KeyCtrl || ImGui::GetIO().KeySuper)) {
    const ImRect timeline_area = GetTimelineArea();
    if (ImGui::IsMouseHoveringRect(timeline_area.Min, timeline_area.Max)) {
      const Microseconds time = PixelToTime(
          ImGui::GetMousePos().x - timeline_area.Min.x, px_per_time_unit());
      AddBookmark(time);
      return true;
    }
  }
  return false;
}

bool Timeline::HandleSelectionOrTimeRangeAddition() {
  if (mouse_mode_ == MouseMode::kSelect && selection_start_pos_ &&
      selection_end_pos_) {
    const float dx = selection_end_pos_->x - selection_start_pos_->x;
    const float dy = selection_end_pos_->y - selection_start_pos_->y;
    const float distance_squared = dx * dx + dy * dy;
    if (distance_squared > kClickDistanceThresholdSquared) {
      ImRect selection_rect =
          ImRect(std::min(selection_start_pos_->x, selection_end_pos_->x),
                 std::min(selection_start_pos_->y, selection_end_pos_->y),
                 std::max(selection_start_pos_->x, selection_end_pos_->x),
                 std::max(selection_start_pos_->y, selection_end_pos_->y));
      FindSelectedEvents(selection_rect);
      CalculateAndEmitMetrics();
      if (redraw_callback_)
        redraw_callback_();  // Trigger redraw to show points
      return true;
    } else if (!event_clicked_this_frame_) {
      // Simple click in Select mode clears rectangle selection in UI.
      CalculateAndEmitMetrics();
    }
  } else if (current_selected_time_range_ &&
             current_selected_time_range_->duration() > 0) {
    selected_time_ranges_.push_back(*current_selected_time_range_);
    if (!event_tooltip_enabled_ && event_callback_) {
      const Microseconds start_us = current_selected_time_range_->start();
      const Microseconds dur_us = current_selected_time_range_->duration();
      const std::string json = absl::StrFormat(
          R"({"selectionStartUs":%.1f,"selectionExtentUs":%.1f})", start_us,
          dur_us);
      EventData event_data;
      event_data.try_emplace(kEventsSelectedData, json);
      PopulateMultiSelectionScheduleDetails(start_us, dur_us, event_data);
      event_callback_(kEventsSelected, event_data);
    }
    return true;
  }
  return false;
}

ImRect Timeline::GetTimelineArea() const {
  const ImVec2 window_pos = ImGui::GetWindowPos();
  const ImVec2 content_min = ImGui::GetWindowContentRegionMin();
  const Pixel start_x = window_pos.x + content_min.x + label_width_;
  const Pixel start_y = window_pos.y + content_min.y;

  const Pixel end_x = start_x + current_timeline_width_;
  const Pixel end_y = window_pos.y + ImGui::GetWindowHeight();

  return {start_x, start_y, end_x, end_y};
}

void Timeline::InitializeLastFetchRequestRange(const TimeRange& visible_range) {
  TimeRange fetch = visible_range.Scale(kFetchRatio);

  if (fetch.duration() < kMinFetchDurationMicros) {
    Microseconds center = fetch.center();
    fetch = {center - kMinFetchDurationMicros / 2.0,
             center + kMinFetchDurationMicros / 2.0};
  }

  ConstrainTimeRange(fetch);
  last_fetch_request_range_ = fetch;
}

void Timeline::MaybeRequestData() {
  // Don't request more data if a request is already in flight or if the entire
  // trace is already loaded at full resolution in memory.
  if (is_incremental_loading_ || (full_trace_loaded_ && !is_compacted_)) return;

  // Hosts whose backend always returns the full dataset opt out entirely, as
  // does full-fidelity mode when the complete trace is resident within budget.
  if (!incremental_fetch_enabled_ || HasFullFidelityStoreWithinBudget()) return;

  // We have several ranges of interest for incremental loading:
  //
  // |-----------data_time_range_---------|                  Full trace duration
  //     |---------fetch----------|         Amount to fetch on load (viewport*3)
  //         |----preserve----|               Buffer to keep loaded (viewport*2)
  //             |viewport|              aka current_visible: On-screen viewport
  //
  // If 'preserve' isn't contained in 'last_fetch_request_range_', or resolution
  // is too coarse, a new load of 'fetch' range is triggered.
  // - last_fetch_request_range_: The time range covered by data currently
  // loaded.
  const TimeRange current_visible = visible_range();

  TimeRange preserve = current_visible.Scale(kPreserveRatio);
  TimeRange fetch = current_visible.Scale(kFetchRatio);

  if (fetch.duration() < kMinFetchDurationMicros) {
    Microseconds center = fetch.center();
    fetch = {center - kMinFetchDurationMicros / 2.0,
             center + kMinFetchDurationMicros / 2.0};
  }

  // Constrain the ranges to the valid data range. This ensures that we don't
  // try to fetch data outside the available trace duration (e.g. negative time
  // or future time), preventing infinite refetch loops at the boundaries.
  ConstrainTimeRange(preserve);
  ConstrainTimeRange(fetch);

  // Refetch data if user zoomed in significantly, making resolution too coarse.
  // We use last_fetch_request_range_ here because it reflects the density of
  // the data we hold (asked for), whereas fetched_data_time_range_ might be
  // smaller/sparse depending on actual event content.
  const bool zoomed_in_too_much =
      (last_fetch_request_range_.duration() / fetch.duration() >
       kRefetchZoomRatio);
  const bool needs_full_resolution_window =
      is_compacted_ && !window_at_full_resolution_ &&
      data_time_range_.duration() > 0 &&
      fetch.duration() <= 0.85 * data_time_range_.duration();

  // Utilize the last fetch request range as a guard to prevent redundant
  // fetches only if we don't need higher resolution data.
  //
  // If `last_fetch_request_range_` contains `preserve`, it means we've already
  // successfully requested a superset of the required data.
  // EXCEPT IF `zoomed_in_too_much` is true: in that case, even if we have the
  // range, the data density might be too low, so we MUST refetch to get
  // higher-resolution data.
  if (!zoomed_in_too_much && !needs_full_resolution_window &&
      last_fetch_request_range_.Contains(preserve)) {
    return;
  }

  if (saved_full_overview_.has_value() && data_time_range_.duration() > 0 &&
      fetch.duration() >= 0.9 * data_time_range_.duration()) {
    RestoreSavedFullOverview();
    return;
  }

  if (data_time_range_.duration() > 0 &&
      fetch.duration() < 0.9 * data_time_range_.duration()) {
    PrepareForFilteredRefetch();
  }

  EventData event_data;
  event_data.try_emplace(kFetchDataStart, MicrosToMillis(fetch.start()));
  event_data.try_emplace(kFetchDataEnd, MicrosToMillis(fetch.end()));

  // Update last_fetch_request_range_ and is_incremental_loading_ BEFORE
  // invoking event_callback_ so synchronous callbacks that call
  // SetTimelineData / set_fetched_data_time_range / set_is_incremental_loading
  // are not overwritten afterward.
  last_fetch_request_range_ = fetch;
  is_incremental_loading_ = true;

  event_callback_(kFetchData, event_data);
}

EntryBitset Timeline::EvaluateEventFilter(const EventFilter& filter) const {
  const size_t total_events = timeline_data_.entry_start_times.size();
  EntryBitset matches;
  matches.Reserve(total_events);
  if (total_events == 0) {
    return matches;
  }
  if (filter.IsEmpty()) {
    for (size_t i = 0; i < total_events; ++i) {
      matches.insert(static_cast<int>(i));
    }
    return matches;
  }

  const bool use_interned_names =
      !timeline_data_.interned_string_pool.empty() &&
      timeline_data_.entry_name_indices.size() == total_events;
  EntryBitset matching_name_idx;
  if (!filter.name_substring.empty() && use_interned_names) {
    const size_t pool_size = timeline_data_.interned_string_pool.size();
    matching_name_idx.Reserve(pool_size);
    for (size_t s = 0; s < pool_size; ++s) {
      const absl::string_view candidate =
          timeline_data_.interned_string_pool[s];
      const bool name_ok =
          filter.exact_name_match
              ? absl::EqualsIgnoreCase(candidate, filter.name_substring)
              : absl::StrContainsIgnoreCase(candidate, filter.name_substring);
      if (name_ok) {
        matching_name_idx.insert(static_cast<int>(s));
      }
    }
    if (matching_name_idx.empty()) {
      return matches;
    }
  }

  auto matches_event_predicates = [&](int i) -> bool {
    if (i < 0 || static_cast<size_t>(i) >= total_events ||
        static_cast<size_t>(i) >= timeline_data_.entry_total_times.size()) {
      return false;
    }
    if (!filter.name_substring.empty()) {
      if (use_interned_names) {
        const uint32_t name_idx = timeline_data_.entry_name_indices[i];
        if (!matching_name_idx.contains(static_cast<int>(name_idx))) {
          return false;
        }
      } else {
        const absl::string_view name = GetEventName(i);
        const bool name_ok =
            filter.exact_name_match
                ? absl::EqualsIgnoreCase(name, filter.name_substring)
                : absl::StrContainsIgnoreCase(name, filter.name_substring);
        if (!name_ok) {
          return false;
        }
      }
    }
    const Microseconds start = timeline_data_.entry_start_times[i];
    const Microseconds dur = timeline_data_.entry_total_times[i];
    const Microseconds end = start + dur;
    if (filter.time_window.has_value()) {
      if (end < filter.time_window->start() ||
          start > filter.time_window->end()) {
        return false;
      }
    }
    if (filter.min_duration.has_value() && dur < *filter.min_duration) {
      return false;
    }
    if (filter.max_duration.has_value() && dur > *filter.max_duration) {
      return false;
    }
    if (!filter.hlo_op_substring.empty()) {
      const absl::string_view hlo_op =
          timeline_data_.GetEntryHloOp(static_cast<size_t>(i));
      if (!absl::StrContainsIgnoreCase(hlo_op, filter.hlo_op_substring)) {
        return false;
      }
    }
    if (!filter.hlo_module_substring.empty()) {
      const absl::string_view hlo_module =
          timeline_data_.GetEntryHloModule(static_cast<size_t>(i));
      if (!absl::StrContainsIgnoreCase(hlo_module,
                                       filter.hlo_module_substring)) {
        return false;
      }
    }
    return true;
  };

  const int num_levels = timeline_data_.total_levels();
  if (num_levels > 0 &&
      timeline_data_.level_event_indices.size() == total_events) {
    const int min_lvl =
        filter.min_level.has_value() ? std::max(0, *filter.min_level) : 0;
    const int max_lvl = filter.max_level.has_value()
                            ? std::min(num_levels - 1, *filter.max_level)
                            : (num_levels - 1);
    for (int level = min_lvl; level <= max_lvl; ++level) {
      if (filter.pid.has_value() || filter.tid.has_value() ||
          !filter.track_name_substring.empty()) {
        const Group* grp = timeline_data_.FindGroupForLevel(level);
        if (grp == nullptr) continue;
        if (filter.pid.has_value() && grp->pid != *filter.pid) continue;
        if (filter.tid.has_value() && grp->tid != *filter.tid) continue;
        if (!filter.track_name_substring.empty() &&
            !absl::StrContainsIgnoreCase(grp->name,
                                         filter.track_name_substring)) {
          continue;
        }
      }
      const absl::Span<const int> indices = timeline_data_.level_events(level);
      if (indices.empty()) continue;

      auto it = indices.begin();
      if (filter.time_window.has_value()) {
        const Microseconds w_start = filter.time_window->start();
        it = std::lower_bound(
            indices.begin(), indices.end(), w_start,
            [this, total_events](int event_idx, Microseconds t) {
              if (event_idx < 0 ||
                  static_cast<size_t>(event_idx) >= total_events ||
                  static_cast<size_t>(event_idx) >=
                      timeline_data_.entry_total_times.size()) {
                return true;
              }
              const Microseconds end =
                  timeline_data_.entry_start_times[event_idx] +
                  timeline_data_.entry_total_times[event_idx];
              return end < t;
            });
      }

      for (; it != indices.end(); ++it) {
        const int i = *it;
        if (i < 0 || static_cast<size_t>(i) >= total_events) continue;
        if (filter.time_window.has_value() &&
            timeline_data_.entry_start_times[i] > filter.time_window->end()) {
          break;
        }
        if (matches_event_predicates(i)) {
          matches.insert(i);
        }
      }
    }
    return matches;
  }

  for (size_t idx = 0; idx < total_events; ++idx) {
    const int i = static_cast<int>(idx);
    const int level = (idx < timeline_data_.entry_levels.size())
                          ? timeline_data_.entry_levels[idx]
                          : 0;
    if (filter.min_level.has_value() && level < *filter.min_level) continue;
    if (filter.max_level.has_value() && level > *filter.max_level) continue;
    if (filter.pid.has_value() || filter.tid.has_value() ||
        !filter.track_name_substring.empty()) {
      const Group* grp = timeline_data_.FindGroupForLevel(level);
      if (grp == nullptr) continue;
      if (filter.pid.has_value() && grp->pid != *filter.pid) continue;
      if (filter.tid.has_value() && grp->tid != *filter.tid) continue;
      if (!filter.track_name_substring.empty() &&
          !absl::StrContainsIgnoreCase(grp->name,
                                       filter.track_name_substring)) {
        continue;
      }
    }
    if (matches_event_predicates(i)) {
      matches.insert(i);
    }
  }
  return matches;
}

void Timeline::ApplyEventFilter(const EventFilter& filter) {
  if (filter.IsEmpty()) {
    ClearEventFilter();
    return;
  }
  active_event_filter_ = filter;
  has_active_event_filter_ = true;
  filtered_event_indices_ = EvaluateEventFilter(filter);
  ++filter_epoch_;
  process_utilization_pyramids_dirty_ = true;
  if (!search_query_lower_.empty()) {
    search_results_.clear();
    matching_event_indices_.clear();
    current_search_result_index_ = -1;
    RecomputeSearchResults();
  } else if (redraw_callback_) {
    redraw_callback_();
  }
}

void Timeline::ClearEventFilter() {
  if (!has_active_event_filter_ && !active_event_filter_.has_value()) {
    return;
  }
  active_event_filter_.reset();
  has_active_event_filter_ = false;
  filtered_event_indices_.release();
  ++filter_epoch_;
  process_utilization_pyramids_dirty_ = true;
  if (!search_query_lower_.empty()) {
    search_results_.clear();
    matching_event_indices_.clear();
    current_search_result_index_ = -1;
    RecomputeSearchResults();
  } else if (redraw_callback_) {
    redraw_callback_();
  }
}

void Timeline::SetSearchQuery(absl::string_view query) {
  const std::string new_query_lower = absl::AsciiStrToLower(query);
  if (search_query_lower_ == new_query_lower) {
    if (redraw_callback_) redraw_callback_();
    return;
  }
  search_query_lower_ = new_query_lower;
  pending_navigation_event_id_.reset();
  search_results_.clear();
  matching_event_indices_.clear();
  current_search_result_index_ = -1;

  if (query.empty()) {
    if (redraw_callback_) redraw_callback_();
    return;
  }

  RecomputeSearchResults();
}

void Timeline::RecomputeSearchResults() {
  const bool use_csr_interned = !timeline_data_.interned_string_pool.empty() &&
                                !timeline_data_.level_offsets.empty() &&
                                timeline_data_.level_event_indices.size() ==
                                    timeline_data_.entry_name_indices.size();

  if (use_csr_interned) {
    const size_t pool_size = timeline_data_.interned_string_pool.size();
    EntryBitset matching_name_idx;
    matching_name_idx.Reserve(pool_size);
    for (size_t s = 0; s < pool_size; ++s) {
      if (absl::StrContainsIgnoreCase(timeline_data_.interned_string_pool[s],
                                      search_query_lower_)) {
        matching_name_idx.insert(static_cast<int>(s));
      }
    }

    const size_t n_entries = timeline_data_.entry_name_indices.size();
    const bool hlo_op_key_matches =
        absl::StrContainsIgnoreCase(kHloOp, search_query_lower_);
    EntryBitset matching_hlo_op_idx;
    if (!timeline_data_.hlo_op_table.empty()) {
      matching_hlo_op_idx.Reserve(timeline_data_.hlo_op_table.size());
      for (size_t s = 1; s < timeline_data_.hlo_op_table.size(); ++s) {
        if (hlo_op_key_matches ||
            absl::StrContainsIgnoreCase(timeline_data_.hlo_op_table[s],
                                        search_query_lower_)) {
          matching_hlo_op_idx.insert(static_cast<int>(s));
        }
      }
    }

    const bool hlo_mod_key_matches =
        absl::StrContainsIgnoreCase(kHloModule, search_query_lower_);
    EntryBitset matching_hlo_mod_idx;
    if (!timeline_data_.hlo_module_table.empty()) {
      matching_hlo_mod_idx.Reserve(timeline_data_.hlo_module_table.size());
      for (size_t s = 1; s < timeline_data_.hlo_module_table.size(); ++s) {
        const std::string& mod = timeline_data_.hlo_module_table[s];
        if (!mod.empty() && mod != kHloModuleDefault &&
            (hlo_mod_key_matches ||
             absl::StrContainsIgnoreCase(mod, search_query_lower_))) {
          matching_hlo_mod_idx.insert(static_cast<int>(s));
        }
      }
    }

    EntryBitset matching_arg_events;
    auto mark_arg_event = [&](size_t idx) {
      if (idx < n_entries) {
        if (matching_arg_events.empty()) {
          matching_arg_events.Reserve(n_entries);
        }
        matching_arg_events.insert(static_cast<int>(idx));
      }
    };
    for (size_t idx = 0; idx < timeline_data_.entry_args.size(); ++idx) {
      if (MatchesArgsMap(timeline_data_.entry_args[idx], search_query_lower_)) {
        mark_arg_event(idx);
      }
    }
    for (const auto& [idx, args_map] : timeline_data_.sparse_entry_args) {
      if (MatchesArgsMap(args_map, search_query_lower_)) {
        mark_arg_event(idx);
      }
    }
    if (!timeline_data_.non_zero_uids_by_event_index.empty()) {
      const bool uid_key_matches =
          absl::StrContainsIgnoreCase("uid", search_query_lower_);
      for (const auto& [idx, uid] :
           timeline_data_.non_zero_uids_by_event_index) {
        if (idx >= 0 &&
            (uid_key_matches || absl::StrContainsIgnoreCase(
                                    absl::StrCat(uid), search_query_lower_))) {
          mark_arg_event(static_cast<size_t>(idx));
        }
      }
    }
    if (!timeline_data_.sparse_entry_group_ids.empty()) {
      const bool group_key_matches =
          absl::StrContainsIgnoreCase("group_id", search_query_lower_);
      for (const auto& run : timeline_data_.sparse_entry_group_ids.runs()) {
        if (group_key_matches ||
            absl::StrContainsIgnoreCase(absl::StrCat(run.group_id),
                                        search_query_lower_)) {
          const uint32_t end =
              std::min<uint32_t>(run.end, static_cast<uint32_t>(n_entries));
          for (uint32_t idx = run.begin; idx < end; ++idx) {
            mark_arg_event(idx);
          }
        }
      }
    }

    const bool check_hlo_op =
        !matching_hlo_op_idx.empty() || hlo_op_key_matches;
    const bool check_hlo_mod = !matching_hlo_mod_idx.empty();
    const bool check_arg_events = !matching_arg_events.empty();

    if (matching_name_idx.empty() && !check_hlo_op && !check_hlo_mod &&
        !check_arg_events) {
      if (redraw_callback_) redraw_callback_();
      return;
    }

    matching_event_indices_.Reserve(n_entries);
    const int num_levels = timeline_data_.total_levels();
    for (int level = 0; level < num_levels; ++level) {
      const absl::Span<const int> indices = timeline_data_.level_events(level);
      for (const int i : indices) {
        if (i < 0 || !IsEventVisibleUnderFilter(i) ||
            static_cast<size_t>(i) >= n_entries ||
            static_cast<size_t>(i) >= timeline_data_.entry_start_times.size() ||
            static_cast<size_t>(i) >= timeline_data_.entry_total_times.size()) {
          continue;
        }
        const uint32_t name_idx = timeline_data_.entry_name_indices[i];
        bool matched =
            (name_idx < pool_size &&
             matching_name_idx.contains(static_cast<int>(name_idx))) ||
            (check_arg_events && matching_arg_events.contains(i));
        if (!matched && check_hlo_op) {
          const uint32_t op_idx =
              timeline_data_.GetEntryHloOpIndex(static_cast<size_t>(i));
          if (op_idx != 0) {
            if (!timeline_data_.hlo_op_table.empty()) {
              matched = matching_hlo_op_idx.contains(static_cast<int>(op_idx));
            } else {
              matched = hlo_op_key_matches ||
                        (op_idx < pool_size &&
                         matching_name_idx.contains(static_cast<int>(op_idx)));
            }
          }
        }
        if (!matched && check_hlo_mod) {
          const uint32_t mod_idx =
              timeline_data_.GetEntryHloModuleIndex(static_cast<size_t>(i));
          if (mod_idx != 0 &&
              matching_hlo_mod_idx.contains(static_cast<int>(mod_idx))) {
            matched = true;
          }
        }
        if (matched) {
          matching_event_indices_.insert(i);
        }
      }
    }
    if (redraw_callback_) redraw_callback_();
    return;
  }

  const int num_entries =
      !timeline_data_.interned_string_pool.empty()
          ? static_cast<int>(timeline_data_.entry_name_indices.size())
          : static_cast<int>(timeline_data_.entry_names.size());
  for (int i = 0; i < num_entries; ++i) {
    if (!IsEventVisibleUnderFilter(i)) continue;
    if (MatchesSearchQuery(timeline_data_, i, search_query_lower_)) {
      const absl::string_view name = GetEventName(i);
      const EventId event_id = GetEventIdFromData(timeline_data_, i);
      const int level = timeline_data_.entry_levels[i];
      const Microseconds start_time = timeline_data_.entry_start_times[i];
      const Microseconds total_time = timeline_data_.entry_total_times[i];
      ProcessId pid = 0;
      ThreadId tid = 0;
      if (const Group* group = timeline_data_.FindGroupForLevel(level)) {
        pid = group->pid;
        tid = group->tid;
      }
      search_results_.push_back(SearchResult{.event_id = event_id,
                                             .level = level,
                                             .start_time = start_time,
                                             .duration = total_time,
                                             .pid = pid,
                                             .tid = tid,
                                             .name = std::string(name),
                                             .loaded_index = i});
      matching_event_indices_.insert(i);
    }
  }

  // Sort results horizontally by track hierarchy and start time.
  absl::c_sort(search_results_,
               [](const SearchResult& a, const SearchResult& b) {
                 return std::tie(a.level, a.start_time, a.loaded_index) <
                        std::tie(b.level, b.start_time, b.loaded_index);
               });

  if (redraw_callback_) redraw_callback_();
}

const std::vector<Timeline::SearchResult>&
Timeline::get_search_results_for_test() const {
  if (search_results_.empty() && !matching_event_indices_.empty()) {
    search_results_.reserve(matching_event_indices_.size());
    const int num_levels = timeline_data_.total_levels();
    for (int level = 0; level < num_levels; ++level) {
      const absl::Span<const int> indices = timeline_data_.level_events(level);
      if (indices.empty()) continue;
      ProcessId pid = 0;
      ThreadId tid = 0;
      if (const Group* grp = timeline_data_.FindGroupForLevel(level)) {
        pid = grp->pid;
        tid = grp->tid;
      }
      for (const int i : indices) {
        if (!matching_event_indices_.contains(i)) continue;
        if (i < 0 ||
            static_cast<size_t>(i) >= timeline_data_.entry_start_times.size() ||
            static_cast<size_t>(i) >= timeline_data_.entry_total_times.size()) {
          continue;
        }
        search_results_.push_back(
            SearchResult{.event_id = GetEventIdFromData(timeline_data_, i),
                         .level = level,
                         .start_time = timeline_data_.entry_start_times[i],
                         .duration = timeline_data_.entry_total_times[i],
                         .pid = pid,
                         .tid = tid,
                         .name = std::string(GetEventName(i)),
                         .loaded_index = i});
      }
    }
    auto cmp = [](const SearchResult& a, const SearchResult& b) {
      return std::tie(a.level, a.start_time, a.loaded_index) <
             std::tie(b.level, b.start_time, b.loaded_index);
    };
    if (!absl::c_is_sorted(search_results_, cmp)) {
      absl::c_sort(search_results_, cmp);
    }
  }
  return search_results_;
}

int Timeline::GetNthMatchingCsrEventIndex(int target_rank) const {
  if (target_rank < 0 ||
      static_cast<size_t>(target_rank) >= matching_event_indices_.size()) {
    return -1;
  }
  int current_rank = 0;
  const int num_levels = timeline_data_.total_levels();
  for (int level = 0; level < num_levels; ++level) {
    const absl::Span<const int> indices = timeline_data_.level_events(level);
    int level_matches = 0;
    int candidate = -1;
    bool sorted = true;
    Microseconds prev_start = -std::numeric_limits<Microseconds>::infinity();
    int prev_idx = -1;
    for (const int i : indices) {
      if (!matching_event_indices_.contains(i)) continue;
      const Microseconds s =
          (i >= 0 &&
           static_cast<size_t>(i) < timeline_data_.entry_start_times.size())
              ? timeline_data_.entry_start_times[i]
              : 0.0;
      if (std::tie(s, i) < std::tie(prev_start, prev_idx)) {
        sorted = false;
      }
      prev_start = s;
      prev_idx = i;
      if (current_rank + level_matches == target_rank) {
        candidate = i;
      }
      ++level_matches;
    }
    if (target_rank < current_rank + level_matches) {
      if (sorted) return candidate;
      std::vector<int> matched_in_level;
      matched_in_level.reserve(level_matches);
      for (const int i : indices) {
        if (matching_event_indices_.contains(i)) {
          matched_in_level.push_back(i);
        }
      }
      absl::c_sort(matched_in_level, [this](int a, int b) {
        const Microseconds sa =
            (a >= 0 &&
             static_cast<size_t>(a) < timeline_data_.entry_start_times.size())
                ? timeline_data_.entry_start_times[a]
                : 0.0;
        const Microseconds sb =
            (b >= 0 &&
             static_cast<size_t>(b) < timeline_data_.entry_start_times.size())
                ? timeline_data_.entry_start_times[b]
                : 0.0;
        return std::tie(sa, a) < std::tie(sb, b);
      });
      return matched_in_level[target_rank - current_rank];
    }
    current_rank += level_matches;
  }
  return -1;
}

void Timeline::SetSearchResults(const ParsedTraceEvents& search_results) {
  const bool navigation_active = (current_search_result_index_ >= 0);

  search_results_.clear();
  matching_event_indices_.clear();
  current_search_result_index_ = -1;

  if (search_query_lower_.empty()) {
    if (redraw_callback_) redraw_callback_();
    return;
  }

  absl::flat_hash_map<EventId, int> event_id_to_loaded_index;
  absl::flat_hash_map<EventId, int> event_id_to_level;
  const int num_loaded = timeline_data_.entry_event_ids.size();
  event_id_to_loaded_index.reserve(num_loaded);
  event_id_to_level.reserve(num_loaded);
  for (int i = 0; i < num_loaded; ++i) {
    const EventId ev_id = timeline_data_.entry_event_ids[i];
    event_id_to_loaded_index.try_emplace(ev_id, i);
    const int lvl = (i < timeline_data_.entry_levels.size())
                        ? timeline_data_.entry_levels[i]
                        : -1;
    event_id_to_level.try_emplace(ev_id, lvl);
  }

  std::optional<absl::flat_hash_map<FallbackKey, std::vector<int>>>
      lazy_loaded_events;

  for (const TraceEvent& event : search_results.flame_events) {
    if (event.ph != Phase::kComplete && event.ph != Phase::kInstant &&
        event.ph != Phase::kInstantDeprecated) {
      continue;
    }
    const EventId computed_event_id =
        event.event_id != 0 ? event.event_id
                            : GenerateEventId(event.name, event.ts, event.dur);
    int level = -1;
    if (const auto it = event_id_to_level.find(computed_event_id);
        it != event_id_to_level.end()) {
      level = it->second;
    }
    EventId event_id = computed_event_id;
    int loaded_index = -1;
    if (const auto it = event_id_to_loaded_index.find(computed_event_id);
        it != event_id_to_loaded_index.end()) {
      loaded_index = it->second;
    } else {
      if (!lazy_loaded_events.has_value()) {
        absl::flat_hash_set<FallbackKey> wanted_keys;
        wanted_keys.reserve(search_results.flame_events.size());
        for (const TraceEvent& ev : search_results.flame_events) {
          if (ev.ph == Phase::kComplete || ev.ph == Phase::kInstant ||
              ev.ph == Phase::kInstantDeprecated) {
            wanted_keys.insert(
                std::make_tuple(ev.pid, ev.tid, absl::string_view(ev.name)));
          }
        }
        lazy_loaded_events = BuildFallbackMap(timeline_data_, wanted_keys);
      }
      loaded_index = FindFallbackMatchedIndex(
          *lazy_loaded_events, timeline_data_, event.pid, event.tid, event.name,
          event.ts, event.dur);
      if (loaded_index != -1) {
        event_id = GetEventIdFromData(timeline_data_, loaded_index);
        if (static_cast<size_t>(loaded_index) <
            timeline_data_.entry_levels.size()) {
          level = timeline_data_.entry_levels[loaded_index];
        }
      }
    }

    if (loaded_index != -1 && !IsEventVisibleUnderFilter(loaded_index)) {
      continue;
    }

    search_results_.push_back(SearchResult{.event_id = event_id,
                                           .level = level,
                                           .start_time = event.ts,
                                           .duration = event.dur,
                                           .pid = event.pid,
                                           .tid = event.tid,
                                           .name = event.name,
                                           .loaded_index = loaded_index});
    if (loaded_index != -1) {
      matching_event_indices_.insert(loaded_index);
    }
  }

  // Merge local loaded events that match by substring or event arguments, which
  // backend prefix search does not return.
  const int num_entries =
      !timeline_data_.interned_string_pool.empty()
          ? static_cast<int>(timeline_data_.entry_name_indices.size())
          : static_cast<int>(timeline_data_.entry_names.size());
  for (int i = 0; i < num_entries; ++i) {
    if (!IsEventVisibleUnderFilter(i)) {
      continue;
    }
    const absl::string_view name = GetEventName(i);
    if (matching_event_indices_.contains(i) ||
        absl::StartsWithIgnoreCase(name, search_query_lower_)) {
      continue;
    }
    if (MatchesSearchQuery(timeline_data_, i, search_query_lower_)) {
      const EventId event_id = GetEventIdFromData(timeline_data_, i);
      const int level =
          (i < static_cast<int>(timeline_data_.entry_levels.size()))
              ? timeline_data_.entry_levels[i]
              : -1;
      const Microseconds start_time =
          (i < static_cast<int>(timeline_data_.entry_start_times.size()))
              ? timeline_data_.entry_start_times[i]
              : 0.0;
      const Microseconds total_time =
          (i < static_cast<int>(timeline_data_.entry_total_times.size()))
              ? timeline_data_.entry_total_times[i]
              : 0.0;
      ProcessId pid = 0;
      ThreadId tid = 0;
      if (const Group* group = timeline_data_.FindGroupForLevel(level)) {
        pid = group->pid;
        tid = group->tid;
      }
      search_results_.push_back(SearchResult{.event_id = event_id,
                                             .level = level,
                                             .start_time = start_time,
                                             .duration = total_time,
                                             .pid = pid,
                                             .tid = tid,
                                             .name = std::string(name),
                                             .loaded_index = i});
      matching_event_indices_.insert(i);
    }
  }

  if (search_results_.empty()) {
    if (redraw_callback_) redraw_callback_();
    return;
  }

  absl::flat_hash_map<ProcessId, uint32_t> process_sort_indices =
      GetProcessSortIndices(search_results);
  absl::flat_hash_map<std::pair<ProcessId, ThreadId>, uint32_t>
      thread_sort_indices;
  for (const TraceEvent& event : search_results.flame_events) {
    if (event.ph == Phase::kMetadata && event.name == kThreadSortIndex) {
      if (const auto it = event.args.find(kSortIndex); it != event.args.end()) {
        double sort_index_double;
        if (absl::SimpleAtod(it->second, &sort_index_double)) {
          thread_sort_indices[{event.pid, event.tid}] =
              static_cast<uint32_t>(sort_index_double);
        }
      }
    }
  }

  absl::flat_hash_map<std::pair<ProcessId, ThreadId>, int> pid_tid_to_min_level;
  for (const auto& grp : timeline_data_.groups) {
    if (grp.nesting_level == kProcessNestingLevel ||
        grp.type == Group::Type::kCounter) {
      continue;
    }
    if (grp.pid != 0 || grp.tid != 0) {
      auto [it, inserted] =
          pid_tid_to_min_level.try_emplace({grp.pid, grp.tid}, grp.start_level);
      if (grp.start_level < it->second) {
        it->second = grp.start_level;
      }
    }
  }
  for (const auto& res : search_results_) {
    if (res.level != -1) {
      auto [it, inserted] =
          pid_tid_to_min_level.try_emplace({res.pid, res.tid}, res.level);
      if (res.level < it->second) {
        it->second = res.level;
      }
    }
  }

  absl::c_sort(
      search_results_, [&](const SearchResult& a, const SearchResult& b) {
        if (a.pid != b.pid) {
          const auto it_a = process_sort_indices.find(a.pid);
          uint32_t sort_index_a = (it_a != process_sort_indices.end())
                                      ? it_a->second
                                      : std::numeric_limits<uint32_t>::max();
          const auto it_b = process_sort_indices.find(b.pid);
          uint32_t sort_index_b = (it_b != process_sort_indices.end())
                                      ? it_b->second
                                      : std::numeric_limits<uint32_t>::max();
          if (sort_index_a != sort_index_b) {
            return sort_index_a < sort_index_b;
          }
          return a.pid < b.pid;
        }

        if (a.tid != b.tid) {
          int min_level_a = -1;
          if (const auto it_min_a = pid_tid_to_min_level.find({a.pid, a.tid});
              it_min_a != pid_tid_to_min_level.end()) {
            min_level_a = it_min_a->second;
          }
          int min_level_b = -1;
          if (const auto it_min_b = pid_tid_to_min_level.find({b.pid, b.tid});
              it_min_b != pid_tid_to_min_level.end()) {
            min_level_b = it_min_b->second;
          }

          if (min_level_a != -1 && min_level_b != -1) {
            if (min_level_a != min_level_b) {
              return min_level_a < min_level_b;
            }
          }

          uint32_t thread_sort_a = std::numeric_limits<uint32_t>::max();
          if (const auto it_thread_a = thread_sort_indices.find({a.pid, a.tid});
              it_thread_a != thread_sort_indices.end()) {
            thread_sort_a = it_thread_a->second;
          }
          uint32_t thread_sort_b = std::numeric_limits<uint32_t>::max();
          if (const auto it_thread_b = thread_sort_indices.find({b.pid, b.tid});
              it_thread_b != thread_sort_indices.end()) {
            thread_sort_b = it_thread_b->second;
          }

          if (thread_sort_a != thread_sort_b) {
            return thread_sort_a < thread_sort_b;
          }
          return a.tid < b.tid;
        }

        int effective_level_a = a.level;
        if (effective_level_a == -1) {
          const auto it_min = pid_tid_to_min_level.find({a.pid, a.tid});
          effective_level_a =
              (it_min != pid_tid_to_min_level.end()) ? it_min->second : 0;
        }
        int effective_level_b = b.level;
        if (effective_level_b == -1) {
          const auto it_min = pid_tid_to_min_level.find({b.pid, b.tid});
          effective_level_b =
              (it_min != pid_tid_to_min_level.end()) ? it_min->second : 0;
        }
        if (effective_level_a != effective_level_b) {
          return effective_level_a < effective_level_b;
        }
        return a.start_time < b.start_time;
      });

  if (navigation_active) {
    current_search_result_index_ = 0;
    NavigateToSearchResult(search_results_[0]);
  }

  if (redraw_callback_) redraw_callback_();
}

void Timeline::NavigateToSearchResult(const SearchResult& result) {
  if (result.loaded_index != -1) {
    ZoomEvent(result.loaded_index);
  } else {
    pending_navigation_event_id_ = result.event_id;
    const Microseconds start = result.start_time;
    const Microseconds event_duration = result.duration;
    const Microseconds duration =
        std::max(kEventNavigationMinDurationMicros,
                 std::min(event_duration * kEventNavigationZoomFactor,
                          kEventNavigationMaxDurationMicros));
    const Microseconds center = start + event_duration / 2.0;
    TimeRange new_range = {center - duration / 2.0, center + duration / 2.0};
    ConstrainTimeRange(new_range);
    SetVisibleRange(new_range, /*animate=*/true);
  }
}

void Timeline::NavigateToNextSearchResult() {
  const int count = get_search_results_count();
  if (count == 0) return;
  ++current_search_result_index_;
  if (current_search_result_index_ >= count) {
    current_search_result_index_ = 0;
  }
  if (!search_results_.empty()) {
    NavigateToSearchResult(search_results_[current_search_result_index_]);
  } else {
    const int matched_idx =
        GetNthMatchingCsrEventIndex(current_search_result_index_);
    if (matched_idx != -1) {
      ZoomEvent(matched_idx);
    }
  }
  if (redraw_callback_) redraw_callback_();
}

void Timeline::NavigateToPrevSearchResult() {
  const int count = get_search_results_count();
  if (count == 0) return;
  --current_search_result_index_;
  if (current_search_result_index_ < 0) {
    current_search_result_index_ = count - 1;
  }
  if (!search_results_.empty()) {
    NavigateToSearchResult(search_results_[current_search_result_index_]);
  } else {
    const int matched_idx =
        GetNthMatchingCsrEventIndex(current_search_result_index_);
    if (matched_idx != -1) {
      ZoomEvent(matched_idx);
    }
  }
  if (redraw_callback_) redraw_callback_();
}

void Timeline::FindSelectedEvents(const ImRect& selection_rect) {
  selected_event_indices_.clear();
  selected_event_indices_.Reserve(timeline_data_.entry_start_times.size());
  selected_counter_points_.clear();
  selected_event_index_ = -1;
  selected_group_index_ = -1;
  selected_counter_index_ = -1;
  selected_counter_bundle_ = -1;

  const ImRect timeline_area = GetTimelineArea();
  const Pixel screen_x_offset = timeline_area.Min.x;
  const double px_per_time = px_per_time_unit();

  for (size_t group_index = 0; group_index < timeline_data_.groups.size();
       ++group_index) {
    const auto& group = timeline_data_.groups[group_index];
    if (!group.expanded) continue;

    if (group.type == Group::Type::kFlame) {
      const int start_level = group.start_level;
      const int end_level =
          std::min(GetNextGroupStartLevel(timeline_data_, group_index),
                   timeline_data_.total_levels());

      for (int level = start_level; level < end_level; ++level) {
        absl::Span<const int> events = timeline_data_.level_events(level);
        const Pixel y_top = tracks_start_screen_pos_.y +
                            visible_level_offsets_[level] - kEventHeight * 0.5f;
        const Pixel y_bottom = y_top + kEventHeight;

        if (y_bottom < selection_rect.Min.y || y_top > selection_rect.Max.y) {
          continue;
        }

        Microseconds selection_start_time =
            PixelToTime(selection_rect.Min.x - screen_x_offset, px_per_time);
        Microseconds selection_end_time =
            PixelToTime(selection_rect.Max.x - screen_x_offset, px_per_time);

        auto it =
            std::lower_bound(events.begin(), events.end(), selection_start_time,
                             [&](int event_idx, Microseconds t) {
                               const Microseconds end =
                                   timeline_data_.entry_start_times[event_idx] +
                                   timeline_data_.entry_total_times[event_idx];
                               return end <= t;
                             });

        for (; it != events.end(); ++it) {
          const int event_index = *it;
          if (!IsEventVisibleUnderFilter(event_index)) {
            continue;
          }
          const Microseconds start =
              timeline_data_.entry_start_times[event_index];
          if (start > selection_end_time) {
            break;
          }

          const Microseconds end =
              start + timeline_data_.entry_total_times[event_index];

          const Pixel left = TimeToScreenX(start, screen_x_offset, px_per_time);
          Pixel right = TimeToScreenX(end, screen_x_offset, px_per_time);
          right = std::max(right, left + kEventMinimumDrawWidth);

          if (right < selection_rect.Min.x || left > selection_rect.Max.x) {
            continue;
          }

          selected_event_indices_.insert(event_index);
        }
      }
    } else if (group.type == Group::Type::kCounter) {
      Pixel y_top = tracks_start_screen_pos_.y + group_offsets_[group_index];
      Pixel group_height = kCounterTrackHeight;
      Pixel y_bottom = y_top + group_height;

      if (y_bottom < selection_rect.Min.y || y_top > selection_rect.Max.y) {
        continue;
      }

      const auto it =
          timeline_data_.counter_data_by_group_index.find(group_index);
      if (it == timeline_data_.counter_data_by_group_index.end()) continue;
      const auto& counter_data = it->second;

      const double value_range =
          counter_data.max_value - counter_data.min_value;
      const float y_ratio = value_range > 0 ? group_height / value_range : 0.0f;

      Microseconds selection_start_time =
          PixelToTime(selection_rect.Min.x - screen_x_offset, px_per_time);
      Microseconds selection_end_time =
          PixelToTime(selection_rect.Max.x - screen_x_offset, px_per_time);

      auto it_start =
          std::lower_bound(counter_data.timestamps.begin(),
                           counter_data.timestamps.end(), selection_start_time);
      auto it_end =
          std::upper_bound(counter_data.timestamps.begin(),
                           counter_data.timestamps.end(), selection_end_time);

      for (auto it = it_start; it != it_end; ++it) {
        size_t i = std::distance(counter_data.timestamps.begin(), it);
        double val = counter_data.values[i];

        Pixel x = TimeToScreenX(*it, screen_x_offset, px_per_time);
        Pixel y = (value_range > 0)
                      ? y_top + group_height -
                            (val - counter_data.min_value) * y_ratio
                      : y_top + group_height / 2.0f;

        if (selection_rect.Contains(ImVec2(x, y))) {
          selected_counter_points_.push_back({group_index, i});
        }
      }
      if (!event_tooltip_enabled_ &&
          selection_end_time > selection_start_time &&
          !counter_data.timestamps.empty()) {
        auto it_active = std::upper_bound(counter_data.timestamps.begin(),
                                          counter_data.timestamps.end(),
                                          selection_start_time);
        size_t first_idx =
            (it_active == counter_data.timestamps.begin())
                ? 0
                : static_cast<size_t>(std::distance(
                      counter_data.timestamps.begin(), std::prev(it_active)));
        const Microseconds schedule_end = GetScheduleEndBundle();
        for (size_t i = first_idx; i < counter_data.timestamps.size() &&
                                   i < counter_data.values.size();
             ++i) {
          const Microseconds seg_start = counter_data.timestamps[i];
          if (seg_start > selection_end_time) break;
          const Microseconds seg_end =
              (i + 1 < counter_data.timestamps.size())
                  ? counter_data.timestamps[i + 1]
                  : std::max(seg_start + 1.0, schedule_end);
          if (seg_end <= selection_start_time) continue;
          const std::pair<int, int> pt(static_cast<int>(group_index),
                                       static_cast<int>(i));
          if (!absl::c_linear_search(selected_counter_points_, pt)) {
            selected_counter_points_.push_back(pt);
          }
        }
      }
    }
  }
}

void Timeline::CalculateAndEmitMetrics() {
  if (selected_event_indices_.empty() && selected_counter_points_.empty()) {
    event_callback_(kEventsSelected, EventData());
    return;
  }

  Microseconds selection_start_us = 0;
  Microseconds selection_extent_us = 0;
  if (selection_start_pos_ && selection_end_pos_ &&
      visible_range_->duration() > 0) {
    ImRect timeline_area = GetTimelineArea();
    double px_per_time = current_timeline_width_ / visible_range_->duration();
    Microseconds start_us =
        PixelToTime(std::min(selection_start_pos_->x, selection_end_pos_->x) -
                        timeline_area.Min.x,
                    px_per_time);
    Microseconds end_us =
        PixelToTime(std::max(selection_start_pos_->x, selection_end_pos_->x) -
                        timeline_area.Min.x,
                    px_per_time);
    selection_start_us = start_us;
    selection_extent_us = end_us - start_us;
  }

  std::string json =
      absl::StrFormat(R"({"selectionStartUs":%.1f,"selectionExtentUs":%.1f)",
                      selection_start_us, selection_extent_us);

  if (!selected_event_indices_.empty()) {
    bool include_metrics = true;
    if (include_metrics) {
      struct Metrics {
        int count = 0;
        Microseconds wall_time = 0;
        Microseconds self_time = 0;
      };
      absl::flat_hash_map<absl::string_view, Metrics> aggregated_metrics;

      selected_event_indices_.ForEachSetBit([&](int event_index) {
        const absl::string_view name = GetEventName(event_index);
        const Microseconds wall = timeline_data_.entry_total_times[event_index];
        const Microseconds self = timeline_data_.GetEntrySelfTime(event_index);
        Metrics& m = aggregated_metrics[name];
        ++m.count;
        m.wall_time += wall;
        m.self_time += self;
      });

      std::string metrics_json = "[";
      bool first = true;
      for (const auto& [name, metrics] : aggregated_metrics) {
        if (!first) absl::StrAppend(&metrics_json, ",");
        first = false;
        absl::StrAppendFormat(
            &metrics_json,
            R"({"name":"%s","count":%d,"wallTimeUs":%.1f,"selfTimeUs":%.1f,"avgWallDurationUs":%.1f})",
            absl::CEscape(name), metrics.count, metrics.wall_time,
            metrics.self_time, metrics.wall_time / metrics.count);
      }
      absl::StrAppend(&metrics_json, "]");

      absl::StrAppend(&json, R"(,"metrics":)", metrics_json);
    }
  }

  if (!selected_counter_points_.empty()) {
    std::string counters_json = "[";
    bool first = true;
    for (const auto& [group_index, point_index] : selected_counter_points_) {
      const auto& group = timeline_data_.groups[group_index];
      const auto it =
          timeline_data_.counter_data_by_group_index.find(group_index);
      if (it == timeline_data_.counter_data_by_group_index.end()) continue;
      const auto& counter_data = it->second;

      Microseconds ts = counter_data.timestamps[point_index];
      double val = counter_data.values[point_index];

      if (!first) absl::StrAppend(&counters_json, ",");
      first = false;
      absl::StrAppendFormat(
          &counters_json,
          R"({"counter":"%s","series":"%s","time":%.1f,"value":%.1f})",
          group.name, counter_data.event_stats, ts, val);
    }
    absl::StrAppend(&counters_json, "]");

    absl::StrAppend(&json, R"(,"counters":)", counters_json);
  }

  absl::StrAppend(&json, "}");

  EventData event_data;
  event_data.try_emplace(kEventsSelectedData, json);

  if (!event_tooltip_enabled_) {
    PopulateMultiSelectionScheduleDetails(selection_start_us,
                                          selection_extent_us, event_data);
  }

  event_callback_(kEventsSelected, event_data);
}

void Timeline::DrawSelectionRectangle() {
  if (mouse_mode_ != MouseMode::kSelect || !is_selecting_ ||
      !selection_start_pos_ || !selection_end_pos_) {
    return;
  }

  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  const ImU32 color = IM_COL32(0, 120, 215, 64);
  const ImU32 border_color = IM_COL32(0, 120, 215, 255);

  ImRect rect =
      ImRect(std::min(selection_start_pos_->x, selection_end_pos_->x),
             std::min(selection_start_pos_->y, selection_end_pos_->y),
             std::max(selection_start_pos_->x, selection_end_pos_->x),
             std::max(selection_start_pos_->y, selection_end_pos_->y));

  draw_list->AddRectFilled(rect.Min, rect.Max, color);
  draw_list->AddRect(rect.Min, rect.Max, border_color, 0.0f, 0, 2.0f);
}

void Timeline::DrawBookmarks(Pixel timeline_width,
                             double px_per_time_unit_val) {
  if (!bookmarks_enabled_ || bookmarks_.empty() || px_per_time_unit_val <= 0)
    return;

  const ImRect timeline_area = GetTimelineArea();
  const Pixel screen_x_offset = timeline_area.Min.x;
  ImDrawList* draw_list = ImGui::GetForegroundDrawList();
  std::optional<Microseconds> bookmark_to_delete;

  for (const Microseconds time : bookmarks_) {
    const Pixel x = TimeToScreenX(time, screen_x_offset, px_per_time_unit_val);

    if (x < timeline_area.Min.x || x > timeline_area.Max.x) continue;

    // Draw vertical line
    draw_list->AddLine(ImVec2(x, timeline_area.Min.y),
                       ImVec2(x, timeline_area.Max.y), kBookmarkColor,
                       kBookmarkThickness);

    // Draw label (timestamp)
    const std::string label =
        time_axis_unit_ == TimeAxisUnit::kUnitless
            ? absl::StrCat(
                  "Bundle ",
                  FormatRulerLabel(std::floor(time) - data_time_range_.start()))
            : FormatTime(time);
    ImGui::PushFont(fonts::label_small);
    const ImVec2 label_size = ImGui::CalcTextSize(label.c_str());
    const ImVec2 label_pos = ImVec2(
        x + kBookmarkLabelPadding, timeline_area.Min.y + kBookmarkLabelPadding);
    draw_list->AddText(label_pos, kBookmarkColor, label.c_str());

    // Unified Delete Button logic (matches TimeRange selection behavior)
    const ImVec2 button_pos =
        ImVec2(label_pos.x + label_size.x + kBookmarkLabelPadding, label_pos.y);
    const ImRect button_rect(button_pos,
                             ImVec2(button_pos.x + kCloseButtonSize,
                                    button_pos.y + kCloseButtonSize));

    // Hover area includes label and button
    const ImRect hover_rect(label_pos, button_rect.Max);

    if (DrawCloseButton(draw_list, button_pos, hover_rect)) {
      bookmark_to_delete = time;
    }

    ImGui::PopFont();
  }

  if (bookmark_to_delete) {
    RemoveBookmark(*bookmark_to_delete);
  }
}

void Timeline::AddBookmark(Microseconds time) {
  if (!bookmarks_enabled_) return;
  const double px_per_time = px_per_time_unit();
  const Microseconds threshold = px_per_time > 0 ? 5.0 / px_per_time : 1e-9;

  auto it = absl::c_find_if(bookmarks_, [time, threshold](Microseconds b) {
    return std::abs(b - time) < threshold;
  });

  if (it == bookmarks_.end()) {
    bookmarks_.push_back(time);
    absl::c_sort(bookmarks_);
    if (redraw_callback_) redraw_callback_();
  }
}

void Timeline::RemoveBookmark(Microseconds time) {
  if (!bookmarks_enabled_) return;
  auto it = absl::c_find(bookmarks_, time);
  if (it != bookmarks_.end()) {
    bookmarks_.erase(it);
    if (redraw_callback_) redraw_callback_();
  }
}

Timeline::GroupRelativeInfo Timeline::FindGroupRelatives(Group* target_group) {
  GroupRelativeInfo info;
  if (!target_group) return info;

  // Child track (e.g., thread or counter track): belongs to a parent process.
  // Resolves its parent group and determines its position within the parent's
  // child_indices list.
  if (target_group->parent_index != -1) {
    if (target_group->parent_index >= 0 &&
        target_group->parent_index < timeline_data_.groups.size()) {
      info.parent = &timeline_data_.groups[target_group->parent_index];
      info.siblings = &timeline_data_.groups;
      const auto& child_indices = info.parent->child_indices;
      for (size_t i = 0; i < child_indices.size(); ++i) {
        if (child_indices[i] >= 0 &&
            child_indices[i] < timeline_data_.groups.size() &&
            timeline_data_.groups[child_indices[i]].original_index ==
                target_group->original_index) {
          info.index_in_siblings = i;
          break;
        }
      }
    }
  } else {
    // Top-level root track (e.g., process track): parent_index is -1 because it
    // has no parent. Its siblings are all other root-level tracks in
    // timeline_data_.groups; determine its relative position among root tracks.
    info.parent = nullptr;
    info.siblings = &timeline_data_.groups;
    int root_count = 0;
    for (size_t i = 0; i < timeline_data_.groups.size(); ++i) {
      if (timeline_data_.groups[i].parent_index == -1) {
        if (timeline_data_.groups[i].original_index ==
            target_group->original_index) {
          info.index_in_siblings = root_count;
          break;
        }
        root_count++;
      }
    }
  }
  return info;
}

bool Timeline::HandleTrackDragAndDrop(int group_index, Group& group,
                                      const ImVec2& tracks_start_pos,
                                      const ImVec2& tracks_start_screen_pos,
                                      Pixel group_height,
                                      Pixel hover_zone_width) {
  const Pixel effective_height =
      group_height > 0.0f
          ? group_height
          : (group_index >= 0 && group_index < group_heights_.size()
                 ? group_heights_[group_index]
                 : 0.0f);
  if (hover_zone_width <= 0.0f || effective_height <= 0.0f) {
    return false;
  }

  ImGui::SetCursorPos(
      ImVec2(tracks_start_pos.x,
             tracks_start_pos.y + group_offsets_[group_index]));
  ImGui::SetNextItemAllowOverlap();
  ImGui::InvisibleButton(
      "RowLabelHoverZone",
      ImVec2(hover_zone_width, effective_height));
  const bool is_drag_and_drop_action =
      ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenOverlapped);

  if (ImGui::BeginDragDropSource(ImGuiDragDropFlags_SourceAllowNullID)) {
    Group* group_ptr_payload = &group;
    ImGui::SetDragDropPayload("TRACK_REORDER", &group_ptr_payload,
                              sizeof(Group*));
    ImGui::Text("%s%s", kMovingTrackNotificationPrefix, group.name.c_str());
    ImGui::EndDragDropSource();
  }

  if (ImGui::BeginDragDropTarget()) {
    if (const ImGuiPayload* group_ptr_payload = ImGui::AcceptDragDropPayload(
            "TRACK_REORDER", ImGuiDragDropFlags_AcceptBeforeDelivery |
                             ImGuiDragDropFlags_AcceptNoDrawDefaultRect)) {
      if (group_ptr_payload->Data != nullptr &&
          group_ptr_payload->DataSize >= sizeof(Group*)) {
        // Dereferencing as *(Group**)group_ptr_payload->Data
        // retrieves the original pointer to the source group.
        Group* source_group = *(Group**)group_ptr_payload->Data;
        Group* target_group = &group;
        if (source_group && target_group) {
          GroupRelativeInfo source_info = FindGroupRelatives(source_group);
          GroupRelativeInfo target_info = FindGroupRelatives(target_group);
          if (source_info.parent == target_info.parent) {
            Pixel line_y =
                tracks_start_screen_pos.y + group_offsets_[group_index];
            bool drop_after =
                ImGui::GetIO().MousePos.y > line_y + group_height * 0.5f;
            if (group_ptr_payload->IsDelivery()) {
              pending_reorder_source_ = source_group->original_index;
              pending_reorder_target_ = target_group->original_index;
              pending_reorder_drop_after_ = drop_after;
              if (redraw_callback_) redraw_callback_();
            } else {
              Pixel line_preview_y = line_y;
              if (drop_after) {
                line_preview_y += group_height;
              }
              reorder_preview_line_y_ = line_preview_y;
            }
          }
        }
      }
    }
    ImGui::EndDragDropTarget();
  }

  return is_drag_and_drop_action;
}

void Timeline::HandleTrackDragAndDropHoverAndFeedback(
    int group_index, Group& group, const ImVec2& tracks_start_pos,
    const ImVec2& tracks_start_screen_pos, Pixel group_height) {
  if (!track_management_enabled_) return;

  // Draw invisible button covering the left panel row label text to handle
  // hover and drag-and-drop.
  Pixel hover_zone_width = label_width_ - kSplitterOffset;
  // Only process tracks render track management buttons (Pin and Hide buttons)
  // on the right side of the label area. Subtract their reserved width (plus
  // gap and margin) so the invisible drag-and-drop button does not overlap or
  // intercept clicks intended for those buttons. Child tracks (threads,
  // counters) do not have management buttons, so their hover zone can safely
  // span the full label width up to the splitter.
  if (group.nesting_level == kProcessNestingLevel) {
    const Pixel track_management_button_width =
        CalculateTrackManagementButtonWidth();
    hover_zone_width -= track_management_button_width;
  }

  if (hover_zone_width <= 0.0f || group_height <= 0.0f) {
    return;
  }

  const bool is_drag_and_drop_action =
      HandleTrackDragAndDrop(group_index, group, tracks_start_pos,
                             tracks_start_screen_pos, group_height,
                             hover_zone_width);

  if (is_drag_and_drop_action && ImGui::GetDragDropPayload() == nullptr) {
    ImGui::SetTooltip("%s", kReorderTrackTooltip);
  }
}

}  // namespace traceviewer
