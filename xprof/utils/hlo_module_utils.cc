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

#include "xprof/utils/hlo_module_utils.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

using tsl::profiler::OpSourceInfo;
using xla::StackFrameIndexProto;

struct StackFrameInfo {
  std::string file_name;
  int32_t line_number;
  int32_t column_number;
};

// Joins the elements of the input `stack_frames` into a single string.

// The result is a newline-separated list of file names, line, and column
// numbers (numbers could be negative). Each line is of the form:
// `<file_name>:<line_number>:<column_number>`. Order of stack frames in the
// result is same as their order in the input.
std::string JoinStackFrames(const std::vector<StackFrameInfo>& stack_frames) {
  std::stringstream result;
  for (auto end = stack_frames.cend(), it = stack_frames.cbegin(); it != end;
       ++it) {
    if (result.tellp() > 0) {
      result << '\n';
    }
    result << it->file_name << ':' << it->line_number << ':'
           << it->column_number;
  }
  return result.str();
}

// Extracts the stack frames from the input `stack_frame_index` starting from
// the input `frame_id` (1-based) and going up the stack. `StackFrameInfo` for
// the input `frame_id`, if not zero, will be the first element in the result.
//
// Frames for which we cannot find all the information remain included in the
// result. If a file name is not found, then an empty string is used instead.
// If a line or column number is not found, then `-1` is used instead.
std::vector<StackFrameInfo> ExtractStackFrames(
    const StackFrameIndexProto& stack_frame_index, int32_t frame_id) {
  std::vector<StackFrameInfo> result;
  while (frame_id > 0 && frame_id <= stack_frame_index.stack_frames_size()) {
    const StackFrameIndexProto::StackFrame& frame =
        stack_frame_index.stack_frames(frame_id - 1);
    frame_id = frame.parent_frame_id();
    std::string file_name;
    int32_t line_number = -1;
    int32_t column_number = -1;
    if (const auto file_location_id = frame.file_location_id();
        file_location_id > 0 &&
        file_location_id <= stack_frame_index.file_locations_size()) {
      const StackFrameIndexProto::FileLocation& file_location =
          stack_frame_index.file_locations(file_location_id - 1);
      line_number = file_location.line();
      column_number = file_location.column();
      if (const auto file_name_id = file_location.file_name_id();
          file_name_id > 0 &&
          file_name_id <= stack_frame_index.file_names_size()) {
        file_name = stack_frame_index.file_names(file_name_id - 1);
      }
    }
    result.emplace_back(StackFrameInfo{.file_name = std::move(file_name),
                                       .line_number = line_number,
                                       .column_number = column_number});
  }
  return result;
}

// Builds the stack frames string for the input `frame_id` (1-based) from the
// input `stack_frame_index`. Stack frame for the input `frame_id`, if not zero,
// will be the last line in the returned value.
std::string BuildStackFrames(const StackFrameIndexProto& stack_frame_index,
                             int32_t frame_id) {
  return JoinStackFrames(ExtractStackFrames(stack_frame_index, frame_id));
}

OpSourceInfo ParseOpSourceInfoFromStackFrame(absl::string_view stack_frame) {
  OpSourceInfo source_info;
  if (!stack_frame.empty()) {
    absl::string_view file_line_col(stack_frame);
    // The format is file:line:col\n...
    // Only parse the first line.
    file_line_col = file_line_col.substr(0, file_line_col.find('\n'));
    // Note: the parsing logic now assumes non-windows file paths to not break.
    std::vector<absl::string_view> parts = absl::StrSplit(file_line_col, ':');
    if (parts.size() >= 2) {
      source_info.source_file = std::string(parts[0]);
      int line;
      if (absl::SimpleAtoi(parts[1], &line)) {
        source_info.source_line = line;
      }
    }
  }
  source_info.stack_frame = std::string(stack_frame);
  return source_info;
}

OpSourceInfo GetSourceInfo(absl::string_view source_file, int32_t source_line,
                           absl::string_view stack_frame) {
  if (!stack_frame.empty()) {
    auto source_info = ParseOpSourceInfoFromStackFrame(stack_frame);
    if (!source_info.source_file.empty() && source_info.source_line) {
      return source_info;
    }
  }
  return {.source_file = std::string(source_file),
          .source_line = source_line,
          .stack_frame = std::string(stack_frame)};
}
}  // namespace

OpSourceInfo GetSourceInfo(const xla::HloInstructionProto& instr,
                           const xla::StackFrameIndexProto& stack_frame_index) {
  const std::string stack_frame =
      BuildStackFrames(stack_frame_index, instr.metadata().stack_frame_id());
  return GetSourceInfo(instr.metadata().source_file(),
                       instr.metadata().source_line(), stack_frame);
}

OpSourceInfo GetSourceInfo(const xla::HloInstruction& instr) {
  const auto stack_frame_id = instr.metadata().stack_frame_id();
  const std::string stack_frame =
      stack_frame_id != 0 ? GetOpLocationStack(stack_frame_id, instr) : "";
  return GetSourceInfo(instr.metadata().source_file(),
                       instr.metadata().source_line(), stack_frame);
}

}  // namespace profiler
}  // namespace tensorflow
