#include "xprof/convert/trace_viewer/delta_series/trace_data_to_compressed_delta_series_proto.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/message_lite.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "tsl/profiler/lib/context_types.h"
#include "xprof/convert/trace_viewer/delta_series/delta_series_converter_internal.h"
#include "xprof/convert/trace_viewer/delta_series/zstd_compression.h"
#include "xprof/convert/trace_viewer/trace_events.h"
#include "xprof/convert/trace_viewer/trace_events_to_json.h"
#include "plugin/xprof/protobuf/trace_data_response.pb.h"
#include "plugin/xprof/protobuf/trace_events.pb.h"
#include "plugin/xprof/protobuf/trace_events_raw.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

constexpr absl::string_view kNameArgName = "Name";
constexpr absl::string_view kHloModuleIdArgName = "hlo_module_id";

constexpr uint8_t kWireTypeVarint = 0;
constexpr uint8_t kWireTypeLengthDelimited = 2;

constexpr uint8_t MakeWireTag(int field_number, uint8_t wire_type) {
  return static_cast<uint8_t>((field_number << 3) | wire_type);
}

absl::string_view ExtractModuleIdFromKernelDetails(
    absl::string_view kernel_details) {
  constexpr absl::string_view kPrefix = "module:";
  size_t search_pos = 0;
  while (search_pos < kernel_details.size()) {
    const size_t pos = kernel_details.find(kPrefix, search_pos);
    if (pos == absl::string_view::npos) break;
    const size_t val_start = pos + kPrefix.size();
    size_t val_end = val_start;
    while (val_end < kernel_details.size() && kernel_details[val_end] != ' ' &&
           kernel_details[val_end] != '\t' && kernel_details[val_end] != '\n' &&
           kernel_details[val_end] != '\r') {
      ++val_end;
    }
    const absl::string_view token =
        kernel_details.substr(val_start, val_end - val_start);
    const size_t underscore = token.rfind('_');
    if (underscore != absl::string_view::npos &&
        underscore + 1 < token.size()) {
      const absl::string_view suffix = token.substr(underscore + 1);
      bool all_digits = true;
      for (char c : suffix) {
        if (c < '0' || c > '9') {
          all_digits = false;
          break;
        }
      }
      if (all_digits) {
        return suffix;
      }
    }
    const size_t open_paren = token.find('(');
    if (open_paren != absl::string_view::npos && open_paren > 0) {
      const size_t close_paren = token.find(')', open_paren + 1);
      if (close_paren != absl::string_view::npos &&
          close_paren > open_paren + 1) {
        return token.substr(open_paren + 1, close_paren - open_paren - 1);
      }
    }
    search_pos = val_end;
  }
  return "";
}

bool MayContainRequiredHloWireKeys(absl::string_view raw_bytes,
                                   internal::HloExtractionTrackKind track_kind,
                                   absl::string_view hlo_op_arg_name,
                                   absl::string_view hlo_module_arg_name) {
  switch (track_kind) {
    case internal::HloExtractionTrackKind::kDataMotionLayer:
      return true;
    case internal::HloExtractionTrackKind::kXlaOps:
      return absl::StrContains(raw_bytes, hlo_op_arg_name) ||
             absl::StrContains(raw_bytes, hlo_module_arg_name);
    case internal::HloExtractionTrackKind::kRegular:
      break;
  }
  return absl::StrContains(raw_bytes, hlo_op_arg_name) &&
         absl::StrContains(raw_bytes, hlo_module_arg_name);
}

// Stack-backed HLO views decoded from one `TraceEventArguments`. Numeric views
// alias `num_bufs`, so copying and moving are deleted.
struct DecodedHloArgs {
  static constexpr size_t kNumBufSize = 24;

  DecodedHloArgs() = default;
  DecodedHloArgs(const DecodedHloArgs&) = delete;
  DecodedHloArgs& operator=(const DecodedHloArgs&) = delete;

  absl::string_view hlo_op;
  absl::string_view name_arg;
  absl::string_view hlo_module;
  absl::string_view hlo_module_id;
  absl::string_view program_id;
  absl::string_view kernel_details;
  char num_bufs[6][kNumBufSize];
};

void DecodeHloArguments(const Trace& trace, const TraceEventArguments& args,
                        internal::HloExtractionTrackKind track_kind,
                        absl::string_view hlo_op_arg_name,
                        absl::string_view hlo_module_arg_name,
                        DecodedHloArgs* out) {
  static const absl::string_view kProgramIdArgName =
      tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kProgramId);
  static const absl::string_view kKernelDetailsArgName =
      tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kKernelDetails);
  const bool is_data_motion_layer =
      (track_kind == internal::HloExtractionTrackKind::kDataMotionLayer);

  for (const TraceEventArguments::Argument& arg : args.arg()) {
    const absl::string_view arg_name = arg.name();
    absl::string_view* target_view = nullptr;
    char* target_num_buf = nullptr;
    if (is_data_motion_layer && arg_name == kNameArgName) {
      target_view = &out->name_arg;
      target_num_buf = out->num_bufs[1];
    } else if (arg_name == hlo_op_arg_name) {
      target_view = &out->hlo_op;
      target_num_buf = out->num_bufs[0];
    } else if (arg_name == hlo_module_arg_name) {
      target_view = &out->hlo_module;
      target_num_buf = out->num_bufs[2];
    } else if (arg_name == kHloModuleIdArgName) {
      target_view = &out->hlo_module_id;
      target_num_buf = out->num_bufs[3];
    } else if (arg_name == kProgramIdArgName) {
      target_view = &out->program_id;
      target_num_buf = out->num_bufs[4];
    } else if (arg_name == kKernelDetailsArgName) {
      target_view = &out->kernel_details;
      target_num_buf = out->num_bufs[5];
    }
    if (target_view == nullptr) continue;

    absl::string_view val;
    if (arg.has_str_value()) {
      val = arg.str_value();
    } else if (arg.has_ref_value()) {
      const auto it = trace.name_table().find(arg.ref_value());
      if (it != trace.name_table().end()) {
        val = it->second;
      }
    } else if (arg.has_int_value()) {
      const absl::AlphaNum num(arg.int_value());
      std::memcpy(target_num_buf, num.data(), num.size());
      val = absl::string_view(target_num_buf, num.size());
    } else if (arg.has_uint_value()) {
      const absl::AlphaNum num(arg.uint_value());
      std::memcpy(target_num_buf, num.data(), num.size());
      val = absl::string_view(target_num_buf, num.size());
    }
    if (!val.empty()) {
      *target_view = val;
    }
  }
}

absl::string_view SelectHloOp(const DecodedHloArgs& decoded,
                              internal::HloExtractionTrackKind track_kind,
                              absl::string_view event_name) {
  switch (track_kind) {
    case internal::HloExtractionTrackKind::kDataMotionLayer:
      return !decoded.name_arg.empty() ? decoded.name_arg : decoded.hlo_op;
    case internal::HloExtractionTrackKind::kXlaOps:
      return decoded.hlo_op != event_name ? decoded.hlo_op
                                          : absl::string_view();
    case internal::HloExtractionTrackKind::kRegular:
      break;
  }
  return decoded.hlo_op;
}

void FormatHloModuleWithSuffix(const DecodedHloArgs& decoded,
                               std::string* out_hlo_module) {
  if (decoded.hlo_module.empty()) return;
  absl::string_view id_suffix;
  if (!decoded.hlo_module_id.empty()) {
    id_suffix = decoded.hlo_module_id;
  } else if (!decoded.program_id.empty()) {
    id_suffix = decoded.program_id;
  } else if (!decoded.kernel_details.empty()) {
    id_suffix = ExtractModuleIdFromKernelDetails(decoded.kernel_details);
  }
  if (id_suffix.empty()) {
    out_hlo_module->assign(decoded.hlo_module.data(),
                           decoded.hlo_module.size());
    return;
  }
  const size_t suffix_len = id_suffix.size() + 2;
  const bool already_has_suffix =
      decoded.hlo_module.size() >= suffix_len &&
      decoded.hlo_module[decoded.hlo_module.size() - suffix_len] == '(' &&
      decoded.hlo_module.back() == ')' &&
      decoded.hlo_module.substr(decoded.hlo_module.size() - suffix_len + 1,
                                id_suffix.size()) == id_suffix;
  if (already_has_suffix) {
    out_hlo_module->assign(decoded.hlo_module.data(),
                           decoded.hlo_module.size());
    return;
  }
  out_hlo_module->clear();
  // NOMUTANTS(AOR): Preallocation capacity hint only; output string identical.
  out_hlo_module->reserve(decoded.hlo_module.size() + suffix_len);
  absl::StrAppend(out_hlo_module, decoded.hlo_module, "(", id_suffix, ")");
}

absl::Status AddDelimitedMessageWireBytes(
    const google::protobuf::MessageLite& msg, absl::string_view label,
    size_t* total_wire_bytes,
    size_t max_wire_bytes = internal::kMaxProtobufSerializedBytes) {
  const size_t msg_size = msg.ByteSizeLong();
  if (absl::Status status =
          internal::ValidateProtobufMessageSize(msg_size, label,
                                                max_wire_bytes);
      !status.ok()) {
    return status;
  }
  return internal::AddWireBytesChecked(
      total_wire_bytes,
      1 + google::protobuf::io::CodedOutputStream::VarintSize64(msg_size) + msg_size,
      "TraceDataResponse", max_wire_bytes);
}

uint8_t* WriteDelimitedMessageWireBytes(uint8_t tag,
                                        const google::protobuf::MessageLite& msg,
                                        uint8_t* ptr) {
  *ptr++ = tag;
  const size_t msg_size = static_cast<size_t>(msg.GetCachedSize());
  ptr = google::protobuf::io::CodedOutputStream::WriteVarint64ToArray(msg_size, ptr);
  uint8_t* const next_ptr = msg.SerializeWithCachedSizesToArray(ptr);
  DCHECK_EQ(next_ptr, ptr + msg_size);
  return next_ptr;
}

static_assert(xprof::TraceDataResponse::kCompleteEventsFieldNumber <= 15);
static_assert(xprof::TraceDataResponse::kCounterEventsFieldNumber <= 15);
static_assert(xprof::TraceDataResponse::kAsyncEventsFieldNumber <= 15);
static_assert(xprof::TraceDataResponse::kMetadataFieldNumber <= 15);
static_assert(xprof::TraceDataResponse::kInternedStringsFieldNumber <= 15);
static_assert(xprof::TraceDataResponse::kDetailsFieldNumber <= 15);
static_assert(xprof::TraceDataResponse::kFullTimespanStartPsFieldNumber <= 15);
static_assert(xprof::TraceDataResponse::kFullTimespanEndPsFieldNumber <= 15);

constexpr uint8_t kCompleteEventsTag =
    MakeWireTag(xprof::TraceDataResponse::kCompleteEventsFieldNumber,
                kWireTypeLengthDelimited);
constexpr uint8_t kCounterEventsTag =
    MakeWireTag(xprof::TraceDataResponse::kCounterEventsFieldNumber,
                kWireTypeLengthDelimited);
constexpr uint8_t kAsyncEventsTag =
    MakeWireTag(xprof::TraceDataResponse::kAsyncEventsFieldNumber,
                kWireTypeLengthDelimited);
constexpr uint8_t kMetadataTag = MakeWireTag(
    xprof::TraceDataResponse::kMetadataFieldNumber, kWireTypeLengthDelimited);
constexpr uint8_t kInternedStringsTag =
    MakeWireTag(xprof::TraceDataResponse::kInternedStringsFieldNumber,
                kWireTypeLengthDelimited);
constexpr uint8_t kDetailsTag = MakeWireTag(
    xprof::TraceDataResponse::kDetailsFieldNumber, kWireTypeLengthDelimited);
constexpr uint8_t kFullTimespanStartPsTag = MakeWireTag(
    xprof::TraceDataResponse::kFullTimespanStartPsFieldNumber, kWireTypeVarint);
constexpr uint8_t kFullTimespanEndPsTag = MakeWireTag(
    xprof::TraceDataResponse::kFullTimespanEndPsFieldNumber, kWireTypeVarint);

template <typename RepeatedMessage>
absl::Status AddRepeatedDelimitedMessageWireBytes(
    const RepeatedMessage& messages, absl::string_view label,
    size_t* total_wire_bytes, size_t max_wire_bytes) {
  for (const auto& msg : messages) {
    if (absl::Status s = AddDelimitedMessageWireBytes(
            msg, label, total_wire_bytes, max_wire_bytes);
        !s.ok()) {
      return s;
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> ComputeHeaderFirstWireSize(
    const xprof::TraceDataResponse& response, size_t max_wire_bytes) {
  size_t total_wire_bytes = 0;

  // Fields 2..8 (header fields) serialized before field 1 (complete_events).
  if (absl::Status s = AddRepeatedDelimitedMessageWireBytes(
          response.counter_events(), "TraceDataResponse.counter_events",
          &total_wire_bytes, max_wire_bytes);
      !s.ok()) {
    return s;
  }
  if (absl::Status s = AddRepeatedDelimitedMessageWireBytes(
          response.async_events(), "TraceDataResponse.async_events",
          &total_wire_bytes, max_wire_bytes);
      !s.ok()) {
    return s;
  }
  if (response.has_metadata()) {
    if (absl::Status s = AddDelimitedMessageWireBytes(
            response.metadata(), "TraceDataResponse.metadata",
            &total_wire_bytes, max_wire_bytes);
        !s.ok()) {
      return s;
    }
  }
  for (const std::string& str : response.interned_strings()) {
    if (absl::Status s = internal::ValidateProtobufMessageSize(
            str.size(), "TraceDataResponse.interned_strings", max_wire_bytes);
        !s.ok()) {
      return s;
    }
    if (absl::Status s = internal::AddWireBytesChecked(
            &total_wire_bytes,
            1 + google::protobuf::io::CodedOutputStream::VarintSize64(str.size()) +
                str.size(),
            "TraceDataResponse", max_wire_bytes);
        !s.ok()) {
      return s;
    }
  }
  if (absl::Status s = AddRepeatedDelimitedMessageWireBytes(
          response.details(), "TraceDataResponse.details", &total_wire_bytes,
          max_wire_bytes);
      !s.ok()) {
    return s;
  }
  if (response.has_full_timespan_start_ps()) {
    if (absl::Status s = internal::AddWireBytesChecked(
            &total_wire_bytes,
            1 + google::protobuf::io::CodedOutputStream::VarintSize64(
                    response.full_timespan_start_ps()),
            "TraceDataResponse", max_wire_bytes);
        !s.ok()) {
      return s;
    }
  }
  if (response.has_full_timespan_end_ps()) {
    if (absl::Status s = internal::AddWireBytesChecked(
            &total_wire_bytes,
            1 + google::protobuf::io::CodedOutputStream::VarintSize64(
                    response.full_timespan_end_ps()),
            "TraceDataResponse", max_wire_bytes);
        !s.ok()) {
      return s;
    }
  }
  if (absl::Status s = AddRepeatedDelimitedMessageWireBytes(
          response.complete_events(), "TraceDataResponse.complete_events",
          &total_wire_bytes, max_wire_bytes);
      !s.ok()) {
    return s;
  }
  return total_wire_bytes;
}

void WriteHeaderFirstWireBytes(const xprof::TraceDataResponse& response,
                               uint8_t* start, size_t expected_wire_bytes) {
  uint8_t* ptr = start;
  for (const xprof::TraceEventSeries& series : response.counter_events()) {
    ptr = WriteDelimitedMessageWireBytes(kCounterEventsTag, series, ptr);
  }
  for (const xprof::TraceEventSeries& series : response.async_events()) {
    ptr = WriteDelimitedMessageWireBytes(kAsyncEventsTag, series, ptr);
  }
  if (response.has_metadata()) {
    ptr =
        WriteDelimitedMessageWireBytes(kMetadataTag, response.metadata(), ptr);
  }
  for (const std::string& str : response.interned_strings()) {
    *ptr++ = kInternedStringsTag;
    ptr = google::protobuf::io::CodedOutputStream::WriteVarint64ToArray(str.size(), ptr);
    if (!str.empty()) {
      std::memcpy(ptr, str.data(), str.size());
      ptr += str.size();
    }
  }
  for (const xprof::DetailItem& detail : response.details()) {
    ptr = WriteDelimitedMessageWireBytes(kDetailsTag, detail, ptr);
  }
  if (response.has_full_timespan_start_ps()) {
    *ptr++ = kFullTimespanStartPsTag;
    ptr = google::protobuf::io::CodedOutputStream::WriteVarint64ToArray(
        response.full_timespan_start_ps(), ptr);
  }
  if (response.has_full_timespan_end_ps()) {
    *ptr++ = kFullTimespanEndPsTag;
    ptr = google::protobuf::io::CodedOutputStream::WriteVarint64ToArray(
        response.full_timespan_end_ps(), ptr);
  }
  for (const xprof::TraceEventSeries& chunk : response.complete_events()) {
    ptr = WriteDelimitedMessageWireBytes(kCompleteEventsTag, chunk, ptr);
  }
  DCHECK_EQ(static_cast<size_t>(ptr - start), expected_wire_bytes);
}

absl::StatusOr<std::string> SerializeTraceDataResponseHeaderFirstImpl(
    const xprof::TraceDataResponse& response, ZstdCompressorContext* zstd_ctx,
    size_t max_wire_bytes = internal::kMaxProtobufSerializedBytes) {
  const absl::StatusOr<size_t> total_wire_bytes =
      ComputeHeaderFirstWireSize(response, max_wire_bytes);
  if (!total_wire_bytes.ok()) {
    return total_wire_bytes.status();
  }

  std::unique_ptr<char[]> wire_buf;
  if (*total_wire_bytes > 0) {
    wire_buf = absl::make_unique_for_overwrite<char[]>(*total_wire_bytes);
    WriteHeaderFirstWireBytes(response,
                              reinterpret_cast<uint8_t*>(wire_buf.get()),
                              *total_wire_bytes);
  }

  const absl::string_view wire_view(wire_buf.get(), *total_wire_bytes);
  if (zstd_ctx != nullptr) {
    std::string compressed;
    if (absl::Status status = zstd_ctx->Compress(wire_view, &compressed);
        !status.ok()) {
      return status;
    }
    return compressed;
  }
  return ZstdCompression::Compress(wire_view);
}

}  // namespace

namespace internal {

absl::StatusOr<std::string> SerializeTraceDataResponseHeaderFirstWithLimit(
    const xprof::TraceDataResponse& response, ZstdCompressorContext* zstd_ctx,
    size_t max_wire_bytes) {
  return SerializeTraceDataResponseHeaderFirstImpl(response, zstd_ctx,
                                                   max_wire_bytes);
}

}  // namespace internal

DeltaSeriesStringTable::DeltaSeriesStringTable() {
  // Reserve index 0 for empty strings to prevent default 0 name_refs
  // from pointing to valid strings.
  interned_strings_.emplace_back("");
  interned_strings_map_.emplace(interned_strings_.back(), 0);
}

uint32_t DeltaSeriesStringTable::InternString(absl::string_view str) {
  auto it = interned_strings_map_.find(str);
  if (it != interned_strings_map_.end()) {
    return it->second;
  }
  const uint32_t interned_string_id =
      static_cast<uint32_t>(interned_strings_.size());
  interned_strings_.emplace_back(str);
  interned_strings_map_.emplace(interned_strings_.back(), interned_string_id);
  return interned_string_id;
}

void DeltaSeriesStringTable::FlushInternedStringsIncrement(
    xprof::TraceDataResponse* response) {
  for (size_t i = num_interned_strings_flushed_; i < interned_strings_.size();
       ++i) {
    response->add_interned_strings(interned_strings_[i]);
  }
  num_interned_strings_flushed_ = interned_strings_.size();
}

void DeltaSeriesStringTable::MoveAllInternedStrings(
    xprof::TraceDataResponse* response) && {
  DCHECK_EQ(num_interned_strings_flushed_, 0);
  interned_strings_map_.clear();
  for (std::string& str : interned_strings_) {
    response->add_interned_strings(std::move(str));
  }
}

void DeltaSeriesStringTable::PopulateFlowAndGroupMetadata(
    std::optional<uint64_t> flow_id, std::optional<uint32_t> flow_category,
    std::optional<int64_t> group_id, xprof::TraceEventMetadata* metadata) {
  if (flow_id.has_value()) {
    metadata->set_flow_id(*flow_id);
  }
  if (flow_category.has_value()) {
    tsl::profiler::ContextType type =
        tsl::profiler::GetSafeContextType(*flow_category);
    if (type != tsl::profiler::ContextType::kGeneric &&
        type != tsl::profiler::ContextType::kLegacy) {
      const char* category = tsl::profiler::GetContextTypeString(type);
      metadata->set_flow_category(InternString(category));
    }
  }
  if (group_id.has_value()) {
    metadata->set_group_id(*group_id);
  }
}

absl::StatusOr<std::string> SerializeTraceDataResponseHeaderFirst(
    const xprof::TraceDataResponse& response, ZstdCompressorContext* zstd_ctx) {
  return SerializeTraceDataResponseHeaderFirstImpl(response, zstd_ctx);
}

absl::StatusOr<std::string> SerializeAndCompress(
    const xprof::TraceDataResponse& response) {
  return SerializeTraceDataResponseHeaderFirstImpl(response,
                                                   /*zstd_ctx=*/nullptr);
}

absl::Status StreamingDeltaSeriesProtoConverter::AppendCompressedFrame(
    const xprof::TraceDataResponse& frame, bool include_magic_header,
    std::string* out) {
  if (out == nullptr) {
    return absl::InvalidArgumentError("Output frame buffer is null.");
  }
  absl::StatusOr<std::string> compressed = SerializeAndCompress(frame);
  if (!compressed.ok()) {
    return compressed.status();
  }
  // `SerializeAndCompress` bounds uncompressed wire bytes to `INT_MAX`
  // (~2 GiB), and `ZSTD_COMPRESSBOUND(INT_MAX)` (~2.008 GiB) is strictly
  // below `UINT32_MAX` (4 GiB).
  DCHECK_LE(compressed->size(),
            static_cast<size_t>(std::numeric_limits<uint32_t>::max()));
  if (include_magic_header) {
    out->append(kXChunkMagic.data(), kXChunkMagic.size());
  }
  const uint32_t frame_len = static_cast<uint32_t>(compressed->size());
  const char len_bytes[4] = {
      static_cast<char>(frame_len & 0xff),
      static_cast<char>((frame_len >> 8) & 0xff),
      static_cast<char>((frame_len >> 16) & 0xff),
      static_cast<char>((frame_len >> 24) & 0xff),
  };
  out->append(len_bytes, 4);
  out->append(*compressed);
  return absl::OkStatus();
}

void StreamingDeltaSeriesProtoConverter::AppendEndOfStreamFrame(
    std::string* out) {
  if (out == nullptr) return;
  constexpr char kEndOfStreamFrame[4] = {0, 0, 0, 0};
  out->append(kEndOfStreamFrame, sizeof(kEndOfStreamFrame));
}

DeltaSeriesProtoConverter::DeltaSeriesProtoConverter(
    const Trace* trace, CounterExtractor counter_extractor,
    const DeltaSeriesProtoConversionOptions& options,
    RawDataArgsExtractor raw_data_args_extractor)
    : trace_(trace),
      name_cache_(trace),
      counter_extractor_(std::move(counter_extractor)),
      raw_data_args_extractor_(
          raw_data_args_extractor
              ? std::move(raw_data_args_extractor)
              : internal::MakeDefaultRawDataArgsExtractor<RawData>()),
      options_(options) {}

xprof::TraceMetadata DeltaSeriesProtoConverter::GetTraceMetadata() const {
  xprof::TraceMetadata metadata;
  std::vector<uint32_t> sorted_device_ids;
  sorted_device_ids.reserve(trace_->devices_size());
  for (const auto& [device_id, _] : trace_->devices()) {
    if (options_.mpmd_pipeline_view &&
        deduplicated_device_ids_.contains(device_id)) {
      continue;
    }
    sorted_device_ids.push_back(device_id);
  }
  absl::c_sort(sorted_device_ids);

  for (const uint32_t device_id : sorted_device_ids) {
    const Device& device = trace_->devices().find(device_id)->second;

    xprof::Process* process = metadata.add_processes();
    process->set_id(device_id);
    if (device.has_name()) {
      process->set_name(device.name());
    }
    if (!options_.mpmd_pipeline_view) {
      process->set_sort_index(device_id);
    } else if (!mpmd_sort_indices_.empty()) {
      const auto it = mpmd_sort_indices_.find(device_id);
      process->set_sort_index(it != mpmd_sort_indices_.end()
                                  ? it->second
                                  : kMpmdUnrankedSortIndexBase + device_id);
    }

    std::vector<uint64_t> sorted_resource_ids;
    sorted_resource_ids.reserve(device.resources_size());
    for (const auto& [resource_id, _] : device.resources()) {
      sorted_resource_ids.push_back(resource_id);
    }
    absl::c_sort(sorted_resource_ids);

    for (const uint64_t resource_id : sorted_resource_ids) {
      const Resource& resource = device.resources().find(resource_id)->second;

      xprof::Thread* thread = process->add_threads();
      thread->set_id(resource_id);
      if (resource.has_name()) {
        thread->set_name(resource.name());
      }
      if (!options_.sort_resources_by_name.contains(device_id)) {
        thread->set_sort_index(resource_id);
      }
    }
  }
  return metadata;
}

namespace internal {

void PopulateTraceEventMetadataWithTable(const TraceEvent& event,
                                         DeltaSeriesStringTable* string_table,
                                         xprof::TraceEventMetadata* metadata) {
  string_table->PopulateFlowAndGroupMetadata(
      event.has_flow_id() ? std::optional<uint64_t>(event.flow_id())
                          : std::nullopt,
      event.has_flow_category() ? std::optional<uint32_t>(event.flow_category())
                                : std::nullopt,
      event.has_group_id() ? std::optional<int64_t>(event.group_id())
                           : std::nullopt,
      metadata);
  if (event.has_serial()) {
    metadata->set_serial(event.serial());
  }
}

HloExtractionTrackKind ClassifyHloExtractionTrack(absl::string_view res_name) {
  if (res_name == tsl::profiler::kXlaOpLineName) {
    return HloExtractionTrackKind::kXlaOps;
  }
  if (IsDataMotionLayerThreadName(res_name)) {
    return HloExtractionTrackKind::kDataMotionLayer;
  }
  return HloExtractionTrackKind::kRegular;
}

bool ExtractHloStringsFromRawData(
    const Trace& trace, const TraceEvent& event,
    HloExtractionTrackKind track_kind, absl::string_view event_name,
    absl::FunctionRef<const TraceEventArguments*(absl::string_view)>
        parse_raw_data_args,
    std::string* out_hlo_op, std::string* out_hlo_module) {
  out_hlo_op->clear();
  out_hlo_module->clear();
  if (!event.has_raw_data()) return false;
  const absl::string_view raw_bytes = event.raw_data();
  if (raw_bytes.empty()) return false;

  static const absl::string_view kHloOpArgName =
      tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloOp);
  static const absl::string_view kHloModuleArgName =
      tsl::profiler::GetStatTypeStr(tsl::profiler::StatType::kHloModule);
  // Protobuf encodes `TraceEventArguments::Argument::name` verbatim on the
  // wire. Short-circuit before parsing `RawData` whenever the required key
  // substrings are absent: XLA Ops requires `hlo_op` or `hlo_module`, while
  // regular tracks require both.
  if (!MayContainRequiredHloWireKeys(raw_bytes, track_kind, kHloOpArgName,
                                     kHloModuleArgName)) {
    return false;
  }

  const TraceEventArguments* const args = parse_raw_data_args(raw_bytes);
  if (args == nullptr) return false;

  DecodedHloArgs decoded;
  DecodeHloArguments(trace, *args, track_kind, kHloOpArgName, kHloModuleArgName,
                     &decoded);
  if (track_kind == HloExtractionTrackKind::kRegular &&
      (decoded.hlo_op.empty() || decoded.hlo_module.empty())) {
    return false;
  }

  const absl::string_view chosen_op =
      SelectHloOp(decoded, track_kind, event_name);
  if (!chosen_op.empty()) {
    out_hlo_op->assign(chosen_op.data(), chosen_op.size());
  }
  FormatHloModuleWithSuffix(decoded, out_hlo_module);
  return !out_hlo_op->empty() || !out_hlo_module->empty();
}

}  // namespace internal

void DeltaSeriesProtoConverter::PopulateTraceEventMetadata(
    const TraceEvent& event, xprof::TraceEventMetadata* metadata) {
  internal::PopulateTraceEventMetadataWithTable(event, &string_table_,
                                                metadata);
}

void DeltaSeriesProtoConverter::MaybeEmitSparseEventMetadata(
    const TraceEvent& event, absl::string_view event_name,
    internal::HloExtractionTrackKind track_kind,
    std::optional<uint64_t> flow_id, uint32_t emitted_index,
    xprof::TraceEventSeries* series) {
  uint32_t hlo_op_ref = 0;
  uint32_t hlo_module_ref = 0;
  const bool has_hlo_refs =
      event.has_raw_data() && !event.raw_data().empty() &&
      internal::ExtractHloStringsFromRawData(
          *trace_, event, track_kind, event_name, raw_data_args_extractor_,
          &scratch_hlo_op_, &scratch_hlo_module_);
  if (has_hlo_refs) {
    if (!scratch_hlo_op_.empty()) {
      hlo_op_ref = string_table_.InternString(scratch_hlo_op_);
    }
    if (!scratch_hlo_module_.empty()) {
      hlo_module_ref = string_table_.InternString(scratch_hlo_module_);
    }
  }
  const bool has_non_generic_category =
      internal::HasNonGenericFlowCategory(event);
  if (!flow_id.has_value() && event.serial() == 0 &&
      !has_non_generic_category && !has_hlo_refs) {
    return;
  }
  xprof::TraceEventMetadata* meta = series->add_event_metadata();
  string_table_.PopulateFlowAndGroupMetadata(
      flow_id,
      has_non_generic_category ? std::optional<uint32_t>(event.flow_category())
                               : std::nullopt,
      /*group_id=*/std::nullopt, meta);
  if (event.has_serial()) {
    meta->set_serial(event.serial());
  }
  if (hlo_op_ref != 0) meta->set_hlo_op_ref(hlo_op_ref);
  if (hlo_module_ref != 0) meta->set_hlo_module_ref(hlo_module_ref);
  series->add_event_metadata_indices(emitted_index);
}

void DeltaSeriesProtoConverter::AddCompleteEventTrack(
    uint32_t pid, uint64_t tid, const TraceEventTrack& events,
    xprof::TraceDataResponse* response) {
  const auto intern_fn = [this](absl::string_view s) {
    return MaybeInternString(s);
  };

  if (!options_.sparse_event_metadata) {
    for (size_t chunk_start = 0; chunk_start < events.size();
         chunk_start += kMaxEventsPerSeriesChunk) {
      const size_t chunk_end =
          std::min(events.size(), chunk_start + kMaxEventsPerSeriesChunk);
      xprof::TraceEventSeries* series = response->add_complete_events();
      xprof::TraceEventSeriesMetadata* series_metadata =
          series->mutable_metadata();
      series_metadata->set_process_id(pid);
      series_metadata->set_thread_id(tid);

      uint64_t last_timestamp = 0;
      for (size_t i = chunk_start; i < chunk_end; ++i) {
        const TraceEvent* event = events[i];
        const uint64_t current_timestamp = event->timestamp_ps();
        series->add_deltas(current_timestamp - last_timestamp);
        last_timestamp = current_timestamp;

        series->add_durations(event->duration_ps());
        series->add_name_refs(
            name_cache_.ResolveAndIntern(*event, intern_fn).interned_id);
        PopulateTraceEventMetadata(*event, series->add_event_metadata());
      }
    }
    return;
  }

  const absl::string_view res_name =
      internal::ResolveResourceName(*trace_, pid, tid);
  const internal::HloExtractionTrackKind track_kind =
      internal::ClassifyHloExtractionTrack(res_name);

  for (size_t chunk_start = 0; chunk_start < events.size();
       chunk_start += kMaxEventsPerSeriesChunk) {
    const size_t chunk_end =
        std::min(events.size(), chunk_start + kMaxEventsPerSeriesChunk);
    xprof::TraceEventSeries* series = response->add_complete_events();
    xprof::TraceEventSeriesMetadata* series_metadata =
        series->mutable_metadata();
    series_metadata->set_process_id(pid);
    series_metadata->set_thread_id(tid);

    uint64_t last_timestamp = 0;
    internal::GroupIdRunBuilder group_id_runs;
    uint32_t emitted_index = 0;

    for (size_t i = chunk_start; i < chunk_end; ++i) {
      const TraceEvent* event = events[i];
      const uint64_t current_timestamp = event->timestamp_ps();
      series->add_deltas(current_timestamp - last_timestamp);
      last_timestamp = current_timestamp;

      series->add_durations(event->duration_ps());

      const internal::TraceNameRefInternCache::CachedEntry resolved_name =
          name_cache_.ResolveAndIntern(*event, intern_fn);
      series->add_name_refs(resolved_name.interned_id);
      const absl::string_view event_name = resolved_name.resolved_name;

      group_id_runs.Add(event->has_group_id() ? event->group_id() : -1, series);

      MaybeEmitSparseEventMetadata(
          *event, event_name, track_kind,
          event->flow_id() != 0 ? std::optional<uint64_t>(event->flow_id())
                                : std::nullopt,
          emitted_index, series);
      ++emitted_index;
    }
    group_id_runs.Flush(series);
  }
}

void DeltaSeriesProtoConverter::AddAsyncEventTrack(
    uint32_t pid, absl::string_view name, const TraceEventTrack& events,
    xprof::TraceDataResponse* response) {
  xprof::TraceEventSeries* series = response->add_async_events();
  xprof::TraceEventSeriesMetadata* series_metadata = series->mutable_metadata();
  series_metadata->set_process_id(pid);
  series_metadata->set_name_ref(MaybeInternString(name));

  if (!options_.sparse_event_metadata) {
    uint64_t last_timestamp = 0;
    for (const TraceEvent* event : events) {
      const uint64_t current_timestamp = event->timestamp_ps();
      series->add_deltas(current_timestamp - last_timestamp);
      last_timestamp = current_timestamp;

      series->add_durations(event->duration_ps());
      PopulateTraceEventMetadata(*event, series->add_event_metadata());
    }
    return;
  }

  uint64_t last_timestamp = 0;
  internal::GroupIdRunBuilder group_id_runs;
  uint32_t emitted_index = 0;

  for (size_t i = 0; i < events.size(); ++i) {
    const TraceEvent* event = events[i];
    const uint64_t current_timestamp = event->timestamp_ps();
    series->add_deltas(current_timestamp - last_timestamp);
    last_timestamp = current_timestamp;

    series->add_durations(event->duration_ps());

    group_id_runs.Add(event->has_group_id() ? event->group_id() : -1, series);

    const bool is_flow_endpoint =
        event->flow_id() != 0 &&
        (event->duration_ps() == 0 ||
         event->flow_entry_type() != TraceEvent::FLOW_NONE);
    MaybeEmitSparseEventMetadata(
        *event, name, internal::HloExtractionTrackKind::kRegular,
        is_flow_endpoint ? std::optional<uint64_t>(event->flow_id())
                         : std::nullopt,
        emitted_index, series);
    ++emitted_index;
  }
  group_id_runs.Flush(series);
}

void DeltaSeriesProtoConverter::AddCounterEventTrack(
    uint32_t pid, absl::string_view name, const TraceEventTrack& events,
    xprof::TraceDataResponse* response) {
  xprof::TraceEventSeries* series = response->add_counter_events();
  xprof::TraceEventSeriesMetadata* series_metadata = series->mutable_metadata();
  series_metadata->set_process_id(pid);
  series_metadata->set_name_ref(MaybeInternString(name));

  bool has_series_name = false;
  std::string series_name;
  uint64_t last_timestamp = 0;
  for (const TraceEvent* event : events) {
    uint64_t current_timestamp = event->timestamp_ps();
    series->add_deltas(current_timestamp - last_timestamp);
    last_timestamp = current_timestamp;

    xprof::TraceEventMetadata* event_metadata = series->add_event_metadata();
    if (event->has_raw_data()) {
      counter_extractor_(event->raw_data(), event_metadata,
                         has_series_name ? nullptr : &series_name);
      if (!has_series_name && !series_name.empty()) {
        series_metadata->set_event_stats_ref(MaybeInternString(series_name));
        has_series_name = true;
      }
    }
  }
}

}  // namespace profiler
}  // namespace tensorflow
