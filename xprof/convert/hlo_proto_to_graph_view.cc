/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/hlo_proto_to_graph_view.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/text_format.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/tsl/platform/statusor.h"
#ifdef PLATFORM_GOOGLE
#include "tensorflow/compiler/mlir/lite/experimental/google/tooling/hlo_adapter/direct_hlo_to_json_graph_convert.h"
#endif  // PLATFORM_GOOGLE
#include "nlohmann/json_fwd.hpp"
#include "nlohmann/json.hpp"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/tsl/platform/errors.h"
#include "xprof/convert/graphviz_helper.h"
#include "xprof/convert/tool_options.h"
#include "xprof/utils/hlo_module_utils.h"
#include "xprof/utils/hlo_proto_to_module.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::tsl::StatusOr;
using ::tsl::errors::InvalidArgument;
using ::xla::HloComputation;
using ::xla::HloInstruction;
using ::xla::HloModule;
using ::xla::HloPrintOptions;
using ::xla::HloProto;
using ::xla::HloRenderOptions;
using ::xla::RenderedGraphFormat;

constexpr char kCenterNodeKey[] = "centerNode";

void CleanUpHloModuleForGraphviz(HloModule* hlo_module) {
  // Infeed config is escaped serialized proto, and graphviz server complains.
  for (HloComputation* computation : hlo_module->computations()) {
    for (HloInstruction* inst : computation->instructions()) {
      if (inst->opcode() == xla::HloOpcode::kInfeed) {
        inst->set_infeed_config("");
      } else if (inst->opcode() == xla::HloOpcode::kOutfeed) {
        inst->set_outfeed_config("");
      }
    }
  }
}

#ifdef PLATFORM_GOOGLE
// Add a custom group node on the graph level, for the center node chosen by the
// user set its attributes like `id`, `name` or `opcode` in `graph_json`.
void AddCenterNodeMetadata(nlohmann::json& graph_json, std::string id,
                           absl::string_view name, absl::string_view opcode) {
  nlohmann::json centerGroupNodeAttributes;
  centerGroupNodeAttributes["name"] = name;
  centerGroupNodeAttributes["id"] = id;
  if (!opcode.empty()) {
    centerGroupNodeAttributes["opcode"] = opcode;
  }
  // Follow ModelExplorer's Graph typing: GraphCollectionFromBuiltinAdapters
  graph_json[0]["subgraphs"][0]["groupNodeAttributes"][kCenterNodeKey] =
      centerGroupNodeAttributes;
}
#endif  // PLATFORM_GOOGLE

void AddGraphMetadata(std::string& graph_json_str,
                      const HloInstruction& instr) {
#ifdef PLATFORM_GOOGLE
  nlohmann::json graph_json = nlohmann::json::parse(graph_json_str);
  // 1. Fusion instruction is represented as a layer on client, use its
  // pinned node as the center node, id of the pinned node is the fusion name.
  // 2. Other instructions are represented as nodes on client, use iteself as
  // the center node, where node id is the instruction name.
  std::string id = absl::StrCat(instr.name());
  AddCenterNodeMetadata(graph_json, id, instr.name(),
                        HloOpcodeString(instr.opcode()));
  graph_json_str = graph_json.dump();
#endif  // PLATFORM_GOOGLE
}

void AddGraphMetadata(std::string& graph_json_str, const HloComputation& comp) {
#ifdef PLATFORM_GOOGLE
  nlohmann::json graph_json = nlohmann::json::parse(graph_json_str);
  // Computation is represented as a layer on client, use its pinned node as the
  // center node,id of the pinned node is the computation name.
  AddCenterNodeMetadata(graph_json, absl::StrCat(comp.name()), comp.name(), "");
  graph_json_str = graph_json.dump();
#endif  // PLATFORM_GOOGLE
}

// This function does the same thing as Plot() but uses the ModelExplorer
// instead of graphviz.
absl::StatusOr<std::string> PlotMe(std::unique_ptr<HloModule> module,
                                   const std::string& node_name,
                                   int graph_width) {
  if (node_name.empty()) {
    // This should not happen.
    return InvalidArgument("node_name should not be empty");
  }
  // Find the node with the given name.
  const HloInstruction* instr = FindInstruction(*module, node_name);
  const HloComputation* comp = FindComputation(*module, node_name);

  if (!instr && !comp) {
    return InvalidArgument(
        absl::StrCat("Couldn't find HloInstruction or HloComputation named ",
                     node_name, "."));
  }
  // Generate the graph and print the resulting string.
  absl::StatusOr<std::string> graph_handle;
  std::string graph_json_str;
// b/360874576: Enable when the adapter is open sourced.
#ifdef PLATFORM_GOOGLE
  if (comp) {
    graph_handle = tooling::visualization_client::HloGraphAdapter(*comp);
  } else {
    graph_handle =
        tooling::visualization_client::HloGraphAdapter(*instr, graph_width);
  }
#endif  // PLATFORM_GOOGLE
  if (graph_handle.ok()) {
    VLOG(1) << graph_handle.value();
    graph_json_str = graph_handle.value();
    if (comp) {
      AddGraphMetadata(graph_json_str, *comp);
    } else {
      AddGraphMetadata(graph_json_str, *instr);
    }
    return graph_json_str;
  } else {
    LOG(ERROR) << "Unable to render graph: " << graph_handle.status();
  }

  return graph_handle;
}

absl::StatusOr<std::string> Plot(std::unique_ptr<HloModule> module,
                                 const std::string& node_name, int graph_width,
                                 const HloRenderOptions& render_options,
                                 const RenderedGraphFormat& format) {
  if (node_name.empty()) {
    // This should not happen.
    return InvalidArgument("node_name should not be empty");
  }
  // Find the node with the given name.
  const HloInstruction* instr = FindInstruction(*module, node_name);
  const HloComputation* comp = FindComputation(*module, node_name);
  if (!instr && !comp) {
    return InvalidArgument(
        absl::StrCat("Couldn't find HloInstruction or HloComputation named ",
                     node_name, "."));
  }
  // Generate the graph and print the resulting string.
  absl::StatusOr<std::string> graph_handle;

  CleanUpHloModuleForGraphviz(module.get());
  if (comp) {
    graph_handle =
        RenderGraphView(*comp, "", comp->parent()->config().debug_options(),
                        format, render_options);
  } else {
    graph_handle = RenderGraphNeighborhoodAround(*instr, graph_width, format,
                                                 render_options);
  }
  if (graph_handle.ok()) {
    VLOG(1) << graph_handle.value();
  } else {
    LOG(ERROR) << "Unable to render graph: " << graph_handle.status();
  }

  return graph_handle;
}
}  // namespace

absl::StatusOr<std::string> GetNodeStyles() {
  std::vector<xla::HloOpcode> async_op_codes = {xla::HloOpcode::kAsyncStart,
                                                xla::HloOpcode::kAsyncUpdate,
                                                xla::HloOpcode::kAsyncDone};
  std::vector<xla::HloOpcode> brown_op_codes = {
      xla::HloOpcode::kAllGather,
      xla::HloOpcode::kAllGatherStart,
      xla::HloOpcode::kAllGatherDone,
      xla::HloOpcode::kAllReduce,
      xla::HloOpcode::kReduceScatter,
      xla::HloOpcode::kAllReduceStart,
      xla::HloOpcode::kAllReduceDone,
      xla::HloOpcode::kAllToAll,
      xla::HloOpcode::kCollectiveBroadcast,
      xla::HloOpcode::kCollectivePermute,
      xla::HloOpcode::kCollectivePermuteStart,
      xla::HloOpcode::kCollectivePermuteDone,
      xla::HloOpcode::kInfeed,
      xla::HloOpcode::kOutfeed,
      xla::HloOpcode::kPartitionId,
      xla::HloOpcode::kRecv,
      xla::HloOpcode::kRecvDone,
      xla::HloOpcode::kSend,
      xla::HloOpcode::kSendDone,
      xla::HloOpcode::kReplicaId};
  std::vector<xla::HloOpcode> dark_blue_op_codes = {
      xla::HloOpcode::kConvolution, xla::HloOpcode::kDot, xla::HloOpcode::kFft,
      xla::HloOpcode::kTriangularSolve, xla::HloOpcode::kCholesky};
  std::vector<xla::HloOpcode> dark_green_op_codes = {
      xla::HloOpcode::kCall, xla::HloOpcode::kConditional,
      xla::HloOpcode::kCustomCall, xla::HloOpcode::kWhile};
  std::vector<xla::HloOpcode> gray_op_codes = {
      xla::HloOpcode::kDomain, xla::HloOpcode::kFusion, xla::HloOpcode::kMap,
      xla::HloOpcode::kGetDimensionSize, xla::HloOpcode::kSetDimensionSize};
  std::vector<xla::HloOpcode> green_op_codes = {
      xla::HloOpcode::kConcatenate, xla::HloOpcode::kDynamicSlice,
      xla::HloOpcode::kReshape,     xla::HloOpcode::kDynamicReshape,
      xla::HloOpcode::kReverse,     xla::HloOpcode::kTranspose,
      xla::HloOpcode::kCopy,        xla::HloOpcode::kCopyStart,
      xla::HloOpcode::kCopyDone};
  std::vector<xla::HloOpcode> orange_op_codes = {xla::HloOpcode::kParameter};
  std::vector<xla::HloOpcode> purple_op_codes = {
      xla::HloOpcode::kBatchNormGrad,     xla::HloOpcode::kBatchNormInference,
      xla::HloOpcode::kBatchNormTraining, xla::HloOpcode::kReduce,
      xla::HloOpcode::kReduceWindow,      xla::HloOpcode::kScatter,
      xla::HloOpcode::kSelectAndScatter,  xla::HloOpcode::kGather};
  std::vector<xla::HloOpcode> yellow_op_codes = {
      xla::HloOpcode::kBroadcast, xla::HloOpcode::kDynamicUpdateSlice};

  auto OpCodesToNames =
      [&](std::vector<xla::HloOpcode> op_codes) -> std::string {
    std::string op_names = "";
    for (const auto& op_code : op_codes) {
      if (!op_names.empty()) {
        op_names += ",";
      }
      op_names += std::string(xla::HloOpcodeString(op_code));
    }
    return op_names;
  };

  return absl::StrReplaceAll(
      R"json({
      "kBlue": "$asyncOpNames",
      "kBrown": "$brownOpNames",
      "kDarkBlue": "$darkBlueOpNames",
      "kDarkGreen": "$darkGreenOpNames",
      "kGray": "$grayOpNames",
      "kGreen": "$greenOpNames",
      "kOrange": "$orangeOpNames",
      "kPurple": "$purpleOpNames",
      "kYellow": "$yellowOpNames"
    })json",
      {
          {"$asyncOpNames", OpCodesToNames(async_op_codes)},
          {"$brownOpNames", OpCodesToNames(brown_op_codes)},
          {"$darkBlueOpNames", OpCodesToNames(dark_blue_op_codes)},
          {"$darkGreenOpNames", OpCodesToNames(dark_green_op_codes)},
          {"$grayOpNames", OpCodesToNames(gray_op_codes)},
          {"$greenOpNames", OpCodesToNames(green_op_codes)},
          {"$orangeOpNames", OpCodesToNames(orange_op_codes)},
          {"$purpleOpNames", OpCodesToNames(purple_op_codes)},
          {"$yellowOpNames", OpCodesToNames(yellow_op_codes)},
      });
}

absl::StatusOr<GraphViewerParams> ParseGraphViewerParams(
    const ToolOptions& options) {
  GraphViewerParams params;
  std::optional<std::string> type = GetParam<std::string>(options, "type");
  if (!type.has_value()) {
    return InvalidArgument("Graph viewer must provide a type option.");
  }
  auto valid_types = {
      kGraphTypeName,    kJsonTypeName,    kProtoTypeName, kProtoTextTypeName,
      kShortTxtTypeName, kLongTxtTypeName, kAdjacentNodes,
  };
  if (std::find(valid_types.begin(), valid_types.end(), type.value()) ==
      valid_types.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown graph viewer type option: ", type.value()));
  }

  // For graph type.
  if (type == kGraphTypeName) {
    params.type = type.value();
    if (std::optional<std::string> node_name =
            GetParam<std::string>(options, "node_name")) {
      params.node_name = node_name.value();
    }

    params.graph_width =
        GetParamWithDefault<int>(options, "graph_width", kDefaultWidth);
    params.render_options.show_backend_config = GetParamWithDefault<int>(
        options, "show_metadata", kDefaultShowMetadata);
    params.render_options.show_fusion_subcomputations =
        !GetParamWithDefault<int>(options, "merge_fusion", kDefaultMergeFusion);
    params.format = GetRenderFormat(GetParamWithDefault<std::string>(
        options, "format", kDefaultFormatString));

    return params;
  } else if (type == kAdjacentNodes) {
    params.type = type.value();
    if (std::optional<std::string> node_name =
            GetParam<std::string>(options, "node_name")) {
      params.node_name = node_name.value();
    }
    return params;
  }

  // For txt type.
  params.type = type.value();
  params.show_metadata =
      GetParamWithDefault(options, "show_metadata", kDefaultShowMetadata);
  params.verbose = type == kLongTxtTypeName;
  return params;
}

xla::RenderedGraphFormat GetRenderFormat(const std::string& format_string) {
  if (format_string == "html") {
    return xla::RenderedGraphFormat::kHtml;
  } else if (format_string == "dot") {
    return xla::RenderedGraphFormat::kDot;
  } else if (format_string == "url") {
    return xla::RenderedGraphFormat::kUrl;
  } else {
    LOG(ERROR) << "Invalid graph format argument: " << format_string
               << ", fallback to default url";
    return xla::RenderedGraphFormat::kUrl;
  }
}

absl::StatusOr<std::string> GetAdjacentNodes(const HloProto& hlo_proto,
                                             const std::string& node_name) {
  if (node_name.empty()) {
    return absl::InvalidArgumentError("node_name should not be empty");
  }
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ConvertHloProtoToModule(hlo_proto));
  const xla::HloInstruction* hlo_instruction =
      FindInstruction(*hlo_module, node_name);
  const xla::HloComputation* hlo_computation =
      FindComputation(*hlo_module, node_name);
  nlohmann::json json;
  // Recursively seeking for first visible operands and users in trace viewer or
  // graph viewer.
  std::function<void(std::vector<std::string>&, const xla::HloInstruction*)>
      find_operands = [&](std::vector<std::string>& operand_names,
                          const xla::HloInstruction* hlo_instruction) {
        for (const auto& operand : hlo_instruction->operands()) {
          if (absl::StartsWith(operand->name(), "get-tuple-element")) {
            find_operands(operand_names, operand);
            continue;
          }
          operand_names.push_back(absl::StrCat(operand->name()));
        }
      };
  std::function<void(std::vector<std::string>&, const xla::HloInstruction*)>
      find_users = [&](std::vector<std::string>& user_names,
                       const xla::HloInstruction* hlo_instruction) {
        for (const auto& user : hlo_instruction->users()) {
          if (absl::StartsWith(user->name(), "get-tuple-element")) {
            find_users(user_names, user);
            continue;
          }
          user_names.push_back(absl::StrCat(user->name()));
        }
      };
  if (hlo_instruction) {
    std::vector<std::string> operand_names;
    std::vector<std::string> user_names;
    find_operands(operand_names, hlo_instruction);
    find_users(user_names, hlo_instruction);
    json["operand_names"] = operand_names;
    json["consumer_names"] = user_names;
    return json.dump();
  } else if (hlo_computation) {
    return absl::UnimplementedError(
        "GetAdjacentNodes is not implemented for HloComputation.");
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Couldn't find HloInstruction or HloComputation named ", node_name, "."));
}

absl::StatusOr<std::string> ConvertHloProtoToGraph(
    const HloProto& hlo_proto, const std::string& node_name, int graph_width,
    const HloRenderOptions& render_options, const RenderedGraphFormat& format) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ConvertHloProtoToModule(hlo_proto));
  return Plot(std::move(hlo_module), node_name, graph_width, render_options,
              format);
}

absl::StatusOr<std::string> ConvertHloProtoToMeGraph(
    const HloProto& hlo_proto, const std::string& node_name, int graph_width) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ConvertHloProtoToModule(hlo_proto));
  return PlotMe(std::move(hlo_module), node_name, graph_width);
}

absl::StatusOr<std::string> PrintJson(const xla::HloProto& proto) {
  return absl::UnimplementedError("Not implemented");
}

absl::StatusOr<std::string> PrintPbTxt(const xla::HloProto& hlo_proto) {
  google::protobuf::TextFormat::Printer printer;
  printer.SetHideUnknownFields(true);
  std::string output;
  if (!printer.PrintToString(hlo_proto, &output)) {
    return absl::InternalError(
        "Unable to convert to xla.HloProto pbtxt representation");
  }
  return output;
}

absl::StatusOr<std::string> ConvertHloProtoToStringView(
    const HloProto& hlo_proto, absl::string_view type, bool verbose,
    bool metadata) {
  if (type == kJsonTypeName) {
    return PrintJson(hlo_proto);
  } else if (type == kProtoTypeName) {
    return hlo_proto.SerializeAsString();
  } else if (type == kProtoTextTypeName) {
    return PrintPbTxt(hlo_proto);
  }
  // for short/long_txt
  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> hlo_module,
                      ConvertHloProtoToModule(hlo_proto));
  HloPrintOptions options;
  if (!verbose) {
    options = HloPrintOptions::ShortParsable();
  }
  options.set_print_large_constants(verbose);
  options.set_print_metadata(metadata);
  return hlo_module->ToString(options);
}

absl::StatusOr<std::string> RenderGraphView(
    const xla::HloComputation& computation, absl::string_view label,
    const xla::DebugOptions& debug_options, xla::RenderedGraphFormat format,
    xla::HloRenderOptions hlo_render_options) {
  auto rendered_dot =
      xla::RenderGraph(computation, label, debug_options,
                       RenderedGraphFormat::kDot, hlo_render_options);
  if (!rendered_dot.ok()) {
    return rendered_dot.status();
  }
  return WrapDotInFormat(rendered_dot.value(), format);
}

absl::StatusOr<std::string> RenderGraphNeighborhoodAround(
    const xla::HloInstruction& node, int radius,
    xla::RenderedGraphFormat format, xla::HloRenderOptions hlo_render_options,
    const absl::flat_hash_set<const xla::HloInstruction*>& boundary) {
  auto rendered_dot = xla::RenderNeighborhoodAround(
      node, radius, RenderedGraphFormat::kDot, hlo_render_options, boundary);
  if (!rendered_dot.ok()) {
    return rendered_dot.status();
  }
  return WrapDotInFormat(rendered_dot.value(), format);
}

}  // namespace profiler
}  // namespace tensorflow
