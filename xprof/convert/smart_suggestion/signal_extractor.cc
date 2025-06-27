#include "xprof/convert/smart_suggestion/signal_extractor.h"

#include <any>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"

namespace tensorflow {
namespace profiler {

// SignalExtractorRegistry Implementation
SignalExtractorRegistry& SignalExtractorRegistry::GetInstance() {
  static SignalExtractorRegistry instance;
  return instance;
}

absl::Status SignalExtractorRegistry::Register(
    std::unique_ptr<SignalExtractorInterface> extractor) {
  std::string signal_name = extractor->GetSignalName();
  if (extractors_.contains(signal_name)) {
    return absl::AlreadyExistsError(absl::StrCat(
        "Extractor for signal '", signal_name, "' already registered"));
  }
  extractors_[signal_name] = std::move(extractor);
  return absl::OkStatus();
}

absl::StatusOr<SignalExtractorInterface*> SignalExtractorRegistry::GetExtractor(
    absl::string_view signal_name) const {
  auto it = extractors_.find(signal_name);
  if (it == extractors_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Extractor for signal '", signal_name, "' not found"));
  }
  return it->second.get();
}

const absl::flat_hash_map<std::string,
                          std::unique_ptr<SignalExtractorInterface>>&
SignalExtractorRegistry::GetAllExtractors() const {
  return extractors_;
}

// HbmUtilizationExtractor Implementation
absl::Status HbmUtilizationExtractor::Extract(const std::any& source,
                                              Signals& signals) {
  try {
    const auto& analysis = std::any_cast<const OverviewPageAnalysis&>(source);
    signals.Set(GetSignalName(),
                analysis.memory_bw_utilization_relative_to_hw_limit_percent());
    return absl::OkStatus();
  } catch (const std::bad_any_cast& e) {
    return absl::InvalidArgumentError(
        "Invalid source type for HbmUtilizationExtractor, expected "
        "OverviewPageAnalysis");
  }
}

// MxuUtilizationExtractor Implementation
absl::Status MxuUtilizationExtractor::Extract(const std::any& source,
                                              Signals& signals) {
  try {
    const auto& analysis = std::any_cast<const OverviewPageAnalysis&>(source);
    signals.Set(GetSignalName(), analysis.mxu_utilization_percent());
    return absl::OkStatus();
  } catch (const std::bad_any_cast& e) {
    return absl::InvalidArgumentError(
        "Invalid source type for MxuUtilizationExtractor, expected "
        "OverviewPageAnalysis");
  }
}

// Register all extractors
static bool kRegisterExtractors = [] {
  auto& registry = SignalExtractorRegistry::GetInstance();
  registry.Register(std::make_unique<HbmUtilizationExtractor>()).IgnoreError();
  registry.Register(std::make_unique<MxuUtilizationExtractor>()).IgnoreError();
  return true;
}();

}  // namespace profiler
}  // namespace tensorflow
