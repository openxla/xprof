#ifndef THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SIGNAL_EXTRACTOR_H_
#define THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SIGNAL_EXTRACTOR_H_

#include <any>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "plugin/xprof/protobuf/overview_page.pb.h"

namespace tensorflow {
namespace profiler {

// Struct to hold all extracted signals.
// Users can add any signal type here.
struct Signals {
  absl::flat_hash_map<std::string, std::any> data;

  template <typename T>
  void Set(absl::string_view key, T value) {
    data[key] = value;
  }

  template <typename T>
  absl::StatusOr<T> Get(absl::string_view key) const {
    auto it = data.find(key);
    if (it == data.end()) {
      return absl::NotFoundError("Signal not found");
    }
    try {
      return std::any_cast<T>(it->second);
    } catch (const std::bad_any_cast& e) {
      return absl::InternalError("Failed to cast signal");
    }
  }
};

// Interface for signal extractors.
class SignalExtractorInterface {
 public:
  virtual ~SignalExtractorInterface() = default;
  // Returns the name of the signal this extractor provides.
  virtual std::string GetSignalName() const = 0;
  // Extracts the signal from the given source and adds it to the Signals
  // struct.
  virtual absl::Status Extract(const std::any& source, Signals& signals) = 0;
};

// Registration system for signal extractors.
class SignalExtractorRegistry {
 public:
  static SignalExtractorRegistry& GetInstance();

  absl::Status Register(std::unique_ptr<SignalExtractorInterface> extractor);
  absl::StatusOr<SignalExtractorInterface*> GetExtractor(
      absl::string_view signal_name) const;
  const absl::flat_hash_map<std::string,
                            std::unique_ptr<SignalExtractorInterface>>&
  GetAllExtractors() const;

 private:
  SignalExtractorRegistry() = default;
  absl::flat_hash_map<std::string, std::unique_ptr<SignalExtractorInterface>>
      extractors_;
};

// Example Extractor for HBM utilization from OverviewPageAnalysis.
class HbmUtilizationExtractor : public SignalExtractorInterface {
 public:
  std::string GetSignalName() const override {
    return "hbm_utilization_percent";
  }
  absl::Status Extract(const std::any& source, Signals& signals) override;
};

// Example Extractor for MXU utilization from OverviewPageAnalysis.
class MxuUtilizationExtractor : public SignalExtractorInterface {
 public:
  std::string GetSignalName() const override {
    return "mxu_utilization_percent";
  }
  absl::Status Extract(const std::any& source, Signals& signals) override;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // THIRD_PARTY_XPROF_CONVERT_SMART_SUGGESTION_SIGNAL_EXTRACTOR_H_
