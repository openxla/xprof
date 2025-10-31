#include "xprof/convert/smart_suggestion/special_op_rule.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xprof/convert/smart_suggestion/signal_provider.h"
#include "xprof/convert/smart_suggestion/tool_data_provider.h"
#include "plugin/xprof/protobuf/smart_suggestion.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Return;
using ::testing::status::IsOkAndHolds;

// Mock ToolDataProvider
class MockToolDataProvider : public ToolDataProvider {
 public:
  MOCK_METHOD(absl::StatusOr<const OverviewPage*>, GetOverviewPage, (),
              (override));
  MOCK_METHOD(absl::StatusOr<const InputPipelineAnalysisResult*>,
              GetInputPipelineAnalysisResult, (), (override));
  MOCK_METHOD(absl::StatusOr<std::vector<float>>,
              GetEventTimeFractionEachStep, (const std::string&), (override));
};

TEST(SpecialOpRuleTest, MeetsConditions) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  // Average is (0.15+0.25)/2 = 0.2, which is 20%. This is > 10%.
  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionEachStep(kSpecialOpName))
      .WillRepeatedly(Return(std::vector<float>{0.15, 0.25}));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  SpecialOpRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(testing::Not(Eq(std::nullopt))));
  EXPECT_EQ((*suggestion)->rule_name(), "SpecialOpRule");
  EXPECT_THAT((*suggestion)->suggestion_text(),
     HasSubstr("20.0% of each step time"));
}

TEST(SpecialOpRuleTest, NotSpecialOpBound) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  // Average is (0.01+0.02)/2 = 0.015, which is 1.5%. This is < 10%.
  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionEachStep(kSpecialOpName))
      .WillRepeatedly(Return(std::vector<float>{0.01, 0.02}));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  SpecialOpRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

TEST(SpecialOpRuleTest, ErrorFetchingPercentile) {
  auto mock_tool_data_provider = std::make_unique<MockToolDataProvider>();
  EXPECT_CALL(*mock_tool_data_provider,
              GetEventTimeFractionEachStep(kSpecialOpName))
      .WillRepeatedly(Return(absl::InternalError("Failed to get percentile")));

  SignalProvider signal_provider(std::move(mock_tool_data_provider));
  SpecialOpRule rule;

  absl::StatusOr<std::optional<SmartSuggestion>> suggestion =
      rule.Apply(signal_provider);
  EXPECT_THAT(suggestion, IsOkAndHolds(Eq(std::nullopt)));
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
