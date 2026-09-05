/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

#include "xprof/convert/events_db/record_consumer.h"

#include <optional>
#include <string>
#include <utility>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xprof/convert/events_db/schema.h"

namespace xprof::events_db {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::Optional;

TEST(StepControlTest, ToString) {
  EXPECT_EQ(StepControlToString(StepControl::kContinue), "CONTINUE");
  EXPECT_EQ(StepControlToString(StepControl::kStop), "STOP");
  EXPECT_EQ(StepControlToString(static_cast<StepControl>(999)), "UNKNOWN");
}

TEST(StepControlTest, FromStringValid) {
  EXPECT_THAT(StepControlFromString("CONTINUE"),
              IsOkAndHolds(Eq(StepControl::kContinue)));
  EXPECT_THAT(StepControlFromString("STOP"),
              IsOkAndHolds(Eq(StepControl::kStop)));
}

TEST(StepControlTest, FromStringInvalid) {
  EXPECT_THAT(StepControlFromString(""),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(StepControlFromString("invalid"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(StepControlFromString("StepControl.CONTINUE"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(StepControlFromString("kContinue"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(StepControlFromString("continue"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(StepControlFromString("StepControl.STOP"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(StepControlFromString("kStop"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(StepControlFromString("stop"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(StepControlFromString("UNKNOWN"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(StepControlTest, RoundTrip) {
  EXPECT_THAT(
      StepControlFromString(StepControlToString(StepControl::kContinue)),
      IsOkAndHolds(Eq(StepControl::kContinue)));
  EXPECT_THAT(StepControlFromString(StepControlToString(StepControl::kStop)),
              IsOkAndHolds(Eq(StepControl::kStop)));
}

TEST(ParseStatusTest, ToString) {
  EXPECT_EQ(ParseStatusToString(ParseStatus::kComplete), "COMPLETE");
  EXPECT_EQ(ParseStatusToString(ParseStatus::kStoppedEarly), "STOPPED_EARLY");
  EXPECT_EQ(ParseStatusToString(static_cast<ParseStatus>(999)), "UNKNOWN");
}

TEST(ParseStatusTest, FromStringValid) {
  EXPECT_THAT(ParseStatusFromString("COMPLETE"),
              IsOkAndHolds(Eq(ParseStatus::kComplete)));
  EXPECT_THAT(ParseStatusFromString("STOPPED_EARLY"),
              IsOkAndHolds(Eq(ParseStatus::kStoppedEarly)));
}

TEST(ParseStatusTest, FromStringInvalid) {
  EXPECT_THAT(ParseStatusFromString(""),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseStatusFromString("invalid"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseStatusFromString("ParseStatus.COMPLETE"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseStatusFromString("kComplete"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseStatusFromString("complete"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseStatusFromString("ParseStatus.STOPPED_EARLY"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseStatusFromString("kStoppedEarly"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseStatusFromString("stopped_early"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ParseStatusFromString("UNKNOWN"),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ParseStatusTest, RoundTrip) {
  EXPECT_THAT(
      ParseStatusFromString(ParseStatusToString(ParseStatus::kComplete)),
      IsOkAndHolds(Eq(ParseStatus::kComplete)));
  EXPECT_THAT(
      ParseStatusFromString(ParseStatusToString(ParseStatus::kStoppedEarly)),
      IsOkAndHolds(Eq(ParseStatus::kStoppedEarly)));
}

TEST(RecordConsumerRefTest, ConsumesViaCallOperator) {
  int call_count = 0;
  auto lambda_consumer = [&](Record&) -> absl::StatusOr<StepControl> {
    call_count++;
    return StepControl::kContinue;
  };

  RecordConsumerRef ref(lambda_consumer);
  Record record;

  EXPECT_THAT(ref(record), IsOkAndHolds(Eq(StepControl::kContinue)));
  EXPECT_THAT(ref.Consume(record), IsOkAndHolds(Eq(StepControl::kContinue)));
  EXPECT_EQ(call_count, 2);
}

TEST(RecordConsumerRefTest, ConsumesViaMemberFunction) {
  struct MethodConsumer {
    int call_count = 0;
    absl::StatusOr<StepControl> Consume(Record&) {
      call_count++;
      return StepControl::kStop;
    }
  };

  MethodConsumer consumer;
  RecordConsumerRef ref(consumer);
  Record record;

  EXPECT_THAT(ref(record), IsOkAndHolds(Eq(StepControl::kStop)));
  EXPECT_THAT(ref.Consume(record), IsOkAndHolds(Eq(StepControl::kStop)));
  EXPECT_EQ(consumer.call_count, 2);
}

TEST(RecordConsumerRefTest, PrioritizesConsumeMemberOverCallOperator) {
  struct BothConsumer {
    bool consume_called = false;
    bool operator_called = false;

    absl::StatusOr<StepControl> Consume(Record&) {
      consume_called = true;
      return StepControl::kContinue;
    }

    absl::StatusOr<StepControl> operator()(Record&) {
      operator_called = true;
      return StepControl::kStop;
    }
  };

  BothConsumer consumer;
  RecordConsumerRef ref(consumer);
  Record record;

  EXPECT_THAT(ref(record), IsOkAndHolds(Eq(StepControl::kContinue)));
  EXPECT_TRUE(consumer.consume_called);
  EXPECT_FALSE(consumer.operator_called);
}

TEST(RecordConsumerRefTest, FinalizeWithResultOutcome) {
  struct FinalizeWithResultConsumer {
    bool finalized = false;
    std::optional<ParseStatus> outcome;
    absl::Status received_status = absl::OkStatus();

    absl::StatusOr<StepControl> operator()(Record&) {
      return StepControl::kContinue;
    }

    absl::Status Finalize(const absl::StatusOr<ParseStatus>& result) {
      finalized = true;
      if (result.ok()) {
        outcome = *result;
      } else {
        received_status = result.status();
      }
      return absl::OkStatus();
    }
  };

  {
    FinalizeWithResultConsumer consumer;
    RecordConsumerRef ref(consumer);
    EXPECT_OK(ref.Finalize(ParseStatus::kComplete));
    EXPECT_TRUE(consumer.finalized);
    EXPECT_THAT(consumer.outcome, Optional(Eq(ParseStatus::kComplete)));
  }

  {
    FinalizeWithResultConsumer consumer;
    RecordConsumerRef ref(consumer);
    EXPECT_OK(ref.Finalize(absl::InternalError("parser failure")));
    EXPECT_TRUE(consumer.finalized);
    EXPECT_THAT(consumer.received_status,
                StatusIs(absl::StatusCode::kInternal, "parser failure"));
  }
}

TEST(RecordConsumerRefTest, VoidFinalizeWithResultOutcome) {
  struct VoidFinalizeWithResultConsumer {
    bool finalized = false;
    std::optional<ParseStatus> outcome;
    absl::Status received_status = absl::OkStatus();

    absl::StatusOr<StepControl> operator()(Record&) {
      return StepControl::kContinue;
    }

    void Finalize(const absl::StatusOr<ParseStatus>& result) {
      finalized = true;
      if (result.ok()) {
        outcome = *result;
      } else {
        received_status = result.status();
      }
    }
  };

  {
    VoidFinalizeWithResultConsumer consumer;
    RecordConsumerRef ref(consumer);
    EXPECT_OK(ref.Finalize(ParseStatus::kComplete));
    EXPECT_TRUE(consumer.finalized);
    EXPECT_THAT(consumer.outcome, Optional(Eq(ParseStatus::kComplete)));
  }

  {
    VoidFinalizeWithResultConsumer consumer;
    RecordConsumerRef ref(consumer);
    EXPECT_OK(ref.Finalize(absl::InternalError("parser failure")));
    EXPECT_TRUE(consumer.finalized);
    EXPECT_THAT(consumer.received_status,
                StatusIs(absl::StatusCode::kInternal, "parser failure"));
  }
}

TEST(RecordConsumerRefTest, FinalizeParameterlessOnlyCalledOnSuccess) {
  struct ParameterlessFinalizeConsumer {
    int finalize_calls = 0;

    absl::StatusOr<StepControl> operator()(Record&) {
      return StepControl::kContinue;
    }

    absl::Status Finalize() {
      finalize_calls++;
      return absl::OkStatus();
    }
  };

  {
    ParameterlessFinalizeConsumer consumer;
    RecordConsumerRef ref(consumer);
    EXPECT_OK(ref.Finalize(ParseStatus::kComplete));
    EXPECT_EQ(consumer.finalize_calls, 1);
  }

  {
    ParameterlessFinalizeConsumer consumer;
    RecordConsumerRef ref(consumer);
    EXPECT_OK(ref.Finalize(absl::InternalError("parser failure")));
    EXPECT_EQ(consumer.finalize_calls, 0);
  }
}

TEST(RecordConsumerRefTest, VoidFinalizeParameterlessOnlyCalledOnSuccess) {
  struct VoidParameterlessFinalizeConsumer {
    int finalize_calls = 0;

    absl::StatusOr<StepControl> operator()(Record&) {
      return StepControl::kContinue;
    }

    void Finalize() { finalize_calls++; }
  };

  {
    VoidParameterlessFinalizeConsumer consumer;
    RecordConsumerRef ref(consumer);
    EXPECT_OK(ref.Finalize(ParseStatus::kComplete));
    EXPECT_EQ(consumer.finalize_calls, 1);
  }

  {
    VoidParameterlessFinalizeConsumer consumer;
    RecordConsumerRef ref(consumer);
    EXPECT_OK(ref.Finalize(absl::InternalError("parser failure")));
    EXPECT_EQ(consumer.finalize_calls, 0);
  }
}

TEST(RecordConsumerRefTest, SupportsImplicitConversion) {
  auto invoke_ref = [](RecordConsumerRef ref, Record& record) {
    return ref(record);
  };

  auto lambda = [](Record&) -> absl::StatusOr<StepControl> {
    return StepControl::kContinue;
  };
  Record record;
  EXPECT_THAT(invoke_ref(lambda, record),
              IsOkAndHolds(Eq(StepControl::kContinue)));
}

TEST(RecordConsumerRefTest, PrioritizesFinalizeWithResultOverParameterless) {
  struct BothFinalizeConsumer {
    bool result_finalize_called = false;
    bool parameterless_finalize_called = false;

    absl::StatusOr<StepControl> operator()(Record&) {
      return StepControl::kContinue;
    }

    absl::Status Finalize(const absl::StatusOr<ParseStatus>&) {
      result_finalize_called = true;
      return absl::OkStatus();
    }

    absl::Status Finalize() {
      parameterless_finalize_called = true;
      return absl::OkStatus();
    }
  };

  BothFinalizeConsumer consumer;
  RecordConsumerRef ref(consumer);
  EXPECT_OK(ref.Finalize(ParseStatus::kComplete));
  EXPECT_TRUE(consumer.result_finalize_called);
  EXPECT_FALSE(consumer.parameterless_finalize_called);
}

TEST(RecordConsumerRefTest, NoopFinalizeWhenNotImplemented) {
  auto simple_lambda = [](Record&) -> absl::StatusOr<StepControl> {
    return StepControl::kContinue;
  };

  RecordConsumerRef ref(simple_lambda);
  EXPECT_OK(ref.Finalize(ParseStatus::kComplete));
  EXPECT_OK(ref.Finalize(absl::InternalError("error")));
}

TEST(RecordConsumerRefTest, CopyAndMoveSemantics) {
  int count = 0;
  auto lambda_consumer = [&](Record&) -> absl::StatusOr<StepControl> {
    count++;
    return StepControl::kContinue;
  };

  RecordConsumerRef ref1(lambda_consumer);
  RecordConsumerRef ref2 = ref1;             // Copy constructor
  RecordConsumerRef ref3 = std::move(ref1);  // Move constructor
  RecordConsumerRef ref4(lambda_consumer);
  ref4 = ref2;             // Copy assignment
  ref4 = std::move(ref3);  // Move assignment

  Record record;
  EXPECT_OK(ref4(record));
  EXPECT_EQ(count, 1);
}

}  // namespace
}  // namespace xprof::events_db
