/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/grpc_metrics/transfer_key_parser.h"

#include <optional>
#include <string>

#include <gtest/gtest.h>

namespace xprof::grpc_metrics {
namespace {

// A key with only the mandatory parts has a label and two device ids.
TEST(TransferKeyParserTest, ParsesKeyWithoutOptionalParts) {
  const std::optional<TransferKeyFields> fields =
      ParseTransferKey("all-reduce.1|s100001->200003d|");

  ASSERT_TRUE(fields.has_value());
  EXPECT_EQ(fields->label, "all-reduce.1");
  EXPECT_EQ(fields->source_megascale_id, 100001);
  EXPECT_EQ(fields->destination_megascale_id, 200003);
  EXPECT_FALSE(fields->chunk.has_value());
  EXPECT_FALSE(fields->launch_id.has_value());
  EXPECT_FALSE(fields->loop_index.has_value());
}

// The chunk, launch id and loop index are read when present.
TEST(TransferKeyParserTest, ParsesAllOptionalParts) {
  const std::optional<TransferKeyFields> fields =
      ParseTransferKey("test_collective|s100001->100002d|$c5$l3^i4");

  ASSERT_TRUE(fields.has_value());
  EXPECT_EQ(fields->label, "test_collective");
  EXPECT_EQ(fields->source_megascale_id, 100001);
  EXPECT_EQ(fields->destination_megascale_id, 100002);
  EXPECT_EQ(fields->chunk, 5);
  EXPECT_EQ(fields->launch_id, 3);
  EXPECT_EQ(fields->loop_index, 4);
}

// Each optional part may appear alone; only the launch id may be negative.
TEST(TransferKeyParserTest, ParsesEachOptionalPartAlone) {
  const std::optional<TransferKeyFields> chunk_only =
      ParseTransferKey("x|s100001->100002d|$c0");
  ASSERT_TRUE(chunk_only.has_value());
  EXPECT_EQ(chunk_only->chunk, 0);
  EXPECT_FALSE(chunk_only->launch_id.has_value());

  const std::optional<TransferKeyFields> launch_only =
      ParseTransferKey("x|s100001->100002d|$l-7");
  ASSERT_TRUE(launch_only.has_value());
  EXPECT_EQ(launch_only->launch_id, -7);
  EXPECT_FALSE(launch_only->loop_index.has_value());

  const std::optional<TransferKeyFields> loop_only =
      ParseTransferKey("x|s100001->100002d|^i9");
  ASSERT_TRUE(loop_only.has_value());
  EXPECT_EQ(loop_only->loop_index, 9);
  EXPECT_FALSE(loop_only->chunk.has_value());
}

// Numbers may have leading zeros.
TEST(TransferKeyParserTest, AcceptsLeadingZeros) {
  const std::optional<TransferKeyFields> fields =
      ParseTransferKey("x|s000100001->0100002d|$c007$l-01^i00");

  ASSERT_TRUE(fields.has_value());
  EXPECT_EQ(fields->source_megascale_id, 100001);
  EXPECT_EQ(fields->destination_megascale_id, 100002);
  EXPECT_EQ(fields->chunk, 7);
  EXPECT_EQ(fields->launch_id, -1);
  EXPECT_EQ(fields->loop_index, 0);
}

// Labels may hold UTF-8 text, but not other bytes.
TEST(TransferKeyParserTest, ChecksLabelEncoding) {
  EXPECT_TRUE(ParseTransferKey("caf\xC3\xA9|s100001->100002d|").has_value());
  EXPECT_FALSE(ParseTransferKey("bad\xFF|s100001->100002d|").has_value());
  EXPECT_FALSE(ParseTransferKey("bad\xC3|s100001->100002d|").has_value());
}

// Malformed keys are rejected.
TEST(TransferKeyParserTest, RejectsMalformedKeys) {
  for (const char* const key : {
           "",
           "no_endpoints",
           "|s100001->100002d|",
           "a~b|s100001->100002d|",
           "a->b|s100001->100002d|",
           "ad|b|s100001->100002d|",
           "a$cb|s100001->100002d|",
           "a$lb|s100001->100002d|",
           "a^ib|s100001->100002d|",
           "a|s100001->100002d|b|s100003->100004d|",
           "a|s->100002d|",
           "a|s100001->d|",
           "a|s100001->100002",
           "a|s100001->100002d",
           "a|s100001-100002d|",
           "a|s100001->100002d|x",
           "a|s100001->100002d|$l3$c5",
           "a|s100001->100002d|$c1$c2",
           "a|s100001->100002d|$c",
           "a|s100001->100002d|$c-1",
           "a|s100001->100002d|^i-1",
           "a|s100001->100002d|$l-",
           "a|s100001->100002d|$l--5",
           "a|s+100001->100002d|",
       }) {
    EXPECT_FALSE(ParseTransferKey(key).has_value()) << key;
  }
}

// Numbers that do not fit their type are rejected.
TEST(TransferKeyParserTest, RejectsOutOfRangeNumbers) {
  EXPECT_FALSE(
      ParseTransferKey("a|s99999999999999999999->100002d|").has_value());
  EXPECT_FALSE(ParseTransferKey("a|s100001->100002d|$c2147483648").has_value());
  EXPECT_FALSE(
      ParseTransferKey("a|s100001->100002d|$l-2147483649").has_value());
  EXPECT_FALSE(ParseTransferKey("a|s100001->100002d|^i2147483648").has_value());
  EXPECT_TRUE(ParseTransferKey("a|s100001->100002d|$l-2147483648").has_value());
}

// Device ids follow the MegaScale rules: an id is at least 100000, and the
// local slice (slice id 2147483647) only has device 0.
TEST(TransferKeyParserTest, ValidatesDeviceIds) {
  EXPECT_FALSE(ParseTransferKey("a|s99999->100002d|").has_value());
  EXPECT_FALSE(ParseTransferKey("a|s100001->99999d|").has_value());

  const std::optional<TransferKeyFields> local_slice =
      ParseTransferKey("a|s214748364800000->100002d|");
  ASSERT_TRUE(local_slice.has_value());
  EXPECT_EQ(local_slice->source_megascale_id, 214748364800000);
  EXPECT_FALSE(ParseTransferKey("a|s214748364800001->100002d|").has_value());
}

// Like the MegaScale runtime, a slice id that does not fit in 32 bits is
// truncated to 32 bits: negative results are rejected and other results give
// a different (rebuilt) device id.
TEST(TransferKeyParserTest, TruncatesLargeSliceIds) {
  // Slice id 2^31 truncates to a negative value.
  EXPECT_FALSE(ParseTransferKey("a|s214748364900000->100002d|").has_value());

  // Slice id 2^32 + 1 truncates to 1, so the device id becomes 200005.
  const std::optional<TransferKeyFields> fields =
      ParseTransferKey("a|s429496729800005->100002d|");
  ASSERT_TRUE(fields.has_value());
  EXPECT_EQ(fields->source_megascale_id, 200005);
}

}  // namespace
}  // namespace xprof::grpc_metrics
