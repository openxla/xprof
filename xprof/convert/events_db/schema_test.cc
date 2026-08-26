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

#include "xprof/convert/events_db/schema.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <type_traits>
#include <vector>

#include "<gtest/gtest.h>"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace xprof::events_db {
namespace {

TEST(FieldIndexTest, DefaultConstructedIsInvalid) {
  FieldIndex index;
  EXPECT_FALSE(index.is_valid());
}

TEST(FieldIndexTest, ComparisonAndOrdering) {
  Schema schema;
  FieldIndex index1 = schema.RegisterFieldName("field1");
  FieldIndex index2 = schema.RegisterFieldName("field2");
  FieldIndex invalid_index;

  EXPECT_TRUE(index1.is_valid());
  EXPECT_TRUE(index2.is_valid());
  EXPECT_FALSE(invalid_index.is_valid());

  // Equality and inequality
  EXPECT_EQ(index1, index1);
  EXPECT_EQ(index2, index2);
  EXPECT_NE(index1, index2);
  EXPECT_NE(index1, invalid_index);

  // Relational ordering (index1 registered before index2)
  EXPECT_LT(index1, index2);
  EXPECT_LE(index1, index2);
  EXPECT_GT(index2, index1);
  EXPECT_GE(index2, index1);
}

TEST(FieldIndexTest, HashingIntegration) {
  Schema schema;
  FieldIndex index1 = schema.RegisterFieldName("field1");
  FieldIndex index2 = schema.RegisterFieldName("field2");
  FieldIndex invalid_index;

  absl::flat_hash_set<FieldIndex> set;
  set.insert(index1);
  set.insert(index2);

  EXPECT_TRUE(set.contains(index1));
  EXPECT_TRUE(set.contains(index2));
  EXPECT_FALSE(set.contains(invalid_index));
}

TEST(TypedFieldIndexTest, ValueTypeTrait) {
  static_assert(
      std::is_same_v<TypedFieldIndex<std::string>::ValueType, std::string>);
  static_assert(std::is_same_v<TypedFieldIndex<uint64_t>::ValueType, uint64_t>);
}

TEST(TypedFieldIndexTest, DefaultConstructedIsInvalid) {
  TypedFieldIndex<int64_t> index;
  EXPECT_FALSE(index.untyped().is_valid());
}

TEST(TypedFieldIndexTest, ExplicitConstructionAndUntypedAccess) {
  Schema schema;
  TypedFieldIndex<std::string> name_idx(schema.RegisterFieldName("name"));
  TypedFieldIndex<uint64_t> duration_idx(schema.RegisterFieldName("duration"));

  EXPECT_TRUE(name_idx.untyped().is_valid());
  EXPECT_TRUE(duration_idx.untyped().is_valid());

  // Resolving via untyped()
  EXPECT_EQ(schema.GetFieldName(name_idx.untyped()).value_or(""), "name");
  EXPECT_EQ(schema.GetFieldName(duration_idx.untyped()).value_or(""),
            "duration");
  EXPECT_EQ(schema.LookupFieldIndex("name"), name_idx.untyped());
}

TEST(TypedFieldIndexTest, ComparisonAndOrdering) {
  Schema schema;
  TypedFieldIndex<std::string> index1(schema.RegisterFieldName("field1"));
  TypedFieldIndex<std::string> index2(schema.RegisterFieldName("field2"));
  TypedFieldIndex<std::string> invalid_index;

  EXPECT_TRUE(index1.untyped().is_valid());
  EXPECT_TRUE(index2.untyped().is_valid());
  EXPECT_FALSE(invalid_index.untyped().is_valid());

  EXPECT_EQ(index1, index1);
  EXPECT_EQ(index2, index2);
  EXPECT_NE(index1, index2);
  EXPECT_NE(index1, invalid_index);

  EXPECT_LT(index1, index2);
  EXPECT_LE(index1, index2);
  EXPECT_GT(index2, index1);
  EXPECT_GE(index2, index1);

  // Comparison against untyped FieldIndex via .untyped()
  FieldIndex raw_index1 = index1.untyped();
  EXPECT_EQ(index1.untyped(), raw_index1);
}

TEST(SchemaTest, RegisterAndResolveFields) {
  Schema schema;
  EXPECT_EQ(schema.max_field_count(), std::numeric_limits<uint32_t>::max());
  EXPECT_EQ(schema.size(), 0);

  FieldIndex name_index = schema.RegisterFieldName("name");
  EXPECT_EQ(schema.size(), 1);
  FieldIndex duration_index = schema.RegisterFieldName("duration_ns");
  EXPECT_EQ(schema.size(), 2);

  EXPECT_TRUE(name_index.is_valid());
  EXPECT_TRUE(duration_index.is_valid());
  EXPECT_NE(name_index, duration_index);

  // Duplicate registration returns identical index
  FieldIndex name_index_dup = schema.RegisterFieldName("name");
  EXPECT_EQ(name_index, name_index_dup);
  EXPECT_EQ(schema.size(), 2);

  // Resolve index to name
  std::optional<absl::string_view> name_resolved =
      schema.GetFieldName(name_index);
  ASSERT_TRUE(name_resolved.has_value());
  EXPECT_EQ(*name_resolved, "name");

  std::optional<absl::string_view> duration_resolved =
      schema.GetFieldName(duration_index);
  ASSERT_TRUE(duration_resolved.has_value());
  EXPECT_EQ(*duration_resolved, "duration_ns");

  // Resolving default-constructed invalid index returns std::nullopt
  EXPECT_FALSE(schema.GetFieldName(FieldIndex()).has_value());

  // Lookup existing field
  std::optional<FieldIndex> lookup_name = schema.LookupFieldIndex("name");
  ASSERT_TRUE(lookup_name.has_value());
  EXPECT_EQ(*lookup_name, name_index);

  // Lookup non-existing field
  std::optional<FieldIndex> lookup_missing = schema.LookupFieldIndex("missing");
  EXPECT_FALSE(lookup_missing.has_value());
}

TEST(SchemaTest, TypedRegistration) {
  Schema schema;

  // Register using templated return helper RegisterFieldName<T>
  TypedFieldIndex<std::string> name_idx =
      schema.RegisterFieldName<std::string>("name");
  EXPECT_TRUE(name_idx.untyped().is_valid());
  EXPECT_EQ(schema.GetFieldName(name_idx.untyped()).value_or(""), "name");

  // Register using output-parameter helper Register(field, name)
  TypedFieldIndex<uint64_t> duration_idx;
  schema.Register(duration_idx, "duration");
  EXPECT_TRUE(duration_idx.untyped().is_valid());
  EXPECT_EQ(schema.GetFieldName(duration_idx.untyped()).value_or(""),
            "duration");

  // Registering duplicate name returns the identical underlying index
  TypedFieldIndex<std::string> name_idx_dup;
  schema.Register(name_idx_dup, "name");
  EXPECT_EQ(name_idx, name_idx_dup);
  EXPECT_EQ(schema.size(), 2);
}

TEST(SchemaTest, ConcurrentRegistration) {
  Schema schema;
  constexpr int kNumThreads = 8;
  constexpr int kFieldsPerThread = 50;

  std::vector<std::vector<FieldIndex>> thread_indices(kNumThreads);

  {
    std::vector<std::jthread> threads;
    threads.reserve(kNumThreads);

    for (int t = 0; t < kNumThreads; ++t) {
      threads.emplace_back([&schema, &thread_indices, t]() {
        thread_indices[t].reserve(2 * kFieldsPerThread);
        for (int i = 0; i < kFieldsPerThread; ++i) {
          // Some shared fields across threads
          std::string shared_field = absl::StrCat("shared_field_", i);
          FieldIndex index1 = schema.RegisterFieldName(shared_field);
          EXPECT_TRUE(index1.is_valid());
          thread_indices[t].push_back(index1);

          // Some thread-unique fields
          std::string unique_field = absl::StrCat("field_", t, "_", i);
          FieldIndex index2 = schema.RegisterFieldName(unique_field);
          EXPECT_TRUE(index2.is_valid());
          thread_indices[t].push_back(index2);

          std::optional<absl::string_view> name_resolved =
              schema.GetFieldName(index2);
          ASSERT_TRUE(name_resolved.has_value());
          EXPECT_EQ(*name_resolved, unique_field);
        }
      });
    }
  }  // Automatically joins all threads on scope exit.

  // Verify all shared fields can be looked up
  for (int i = 0; i < kFieldsPerThread; ++i) {
    std::string shared_field = absl::StrCat("shared_field_", i);
    std::optional<FieldIndex> lookup = schema.LookupFieldIndex(shared_field);
    ASSERT_TRUE(lookup.has_value());
    EXPECT_EQ(schema.GetFieldName(*lookup).value_or(""), shared_field);
  }

  // Verify all thread-unique fields can be looked up
  for (int t = 0; t < kNumThreads; ++t) {
    for (int i = 0; i < kFieldsPerThread; ++i) {
      std::string unique_field = absl::StrCat("field_", t, "_", i);
      std::optional<FieldIndex> lookup = schema.LookupFieldIndex(unique_field);
      ASSERT_TRUE(lookup.has_value());
      EXPECT_EQ(schema.GetFieldName(*lookup).value_or(""), unique_field);
    }
  }

  // Verify that the total number of unique `FieldIndex` values generated
  // matches the expected count of distinct registered fields:
  // (`kFieldsPerThread shared fields + kNumThreads * kFieldsPerThread` unique
  // fields).
  absl::flat_hash_set<FieldIndex> unique_indices;
  for (const std::vector<FieldIndex>& indices : thread_indices) {
    for (FieldIndex index : indices) {
      unique_indices.insert(index);
    }
  }
  EXPECT_EQ(unique_indices.size(), (kNumThreads + 1) * kFieldsPerThread);
  EXPECT_EQ(schema.size(), (kNumThreads + 1) * kFieldsPerThread);
}

TEST(SchemaTest, MaxFieldsCapacityExceeded) {
  constexpr uint32_t kMaxFieldCount = 2;
  Schema schema(kMaxFieldCount);
  EXPECT_EQ(schema.max_field_count(), kMaxFieldCount);
  EXPECT_EQ(schema.size(), 0);

  FieldIndex index1 = schema.RegisterFieldName("field1");
  FieldIndex index2 = schema.RegisterFieldName("field2");

  EXPECT_TRUE(index1.is_valid());
  EXPECT_TRUE(index2.is_valid());
  EXPECT_NE(index1, index2);
  EXPECT_EQ(schema.size(), 2);

  // Re-registering existing fields returns existing valid tokens even when at
  // capacity.
  FieldIndex index1_dup = schema.RegisterFieldName("field1");
  EXPECT_TRUE(index1_dup.is_valid());
  EXPECT_EQ(index1_dup, index1);
  EXPECT_EQ(schema.size(), 2);

  // Registering a new field beyond max capacity returns an invalid FieldIndex.
  FieldIndex index3 = schema.RegisterFieldName("field3");
  EXPECT_FALSE(index3.is_valid());
  EXPECT_EQ(index3, FieldIndex());
  EXPECT_EQ(schema.size(), 2);

  // Resolving or looking up the unregistered field fails.
  EXPECT_FALSE(schema.GetFieldName(index3).has_value());
  EXPECT_FALSE(schema.LookupFieldIndex("field3").has_value());

  // Existing registered fields remain unaffected.
  EXPECT_EQ(schema.GetFieldName(index1).value_or(""), "field1");
  EXPECT_EQ(schema.GetFieldName(index2).value_or(""), "field2");
  EXPECT_EQ(schema.LookupFieldIndex("field1"), index1);
  EXPECT_EQ(schema.LookupFieldIndex("field2"), index2);
}

TEST(SchemaTest, ZeroCapacitySchema) {
  Schema schema(/*max_field_count=*/0);
  EXPECT_EQ(schema.max_field_count(), 0);
  EXPECT_EQ(schema.size(), 0);

  FieldIndex index = schema.RegisterFieldName("field");
  EXPECT_FALSE(index.is_valid());
  EXPECT_EQ(schema.size(), 0);
}

}  // namespace
}  // namespace xprof::events_db
