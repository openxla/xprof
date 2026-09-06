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
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace xprof::events_db {
namespace {

using ::testing::ElementsAre;

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

TEST(RecordTest, DefaultConstructedIsEmpty) {
  Schema schema;
  FieldIndex field_index = schema.RegisterFieldName("field1");

  Record record;
  EXPECT_EQ(record.size(), 0);
  EXPECT_FALSE(record.HasField(field_index));

  // Accessing missing field via `Get()` or `const operator[]` returns monostate
  // and does not modify record size.
  const FieldValue& val = record.Get(field_index);
  EXPECT_TRUE(std::holds_alternative<std::monostate>(val));
  EXPECT_EQ(record.size(), 0);
}

TEST(RecordTest, SetAndGetScalarFieldValues) {
  Schema schema;
  FieldIndex is_kernel_index = schema.RegisterFieldName("is_kernel");
  FieldIndex stream_id_index = schema.RegisterFieldName("stream_id");
  FieldIndex correlation_id_index = schema.RegisterFieldName("correlation_id");
  FieldIndex offset_ns_index = schema.RegisterFieldName("offset_ns");
  FieldIndex start_ns_index = schema.RegisterFieldName("start_ns");
  FieldIndex utilization_index = schema.RegisterFieldName("utilization");
  FieldIndex kernel_name_index = schema.RegisterFieldName("kernel_name");

  Record record;
  record[is_kernel_index] = true;
  record[stream_id_index] = int32_t{-7};
  record[correlation_id_index] = uint32_t{42};
  record[offset_ns_index] = int64_t{-100000};
  record[start_ns_index] = uint64_t{1234567890123ULL};
  record[utilization_index] = 0.875;
  record[kernel_name_index] = std::string("matmul_kernel");

  EXPECT_EQ(record.size(), 7);
  EXPECT_TRUE(record.HasField(is_kernel_index));
  EXPECT_TRUE(record.HasField(stream_id_index));
  EXPECT_TRUE(record.HasField(correlation_id_index));
  EXPECT_TRUE(record.HasField(offset_ns_index));
  EXPECT_TRUE(record.HasField(start_ns_index));
  EXPECT_TRUE(record.HasField(utilization_index));
  EXPECT_TRUE(record.HasField(kernel_name_index));

  const Record& const_record = record;

  EXPECT_EQ(std::get<bool>(const_record[is_kernel_index]), true);
  EXPECT_EQ(std::get<int32_t>(const_record[stream_id_index]), -7);
  EXPECT_EQ(std::get<uint32_t>(const_record[correlation_id_index]), 42);
  EXPECT_EQ(std::get<int64_t>(const_record[offset_ns_index]), -100000);
  EXPECT_EQ(std::get<uint64_t>(const_record[start_ns_index]), 1234567890123ULL);
  EXPECT_DOUBLE_EQ(std::get<double>(const_record[utilization_index]), 0.875);
  EXPECT_EQ(std::get<std::string>(const_record[kernel_name_index]),
            "matmul_kernel");
}

TEST(RecordTest, SetAndGetRepeatedFieldValues) {
  Schema schema;
  FieldIndex vec_i32_index = schema.RegisterFieldName("vec_i32");
  FieldIndex vec_u32_index = schema.RegisterFieldName("vec_u32");
  FieldIndex vec_i64_index = schema.RegisterFieldName("vec_i64");
  FieldIndex vec_u64_index = schema.RegisterFieldName("vec_u64");
  FieldIndex vec_double_index = schema.RegisterFieldName("vec_double");
  FieldIndex input_tensors_index = schema.RegisterFieldName("input_tensors");

  Record record;
  record[vec_i32_index] = std::vector<int32_t>{-1, 0, 1};
  record[vec_u32_index] = std::vector<uint32_t>{10, 20};
  record[vec_i64_index] = std::vector<int64_t>{-100, 200};
  record[vec_u64_index] = std::vector<uint64_t>{1000ULL, 2000ULL};
  record[vec_double_index] = std::vector<double>{0.1, 0.2, 0.3};
  record[input_tensors_index] =
      std::vector<std::string>{"tensor_a", "tensor_b"};

  EXPECT_EQ(record.size(), 6);

  const Record& const_record = record;

  EXPECT_THAT(std::get<std::vector<int32_t>>(const_record[vec_i32_index]),
              ElementsAre(-1, 0, 1));
  EXPECT_THAT(std::get<std::vector<uint32_t>>(const_record[vec_u32_index]),
              ElementsAre(10, 20));
  EXPECT_THAT(std::get<std::vector<int64_t>>(const_record[vec_i64_index]),
              ElementsAre(-100, 200));
  EXPECT_THAT(std::get<std::vector<uint64_t>>(const_record[vec_u64_index]),
              ElementsAre(1000ULL, 2000ULL));
  EXPECT_THAT(std::get<std::vector<double>>(const_record[vec_double_index]),
              ElementsAre(0.1, 0.2, 0.3));
  EXPECT_THAT(
      std::get<std::vector<std::string>>(const_record[input_tensors_index]),
      ElementsAre("tensor_a", "tensor_b"));
}

TEST(RecordTest, MutableOperatorIndexUpdatesValueInPlace) {
  Schema schema;
  FieldIndex count_index = schema.RegisterFieldName("count");

  Record record;
  record[count_index] = int64_t{10};
  EXPECT_EQ(std::get<int64_t>(record.Get(count_index)), 10);

  // Modify in-place via mutable reference
  std::get<int64_t>(record[count_index]) = 25;
  EXPECT_EQ(std::get<int64_t>(record.Get(count_index)), 25);
}

TEST(RecordTest, TypedFieldAccess) {
  Schema schema;
  TypedFieldIndex<std::string> name_index =
      schema.RegisterFieldName<std::string>("name");
  TypedFieldIndex<uint64_t> count_index =
      schema.RegisterFieldName<uint64_t>("count");
  TypedFieldIndex<std::vector<int32_t>> values_index =
      schema.RegisterFieldName<std::vector<int32_t>>("values");
  TypedFieldIndex<double> missing_index =
      schema.RegisterFieldName<double>("missing");

  Record record;

  // Accessing via mutable typed `operator[]` inserts a default-constructed
  // `T` and returns a mutable reference `T&`.
  EXPECT_EQ(record[name_index], "");
  EXPECT_EQ(record[count_index], 0);

  // Directly assign or mutate typed values in place without pointer
  // dereference.
  record[name_index] = "test_event";
  record[count_index] = 42;
  record[values_index].push_back(100);
  record[values_index].push_back(200);

  const Record& const_record = record;

  // `Get()` and `const operator[]` return `const T&`.
  EXPECT_EQ(const_record.Get(name_index), "test_event");
  EXPECT_EQ(const_record[name_index], "test_event");
  EXPECT_EQ(const_record[count_index], 42);
  EXPECT_THAT(const_record.Get(values_index), ElementsAre(100, 200));

  // `TryGet()` returns `const T*` / `T*`, or `nullptr` if unset or mismatched.
  ASSERT_NE(const_record.TryGet(name_index), nullptr);
  EXPECT_EQ(*const_record.TryGet(name_index), "test_event");
  EXPECT_EQ(const_record.TryGet(missing_index), nullptr);
  EXPECT_EQ(record.TryGet(missing_index), nullptr);
  EXPECT_FALSE(const_record.HasField(missing_index));

  // Mutating via mutable `TryGet()` pointer.
  ASSERT_NE(record.TryGet(name_index), nullptr);
  *record.TryGet(name_index) = "updated_event";
  EXPECT_EQ(const_record[name_index], "updated_event");

  // `HasField` returns true when field exists and holds matching type `T`.
  EXPECT_TRUE(const_record.HasField(name_index));
  EXPECT_TRUE(const_record.HasField(count_index));
  EXPECT_TRUE(const_record.HasField(values_index));

  // Mismatched types on an existing field return false for `HasField` and
  // `nullptr` for `TryGet()`.
  TypedFieldIndex<uint64_t> name_as_int(name_index.untyped());
  EXPECT_FALSE(const_record.HasField(name_as_int));
  EXPECT_FALSE(record.HasField(name_as_int));
  EXPECT_EQ(const_record.TryGet(name_as_int), nullptr);
  EXPECT_EQ(record.TryGet(name_as_int), nullptr);
}

TEST(RecordTest, ClearRemovesAllFields) {
  Schema schema;
  FieldIndex field1_index = schema.RegisterFieldName("field1");
  FieldIndex field2_index = schema.RegisterFieldName("field2");

  Record record;
  record[field1_index] = int64_t{1};
  record[field2_index] = int64_t{2};
  EXPECT_EQ(record.size(), 2);
  EXPECT_TRUE(record.HasField(field1_index));
  EXPECT_TRUE(record.HasField(field2_index));

  record.clear();

  EXPECT_EQ(record.size(), 0);
  EXPECT_FALSE(record.HasField(field1_index));
  EXPECT_FALSE(record.HasField(field2_index));
}

TEST(RecordTest, EqualityComparison) {
  Schema schema;
  FieldIndex field1_index = schema.RegisterFieldName("field1");
  FieldIndex field2_index = schema.RegisterFieldName("field2");
  FieldIndex field3_index = schema.RegisterFieldName("field3");

  Record r1;
  Record r2;
  EXPECT_EQ(r1, r2);

  // Different sizes due to trailing monostates still compare equal.
  r2[field3_index];
  EXPECT_EQ(r1.size(), 0);
  EXPECT_EQ(r2.size(), 3);
  EXPECT_EQ(r1, r2);
  EXPECT_EQ(r2, r1);

  // Setting field1 on both records makes them equal despite size difference.
  r1[field1_index] = int64_t{42};
  r2[field1_index] = int64_t{42};
  EXPECT_EQ(r1, r2);
  EXPECT_EQ(r2, r1);

  // Different variant types for the same field index compare unequal even if
  // their numeric values are equivalent.
  r2[field1_index] = uint64_t{42};
  EXPECT_NE(r1, r2);
  r2[field1_index] = int64_t{42};
  EXPECT_EQ(r1, r2);

  // Setting an additional field on only one record makes them unequal.
  r1[field2_index] = std::string("test");
  EXPECT_NE(r1, r2);
  EXPECT_NE(r2, r1);

  r2[field2_index] = std::string("different");
  EXPECT_NE(r1, r2);

  // Setting the same value makes them equal again.
  r2[field2_index] = std::string("test");
  EXPECT_EQ(r1, r2);
  EXPECT_EQ(r2, r1);

  // Setting trailing field3 to a non-monostate value on r2 makes them unequal.
  r2[field3_index] = double{3.14};
  EXPECT_NE(r1, r2);
  EXPECT_NE(r2, r1);
}

TEST(RecordTest, VectorDynamicExpansion) {
  Schema schema;
  std::vector<FieldIndex> indices;
  indices.reserve(50);
  for (int i = 0; i < 50; ++i) {
    indices.push_back(schema.RegisterFieldName(absl::StrCat("field_", i)));
  }

  Record record;
  for (int i = 0; i < 50; ++i) {
    record[indices[i]] = int64_t{i * 10};
  }

  EXPECT_EQ(record.size(), 50);
  for (int i = 0; i < 50; ++i) {
    EXPECT_TRUE(record.HasField(indices[i]));
    EXPECT_EQ(std::get<int64_t>(record[indices[i]]), i * 10);
  }
}

TEST(RecordTest, SetScalarMethods) {
  Schema schema;
  FieldIndex b_idx = schema.RegisterFieldName("bool_field");
  FieldIndex i32_idx = schema.RegisterFieldName("i32_field");
  FieldIndex u32_idx = schema.RegisterFieldName("u32_field");
  FieldIndex i64_idx = schema.RegisterFieldName("i64_field");
  FieldIndex u64_idx = schema.RegisterFieldName("u64_field");
  FieldIndex d_idx = schema.RegisterFieldName("double_field");
  FieldIndex s_idx = schema.RegisterFieldName("str_field");
  FieldIndex sv_idx = schema.RegisterFieldName("sv_field");

  Record record;
  record.SetBool(b_idx, true);
  record.SetInt32(i32_idx, -42);
  record.SetUint32(u32_idx, 100);
  record.SetInt64(i64_idx, -10000000000LL);
  record.SetUint64(u64_idx, 18446744073709551615ULL);
  record.SetDouble(d_idx, 3.14159);
  record.SetString(s_idx, std::string("owned_string"));
  record.SetString(sv_idx, absl::string_view("view_string"));

  EXPECT_EQ(std::get<bool>(record[b_idx]), true);
  EXPECT_EQ(std::get<int32_t>(record[i32_idx]), -42);
  EXPECT_EQ(std::get<uint32_t>(record[u32_idx]), 100);
  EXPECT_EQ(std::get<int64_t>(record[i64_idx]), -10000000000LL);
  EXPECT_EQ(std::get<uint64_t>(record[u64_idx]), 18446744073709551615ULL);
  EXPECT_DOUBLE_EQ(std::get<double>(record[d_idx]), 3.14159);
  EXPECT_EQ(std::get<std::string>(record[s_idx]), "owned_string");
  EXPECT_EQ(std::get<std::string>(record[sv_idx]), "view_string");
}

TEST(RecordTest, SetSequenceMethods) {
  Schema schema;
  FieldIndex vec_i32_idx = schema.RegisterFieldName("vec_i32");
  FieldIndex vec_u32_idx = schema.RegisterFieldName("vec_u32");
  FieldIndex vec_i64_idx = schema.RegisterFieldName("vec_i64");
  FieldIndex vec_u64_idx = schema.RegisterFieldName("vec_u64");
  FieldIndex vec_d_idx = schema.RegisterFieldName("vec_double");
  FieldIndex vec_s_idx = schema.RegisterFieldName("vec_str");

  Record record;
  record.SetInt32Sequence(vec_i32_idx, {-1, 0, 1});
  record.SetUint32Sequence(vec_u32_idx, {10, 20});
  record.SetInt64Sequence(vec_i64_idx, {-100, 200});
  record.SetUint64Sequence(vec_u64_idx, {1000ULL, 2000ULL});
  record.SetDoubleSequence(vec_d_idx, {0.1, 0.2});
  record.SetStringSequence(vec_s_idx, {"a", "b"});

  EXPECT_THAT(std::get<std::vector<int32_t>>(record[vec_i32_idx]),
              ElementsAre(-1, 0, 1));
  EXPECT_THAT(std::get<std::vector<uint32_t>>(record[vec_u32_idx]),
              ElementsAre(10, 20));
  EXPECT_THAT(std::get<std::vector<int64_t>>(record[vec_i64_idx]),
              ElementsAre(-100, 200));
  EXPECT_THAT(std::get<std::vector<uint64_t>>(record[vec_u64_idx]),
              ElementsAre(1000ULL, 2000ULL));
  EXPECT_THAT(std::get<std::vector<double>>(record[vec_d_idx]),
              ElementsAre(0.1, 0.2));
  EXPECT_THAT(std::get<std::vector<std::string>>(record[vec_s_idx]),
              ElementsAre("a", "b"));
}

TEST(RecordTest, GenericSetMethod) {
  Schema schema;
  FieldIndex untyped_idx = schema.RegisterFieldName("untyped");
  TypedFieldIndex<int64_t> typed_i64 =
      schema.RegisterFieldName<int64_t>("typed_i64");
  TypedFieldIndex<std::string> typed_str =
      schema.RegisterFieldName<std::string>("typed_str");
  TypedFieldIndex<double> typed_dbl =
      schema.RegisterFieldName<double>("typed_dbl");
  TypedFieldIndex<std::vector<int64_t>> typed_vec =
      schema.RegisterFieldName<std::vector<int64_t>>("typed_vec");

  Record record;

  // Untyped FieldIndex: rvalue and lvalue
  record.Set(untyped_idx, int32_t{42});
  EXPECT_EQ(std::get<int32_t>(record[untyped_idx]), 42);
  int32_t untyped_lvalue = 84;
  record.Set(untyped_idx, untyped_lvalue);
  EXPECT_EQ(std::get<int32_t>(record[untyped_idx]), 84);

  // TypedFieldIndex: rvalue
  record.Set(typed_i64, int64_t{99});
  EXPECT_EQ(record[typed_i64], 99);

  // TypedFieldIndex: lvalue variable
  int64_t i64_lvalue = 123;
  record.Set(typed_i64, i64_lvalue);
  EXPECT_EQ(record[typed_i64], 123);

  // TypedFieldIndex: implicit type conversion (int literal converted to
  // int64_t) Ensures value is stored as int64_t in variant, avoiding
  // std::bad_variant_access.
  record.Set(typed_i64, 456);
  EXPECT_EQ(record[typed_i64], 456);

  // TypedFieldIndex<std::string>: implicit conversion from string literal
  record.Set(typed_str, "literal_value");
  EXPECT_EQ(record[typed_str], "literal_value");

  // TypedFieldIndex<std::string>: lvalue and rvalue std::string
  std::string str_lvalue = "lvalue_value";
  record.Set(typed_str, str_lvalue);
  EXPECT_EQ(record[typed_str], "lvalue_value");
  record.Set(typed_str, std::string("rvalue_value"));
  EXPECT_EQ(record[typed_str], "rvalue_value");

  // TypedFieldIndex<double>: implicit conversion from int literal
  record.Set(typed_dbl, 10);
  EXPECT_DOUBLE_EQ(record[typed_dbl], 10.0);

  // TypedFieldIndex<std::vector<int64_t>>: lvalue and rvalue vector
  std::vector<int64_t> vec_lvalue = {1, 2, 3};
  record.Set(typed_vec, vec_lvalue);
  EXPECT_THAT(record[typed_vec], ElementsAre(1, 2, 3));
  record.Set(typed_vec, std::vector<int64_t>{4, 5});
  EXPECT_THAT(record[typed_vec], ElementsAre(4, 5));
}

TEST(RecordTest, OverwriteFieldWithDifferentTypes) {
  Schema schema;
  FieldIndex field = schema.RegisterFieldName("dynamic_field");

  Record record;
  // Initially int32
  record.SetInt32(field, 123);
  EXPECT_TRUE(record.HasField(field));
  EXPECT_EQ(std::get<int32_t>(record[field]), 123);

  // Overwrite with double
  record.SetDouble(field, 3.14);
  EXPECT_DOUBLE_EQ(std::get<double>(record[field]), 3.14);

  // Overwrite with string
  record.SetString(field, "hello");
  EXPECT_EQ(std::get<std::string>(record[field]), "hello");

  // Overwrite with sequence
  record.SetInt32Sequence(field, {-1, 0, 1});
  EXPECT_THAT(std::get<std::vector<int32_t>>(record[field]),
              ElementsAre(-1, 0, 1));

  // Overwrite with generic Set
  record.Set(field, uint64_t{9999999999ULL});
  EXPECT_EQ(std::get<uint64_t>(record[field]), 9999999999ULL);
}

}  // namespace
}  // namespace xprof::events_db
