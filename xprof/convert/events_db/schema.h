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

#ifndef THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_SCHEMA_H_
#define THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_SCHEMA_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace xprof::events_db {

class Schema;

// Opaque token representing a field. Wraps a `uint32_t` for O(1) hashing and
// lookup. Pass by value.
//
// Note: `FieldIndex` tokens are process-local and specific to a given `Schema`
// instance. Because they are assigned sequentially based on registration order,
// tokens are NOT stable across different process executions, runs, or distinct
// `Schema` instances. Do NOT serialize or persist `FieldIndex` tokens directly;
// persist field names instead.
class FieldIndex {
 public:
  FieldIndex() : id_(kInvalidId) {}

  // C++20 Defaulted Comparison Operators (automatically generates `==`, `!=`,
  // `<`, `<=`, `>`, and `>=`).
  auto operator<=>(const FieldIndex& other) const = default;

  bool is_valid() const { return id_ != kInvalidId; }

  // `AbslHashValue` is a friend function required to integrate with Abseil's
  // high-performance hashing framework (`absl::Hash`). It allows `FieldIndex`
  // to be used as a key in Abseil hash containers (like `absl::flat_hash_map`)
  // directly with minimal overhead, avoiding character-by-character string
  // hashing.
  template <typename H>
  friend H AbslHashValue(H hash_value, const FieldIndex& index) {
    return H::combine(std::move(hash_value), index.id_);
  }

 private:
  friend class Schema;
  explicit FieldIndex(uint32_t id) : id_(id) {}

  static constexpr uint32_t kInvalidId = 0xFFFFFFFF;
  uint32_t id_;
};

// A lightweight wrapper around FieldIndex that associates a compile-time C++
// type with a schema field for self-documentation and type safety. Pass by
// value.
//
// Note: Similar to `FieldIndex`, `TypedFieldIndex` tokens are process-local and
// specific to a given `Schema`. Do NOT serialize or persist them directly;
// persist field names instead.
template <typename T>
class TypedFieldIndex {
 public:
  using ValueType = T;

  TypedFieldIndex() = default;

  explicit TypedFieldIndex(FieldIndex index) : index_(index) {}

  auto operator<=>(const TypedFieldIndex&) const = default;

  FieldIndex untyped() const { return index_; }

 private:
  FieldIndex index_;
};

// Schema manages the mapping between string names and indices. Thread-safe.
class Schema {
 public:
  explicit Schema(
      uint32_t max_field_count = std::numeric_limits<uint32_t>::max())
      : max_field_count_(max_field_count) {}

  Schema(const Schema&) = delete;
  Schema& operator=(const Schema&) = delete;

  // Registers a name and returns its unique index. If already registered,
  // returns the existing index. If `max_field_count()` names have already
  // been registered and the input name has not been registered before, an
  // invalid index is returned (where `is_valid()` is false).
  FieldIndex RegisterFieldName(absl::string_view name);

  // Registers a field name and returns a strongly-typed index token of type
  // `T`. Behaves identically to `RegisterFieldName(name)` regarding
  // deduplication and schema capacity limits.
  template <typename T>
  TypedFieldIndex<T> RegisterFieldName(absl::string_view name) {
    return TypedFieldIndex<T>(RegisterFieldName(name));
  }

  // Registers a field name and assigns the resulting strongly-typed token
  // directly to `field`, allowing the compiler to deduce `T` without repeating
  // the type at the call site.
  template <typename T>
  void Register(TypedFieldIndex<T>& field, absl::string_view name) {
    field = TypedFieldIndex<T>(RegisterFieldName(name));
  }

  // Resolves an index back to its name.
  std::optional<absl::string_view> GetFieldName(FieldIndex field) const;

  // Looks up an existing field without registering it.
  std::optional<FieldIndex> LookupFieldIndex(absl::string_view name) const;

  // Returns the maximum number of names that can be registered.
  uint32_t max_field_count() const { return max_field_count_; }

  // Returns the number of names currently registered.
  uint32_t size() const;

 private:
  const uint32_t max_field_count_;
  mutable absl::Mutex mutex_;
  // `absl::node_hash_map` guarantees key pointer stability upon rehashing,
  // allowing `name_by_id_` to hold direct views to the keys in memory.
  absl::node_hash_map<std::string, uint32_t> id_by_name_
      ABSL_GUARDED_BY(mutex_);
  std::vector<absl::string_view> name_by_id_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xprof::events_db

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_SCHEMA_H_
