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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/meta/type_traits.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace xprof::events_db {

class Schema;
class Record;

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

  bool operator==(const FieldIndex& other) const { return id_ == other.id_; }
  bool operator!=(const FieldIndex& other) const { return id_ != other.id_; }
  bool operator<(const FieldIndex& other) const { return id_ < other.id_; }
  bool operator<=(const FieldIndex& other) const { return id_ <= other.id_; }
  bool operator>(const FieldIndex& other) const { return id_ > other.id_; }
  bool operator>=(const FieldIndex& other) const { return id_ >= other.id_; }

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
  friend class Record;
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

  bool operator==(const TypedFieldIndex& other) const {
    return index_ == other.index_;
  }
  bool operator!=(const TypedFieldIndex& other) const {
    return index_ != other.index_;
  }
  bool operator<(const TypedFieldIndex& other) const {
    return index_ < other.index_;
  }
  bool operator<=(const TypedFieldIndex& other) const {
    return index_ <= other.index_;
  }
  bool operator>(const TypedFieldIndex& other) const {
    return index_ > other.index_;
  }
  bool operator>=(const TypedFieldIndex& other) const {
    return index_ >= other.index_;
  }

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

// Dynamically typed field value in a Record. `std::monostate` represents a
// null, unset, or missing value.
using FieldValue = std::variant<std::monostate, bool, int32_t, uint32_t,
                                int64_t, uint64_t, double, std::string,
                                std::vector<int32_t>, std::vector<uint32_t>,
                                std::vector<int64_t>, std::vector<uint64_t>,
                                std::vector<double>, std::vector<std::string>>;

// Represents an extensible, row-oriented record mapping field indices to
// dynamically typed values. Unset fields implicitly hold `std::monostate`.
// Not thread-safe.
class Record {
 public:
  Record() = default;

  // Two records are considered equal if all set fields have matching values.
  // Unset fields (holding `std::monostate`) do not affect equality.
  bool operator==(const Record& other) const;
  bool operator!=(const Record& other) const { return !(*this == other); }

  // Checks if the field has a set value in the record (i.e. not
  // `std::monostate`).
  bool HasField(FieldIndex field) const {
    return field.id_ < fields_.size() &&
           !std::holds_alternative<std::monostate>(fields_[field.id_]);
  }

  // Checks if the field is set in the record and holds a value of type `T`.
  template <typename T>
  bool HasField(TypedFieldIndex<T> field) const {
    VerifyNonMonostate<T>();
    const FieldIndex untyped = field.untyped();
    return untyped.id_ < fields_.size() &&
           std::holds_alternative<T>(fields_[untyped.id_]);
  }

  // Retrieves the value associated with the given field. Returns
  // `std::monostate` if the field is unset or missing.
  const FieldValue& operator[](FieldIndex field) const;

  // Retrieves a mutable reference to the value associated with the given
  // field. If the field was not previously set, it is initialized to
  // `std::monostate`.
  FieldValue& operator[](FieldIndex field);

  // Retrieves the value associated with the given field. Returns
  // `std::monostate` if the field is unset or missing.
  const FieldValue& Get(FieldIndex field) const { return operator[](field); }

  // Sets the value associated with the given field. It is OK if the field was
  // previously unset or held a different type.
  template <typename T>
  void Set(FieldIndex field, T&& value) {
    VerifyNonMonostate<T>();
    operator[](field) = std::forward<T>(value);
  }

  // Retrieves a const reference to the value associated with `field`. The
  // caller must ensure that the field is set and holds type `T` (e.g. via
  // `HasField`).
  template <typename T>
  const T& operator[](TypedFieldIndex<T> field) const {
    VerifyNonMonostate<T>();
    return std::get<T>(operator[](field.untyped()));
  }

  // Retrieves a mutable reference to the value associated with `field`. The
  // caller must ensure that the field is either unset or holds type `T` (e.g.
  // via `HasField`). If the field is unset, a default-constructed `T` is
  // emplaced and returned.
  template <typename T>
  T& operator[](TypedFieldIndex<T> field) {
    static_assert(std::is_default_constructible_v<T>);
    VerifyNonMonostate<T>();
    FieldValue& val = operator[](field.untyped());
    if (std::holds_alternative<std::monostate>(val)) {
      val.emplace<T>();
    }
    return std::get<T>(val);
  }

  // Retrieves a const reference to the value associated with `field`. The
  // caller must ensure that the field is set and holds type `T` (e.g. via
  // `HasField`).
  template <typename T>
  const T& Get(TypedFieldIndex<T> field) const {
    return operator[](field);
  }

  // Sets the value associated with the given typed field. It is OK if the field
  // was previously unset or held a different type.
  template <typename T, typename U = T>
  void Set(TypedFieldIndex<T> field, U&& value) {
    VerifyNonMonostate<T>();
    if constexpr (std::is_same_v<T, absl::remove_cvref_t<U>>)
      operator[](field.untyped()) = std::forward<U>(value);
    else
      operator[](field.untyped()) = static_cast<T>(std::forward<U>(value));
  }

  // Retrieves a const pointer to the value associated with `field`. Returns
  // `nullptr` if the field is unset or holds a different type than `T`.
  template <typename T>
  const T* TryGet(TypedFieldIndex<T> field) const {
    VerifyNonMonostate<T>();
    const FieldIndex untyped = field.untyped();
    if (untyped.id_ >= fields_.size()) return nullptr;
    return std::get_if<T>(&fields_[untyped.id_]);
  }

  // Retrieves a mutable pointer to the value associated with `field`. Returns
  // `nullptr` if the field is unset or holds a different type than `T`.
  template <typename T>
  T* TryGet(TypedFieldIndex<T> field) {
    VerifyNonMonostate<T>();
    const FieldIndex untyped = field.untyped();
    if (untyped.id_ >= fields_.size()) return nullptr;
    return std::get_if<T>(&fields_[untyped.id_]);
  }

  // Returns the number of field entries currently tracked in the record.
  size_t size() const { return fields_.size(); }

  // Removes all fields from the record.
  void clear() { fields_.clear(); }

  void SetBool(FieldIndex field, bool value) { operator[](field) = value; }
  void SetInt32(FieldIndex field, int32_t value) { operator[](field) = value; }
  void SetUint32(FieldIndex field, uint32_t value) {
    operator[](field) = value;
  }
  void SetInt64(FieldIndex field, int64_t value) { operator[](field) = value; }
  void SetUint64(FieldIndex field, uint64_t value) {
    operator[](field) = value;
  }
  void SetDouble(FieldIndex field, double value) { operator[](field) = value; }
  void SetString(FieldIndex field, absl::string_view value) {
    operator[](field) = std::string(value);
  }
  void SetString(FieldIndex field, const char* value) {
    operator[](field) = std::string(value);
  }
  void SetString(FieldIndex field, std::string&& value) {
    operator[](field) = std::move(value);
  }

  void SetInt32Sequence(FieldIndex field, std::vector<int32_t> value) {
    operator[](field) = std::move(value);
  }
  void SetUint32Sequence(FieldIndex field, std::vector<uint32_t> value) {
    operator[](field) = std::move(value);
  }
  void SetInt64Sequence(FieldIndex field, std::vector<int64_t> value) {
    operator[](field) = std::move(value);
  }
  void SetUint64Sequence(FieldIndex field, std::vector<uint64_t> value) {
    operator[](field) = std::move(value);
  }
  void SetDoubleSequence(FieldIndex field, std::vector<double> value) {
    operator[](field) = std::move(value);
  }
  void SetStringSequence(FieldIndex field, std::vector<std::string> value) {
    operator[](field) = std::move(value);
  }

 private:
  template <typename T>
  static constexpr void VerifyNonMonostate() noexcept {
    static_assert(!std::is_same_v<absl::remove_cvref_t<T>, std::monostate>,
                  "std::monostate represents missing values and cannot be "
                  "queried as a field type.");
  }
  std::vector<FieldValue> fields_;
};

}  // namespace xprof::events_db

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_SCHEMA_H_
