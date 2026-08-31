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

#ifndef THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_ARROW_UTILS_H_
#define THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_ARROW_UTILS_H_

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "third_party/arrow/api.h"
#include "third_party/arrow/array/util.h"
#include "third_party/arrow/type_traits.h"
#include "third_party/arrow/util/bit_util.h"
#include "xprof/convert/events_db/schema.h"

namespace xprof::events_db::internal {

// Converts an `arrow::Status` to an `absl::Status`.
absl::Status ToAbslStatus(const arrow::Status& status);

// Converts an `absl::Status` to an `arrow::Status`.
arrow::Status ToArrowStatus(const absl::Status& status);

// Returns the number of bytes required to hold `num_bits` bits. Avoids overflow
// on large bit counts by avoiding `(num_bits + 7) / 8`.
constexpr uint64_t BytesForBits(uint64_t num_bits) {
  return (num_bits >> 3) + ((num_bits & 7u) != 0);
}

// Converts an `arrow::Result<T>` to an `absl::StatusOr<T>`.
template <typename T>
absl::StatusOr<T> ToAbslStatusOr(arrow::Result<T> result) {
  if (result.ok()) return std::move(result).ValueUnsafe();
  return ToAbslStatus(result.status());
}

// Converts an `absl::StatusOr<T>` to an `arrow::Result<T>`.
template <typename T>
arrow::Result<T> ToArrowResult(absl::StatusOr<T> status) {
  if (status.ok()) return std::move(*status);
  return ToArrowStatus(status.status());
}

// Type trait to identify `std::vector` types.
template <typename>
struct is_std_vector : std::false_type {};

template <typename... Args>
struct is_std_vector<std::vector<Args...>> : std::true_type {};

template <typename T>
constexpr bool is_std_vector_v = is_std_vector<T>::value;

template <typename>
constexpr bool always_false_v = false;

// Returns the corresponding Arrow DataType for a C++ type T supported by
// Events DB.
template <typename T>
std::shared_ptr<arrow::DataType> GetArrowType() {
  using CleanT = std::remove_cvref_t<T>;
  if constexpr (is_std_vector_v<CleanT>) {
    static_assert(
        !is_std_vector_v<std::remove_cvref_t<typename CleanT::value_type>>,
        "Nested vectors are not supported.");
  }
  if constexpr (std::is_same_v<CleanT, std::monostate>)
    return arrow::null();
  else if constexpr (std::is_same_v<CleanT, bool>)
    return arrow::boolean();
  else if constexpr (std::is_same_v<CleanT, int32_t>)
    return arrow::int32();
  else if constexpr (std::is_same_v<CleanT, uint32_t>)
    return arrow::uint32();
  else if constexpr (std::is_same_v<CleanT, int64_t>)
    return arrow::int64();
  else if constexpr (std::is_same_v<CleanT, uint64_t>)
    return arrow::uint64();
  else if constexpr (std::is_same_v<CleanT, double>)
    return arrow::float64();
  else if constexpr (std::is_same_v<CleanT, std::string>)
    return arrow::utf8();
  else if constexpr (is_std_vector_v<CleanT>)
    return arrow::list(GetArrowType<typename CleanT::value_type>());
  else
    static_assert(always_false_v<T>,
                  "Unsupported type for Arrow Parquet export.");
}

// Represents an in-memory column buffer for type `T` across a batch of records.
//
// Thread-safety:
// - `SetValue` and `SetNull` are safe to call concurrently from multiple
//   threads as long as each thread operates on a distinct row index (disjoint
//   indices). Concurrent writes to the same row index require external
//   synchronization.
// - `ToArrowArray` is NOT thread-safe with respect to concurrent `SetValue` or
//   `SetNull` calls; it must only be called after all writes to the column have
//   completed.
template <typename T>
class Column {
 public:
  static constexpr bool kIsBool = std::is_same_v<T, bool>;
  static_assert(!std::is_same_v<T, std::monostate>,
                "std::monostate is not supported.");

  Column(TypedFieldIndex<T> idx, absl::string_view name, uint64_t size)
      : field_index_(idx),
        name_(name),
        size_(size),
        values_(std::is_same_v<T, bool> ? BytesForBits(size) : size),
        null_bitmap_(BytesForBits(size)) {}

  Column(const Column&) = delete;
  Column& operator=(const Column&) = delete;
  Column(Column&&) = default;
  Column& operator=(Column&&) = default;

  // Returns the typed field index associated with this column.
  TypedFieldIndex<T> field_index() const { return field_index_; }

  // Returns the name of the column.
  absl::string_view name() const { return name_; }

  // Creates the Arrow Field schema descriptor for this column.
  std::shared_ptr<arrow::Field> ToArrowField() const {
    return arrow::field(name_, GetArrowType<T>(), /*nullable=*/true);
  }

  // Sets the value at `index` and marks the row as valid (non-null).
  // Thread-safe when called concurrently across distinct row indices.
  void SetValue(uint64_t index, T value) {
    DCHECK_LT(index, size_);
    const uint64_t byte_idx = index / 8;
    const uint8_t mask = static_cast<uint8_t>(1u << (index % 8));
    std::atomic_ref<uint8_t>(null_bitmap_[byte_idx])
        .fetch_or(mask, std::memory_order_relaxed);
    if constexpr (!kIsBool)
      values_[index] = std::move(value);
    else if (value)
      std::atomic_ref<uint8_t>(values_[byte_idx])
          .fetch_or(mask, std::memory_order_relaxed);
    else
      std::atomic_ref<uint8_t>(values_[byte_idx])
          .fetch_and(~mask, std::memory_order_relaxed);
  }

  // Marks the row at `index` as null. Thread-safe.
  void SetNull(uint64_t index) {
    DCHECK_LT(index, size_);
    const uint64_t byte_idx = index / 8;
    const uint8_t mask = static_cast<uint8_t>(1u << (index % 8));
    std::atomic_ref<uint8_t>(null_bitmap_[byte_idx])
        .fetch_and(~mask, std::memory_order_relaxed);
  }

  // Sets the value at `index` based on the value in the given `record`.
  //
  // If the field is not set in the `record`, the column value is marked as
  // null. Thread-safe when called concurrently across distinct row indices.
  void Set(Record& record, uint32_t index) {
    if (record.HasField(field_index_))
      SetValue(index, std::move(record[field_index_]));
    else
      SetNull(index);
  }

  // Converts the populated column data into an Arrow Array of length `count`.
  //
  // The returned `arrow::Array` may wrap the internal memory buffers of this
  // `Column` object without copying. Therefore, this `Column` instance must
  // outlive the returned `arrow::Array`.
  //
  // Not thread-safe with respect to concurrent `SetValue` or `SetNull` calls.
  // All writes for the batch must be finished before invoking this method.
  absl::StatusOr<std::shared_ptr<arrow::Array>> ToArrowArray(
      uint64_t count,
      arrow::MemoryPool* pool = arrow::default_memory_pool()) const {
    if (count >
        std::min(std::numeric_limits<int64_t>::max() / sizeof(T), size_)) {
      return absl::InvalidArgumentError(
          "Requested count exceeds column capacity.");
    }
    if constexpr (std::is_same_v<T, bool>) {
      std::shared_ptr<arrow::Buffer> null_buffer =
          arrow::Buffer::Wrap(null_bitmap_.data(), BytesForBits(count));
      std::shared_ptr<arrow::Buffer> values_buffer =
          arrow::Buffer::Wrap(values_.data(), BytesForBits(count));
      return arrow::MakeArray(arrow::ArrayData::Make(
          arrow::boolean(), count,
          {std::move(null_buffer), std::move(values_buffer)}));
    } else if constexpr (std::is_arithmetic_v<T>) {
      std::shared_ptr<arrow::Buffer> null_buffer =
          arrow::Buffer::Wrap(null_bitmap_.data(), BytesForBits(count));
      std::shared_ptr<arrow::Buffer> values_buffer = arrow::Buffer::Wrap(
          reinterpret_cast<const uint8_t*>(values_.data()), count * sizeof(T));
      return arrow::MakeArray(arrow::ArrayData::Make(
          GetArrowType<T>(), count,
          {std::move(null_buffer), std::move(values_buffer)}));
    } else if constexpr (std::is_same_v<T, std::string>) {
      arrow::StringBuilder builder(pool);
      RETURN_IF_ERROR(ToAbslStatus(builder.Reserve(count)));
      const uint8_t* null_data = null_bitmap_.data();
      for (uint64_t i = 0; i < count; ++i) {
        if (arrow::bit_util::GetBit(null_data, i)) {
          RETURN_IF_ERROR(ToAbslStatus(builder.Append(values_[i])));
        } else {
          RETURN_IF_ERROR(ToAbslStatus(builder.AppendNull()));
        }
      }
      std::shared_ptr<arrow::Array> array;
      RETURN_IF_ERROR(ToAbslStatus(builder.Finish(&array)));
      return array;
    } else if constexpr (is_std_vector_v<T>) {
      using ValueType = typename T::value_type;
      std::unique_ptr<arrow::ArrayBuilder> raw_builder;
      RETURN_IF_ERROR(ToAbslStatus(
          arrow::MakeBuilder(pool, GetArrowType<T>(), &raw_builder)));
      arrow::ListBuilder* list_builder =
          static_cast<arrow::ListBuilder*>(raw_builder.get());
      RETURN_IF_ERROR(ToAbslStatus(list_builder->Reserve(count)));

      auto append_items = [&](auto* val_builder) -> absl::Status {
        const uint8_t* null_data = null_bitmap_.data();
        for (uint64_t i = 0; i < count; ++i) {
          if (arrow::bit_util::GetBit(null_data, i)) {
            RETURN_IF_ERROR(ToAbslStatus(list_builder->Append()));
            for (const ValueType& item : values_[i]) {
              RETURN_IF_ERROR(ToAbslStatus(val_builder->Append(item)));
            }
          } else {
            RETURN_IF_ERROR(ToAbslStatus(list_builder->AppendNull()));
          }
        }
        return absl::OkStatus();
      };

      if constexpr (std::is_same_v<ValueType, bool>) {
        RETURN_IF_ERROR(append_items(static_cast<arrow::BooleanBuilder*>(
            list_builder->value_builder())));
      } else if constexpr (std::is_arithmetic_v<ValueType>) {
        using ArrowType = typename arrow::CTypeTraits<ValueType>::ArrowType;
        RETURN_IF_ERROR(
            append_items(static_cast<arrow::NumericBuilder<ArrowType>*>(
                list_builder->value_builder())));
      } else if constexpr (std::is_same_v<ValueType, std::string>) {
        RETURN_IF_ERROR(append_items(
            static_cast<arrow::StringBuilder*>(list_builder->value_builder())));
      } else {
        static_assert(
            always_false_v<ValueType>,
            "Unsupported nested vector element type for Arrow export.");
      }

      std::shared_ptr<arrow::Array> array;
      RETURN_IF_ERROR(ToAbslStatus(list_builder->Finish(&array)));
      return array;
    } else {
      static_assert(always_false_v<T>, "Unsupported type for Parquet export.");
    }
  }

 private:
  TypedFieldIndex<T> field_index_;
  std::string name_;
  uint64_t size_;
  std::vector<std::conditional_t<kIsBool, uint8_t, T>> values_;
  std::vector<uint8_t> null_bitmap_;
};

// Explicit deduction guide for Class Template Argument Deduction (CTAD):
template <typename T>
Column(TypedFieldIndex<T>, absl::string_view, uint64_t) -> Column<T>;

}  // namespace xprof::events_db::internal

#endif  // THIRD_PARTY_XPROF_CONVERT_EVENTS_DB_ARROW_UTILS_H_
