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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

namespace xprof::events_db {

FieldIndex Schema::RegisterFieldName(absl::string_view name) {
  absl::MutexLock lock(mutex_);
  auto it = id_by_name_.find(name);
  if (it != id_by_name_.end()) {
    return FieldIndex(it->second);
  }

  uint32_t new_id = static_cast<uint32_t>(name_by_id_.size());
  if (new_id >= max_field_count_) return FieldIndex();

  auto [inserted_it, inserted] = id_by_name_.emplace(std::string(name), new_id);
  name_by_id_.push_back(inserted_it->first);

  return FieldIndex(new_id);
}

std::optional<absl::string_view> Schema::GetFieldName(FieldIndex field) const {
  absl::ReaderMutexLock lock(mutex_);
  if (field.id_ >= name_by_id_.size()) return std::nullopt;
  return name_by_id_[field.id_];
}

std::optional<FieldIndex> Schema::LookupFieldIndex(
    absl::string_view name) const {
  absl::ReaderMutexLock lock(mutex_);
  auto it = id_by_name_.find(name);
  if (it == id_by_name_.end()) return std::nullopt;
  return FieldIndex(it->second);
}

uint32_t Schema::size() const {
  absl::ReaderMutexLock lock(mutex_);
  return static_cast<uint32_t>(name_by_id_.size());
}

bool Record::operator==(const Record& other) const {
  constexpr auto is_unset = [](const FieldValue& v) {
    return std::holds_alternative<std::monostate>(v);
  };
  const size_t min_size = std::min(fields_.size(), other.fields_.size());
  const std::vector<FieldValue>& longer =
      (fields_.size() > other.fields_.size()) ? fields_ : other.fields_;
  return std::all_of(longer.begin() + min_size, longer.end(), is_unset) &&
         std::equal(fields_.begin(), fields_.begin() + min_size,
                    other.fields_.begin());
}

FieldValue& Record::operator[](FieldIndex field) {
  DCHECK(field.is_valid());
  if (field.id_ >= fields_.size()) {
    fields_.resize(field.id_ + 1);
  }
  return fields_[field.id_];
}

const FieldValue& Record::operator[](FieldIndex field) const {
  DCHECK(field.is_valid());
  if (field.id_ >= fields_.size()) {
    // Use `absl::NoDestructor` because `FieldValue` (`std::variant`) contains
    // `std::string` and `std::vector`, which are not trivially destructible. A
    // raw static variable of a non-trivially-destructible type is forbidden in
    // Google C++ style to prevent static-destruction-order issues at program
    // exit.
    static const absl::NoDestructor<FieldValue> kNullValue(std::monostate{});
    return *kNullValue;
  }
  return fields_[field.id_];
}

}  // namespace xprof::events_db
