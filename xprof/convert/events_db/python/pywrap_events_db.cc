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

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "third_party/nanobind/include/nanobind/make_iterator.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/operators.h"
#include "third_party/nanobind/include/nanobind/stl/optional.h"
#include "third_party/nanobind/include/nanobind/stl/string.h"
#include "third_party/nanobind/include/nanobind/stl/string_view.h"
#include "third_party/nanobind/include/nanobind/stl/variant.h"
#include "xprof/convert/events_db/schema.h"

namespace nb = nanobind;
using nanobind::literals::operator""_a;

namespace xprof::events_db {

NB_MODULE(pywrap_events_db, m) {
  m.doc() = "Native C++ bindings for XProf Events DB Schema and Record";

  // Bind FieldIndex opaque token
  nb::class_<FieldIndex>(m, "FieldIndex",
                         "Opaque token representing a field. Wraps an integer "
                         "for O(1) hashing and lookup.")
      .def(nb::init<>())
      .def("is_valid", &FieldIndex::is_valid,
           "Returns true if the `FieldIndex` represents a valid registered "
           "field.")
      .def(nb::self == nb::self)
      .def(nb::self != nb::self)
      .def(nb::self < nb::self)
      .def(nb::self <= nb::self)
      .def(nb::self > nb::self)
      .def(nb::self >= nb::self)
      .def("__hash__",
           [](const FieldIndex& self) -> size_t {
             return absl::Hash<FieldIndex>{}(self);
           })
      .def("__repr__", [](const FieldIndex& self) {
        return self.is_valid() ? "FieldIndex(valid)" : "FieldIndex(invalid)";
      });

  // Bind Schema
  nb::class_<Schema>(
      m, "Schema",
      "Manager for the mapping between string names and indices. Thread-safe.")
      .def(nb::init<uint32_t>(),
           "max_field_count"_a = std::numeric_limits<uint32_t>::max(),
           "Constructs a Schema with an optional maximum number of fields.")
      .def(
          "register_field_name",
          [](Schema& self, std::string_view name) {
            return self.RegisterFieldName(name);
          },
          "name"_a,
          "Registers a name and returns its unique index. If already "
          "registered, returns the existing index. If `max_field_count` names "
          "have already been registered and the input name has not been "
          "registered before, returns an invalid index.")
      .def(
          "get_field_name",
          [](const Schema& self,
             FieldIndex field) -> std::optional<std::string_view> {
            std::optional<absl::string_view> name = self.GetFieldName(field);
            if (!name.has_value()) return std::nullopt;
            return std::string_view(name->data(), name->size());
          },
          "field"_a, "Resolves an index back to its name.")
      .def(
          "lookup_field_index",
          [](const Schema& self, std::string_view name) {
            return self.LookupFieldIndex(name);
          },
          "name"_a, "Looks up an existing field without registering it.")
      .def("__len__", &Schema::size)
      .def_prop_ro("size", &Schema::size,
                   "Returns the number of names currently registered.")
      .def_prop_ro(
          "max_field_count", &Schema::max_field_count,
          "Returns the maximum number of names that can be registered.");
}

}  // namespace xprof::events_db
